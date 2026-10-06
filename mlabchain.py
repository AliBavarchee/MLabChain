#!/usr/bin/env python3
# Copyright 2026 Ali Bavarchee
# SPDX-License-Identifier: Apache-2.0
"""
MLabChain + Mera v0.2
=====================

Python remains the scientific/ML layer. The C++ core is authoritative for:
  * Ed25519 signature verification
  * account nonces and balances
  * Merkle roots / block hashes / PoW sealing
  * deterministic integer reward accounting
  * maximum-supply enforcement
  * persistent SQLite state

The important monetary rule is deliberately different from ordinary PoW:
Mera issuance is tied to a signed, reproducible ML-work certificate. Reported
wall time is retained as an audit metric, but is NOT used to mint Mera.

This build is a reference devnet/testnet implementation, not an exchange-
ready public L1. See README.md for the remaining hard problems, especially
fully decentralized verification of arbitrary ML training.
"""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import json
import math
import os
import pickle
import random
import secrets
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "mera_data"
WALLET_FILE = DATA_DIR / "wallet.json"
CHALLENGE_DIR = DATA_DIR / "challenges"
MODEL_DIR = DATA_DIR / "models"
PROOF_DIR = DATA_DIR / "proofs"

ATOMIC_PER_MERA = 100_000_000
MAX_SUPPLY_ATOMIC = 100_000_000 * ATOMIC_PER_MERA
OPS_PER_MERA = 10_000_000
MAX_WORK_REWARD_ATOMIC = 50 * ATOMIC_PER_MERA
DEFAULT_DIFFICULTY = 4
MIN_FEE_ATOMIC = 1
SCHEMA_VERSION = 3
WALLET_KDF_ITERATIONS = 600_000

LSWU_N0 = 1e5
LSWU_T0 = 60.0
LSWU_ETA = 2.0
LSWU_WN = 0.15
LSWU_WT = 0.30
LSWU_WQ = 1.0

DEFAULT_CONFIG = {
    "model_type": "linear-regression",
    "framework": "mlabchain-native",
    "seed": 42,
    "learning_rate": 0.01,
    "epochs": 30,
}


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_dirs() -> None:
    for p in (DATA_DIR, CHALLENGE_DIR, MODEL_DIR, PROOF_DIR):
        p.mkdir(parents=True, exist_ok=True)


def hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_from_mera(value: str) -> int:
    d = Decimal(value)
    if d <= 0:
        raise ValueError("amount must be positive")
    q = (d * ATOMIC_PER_MERA).quantize(Decimal("1"), rounding=ROUND_DOWN)
    return int(q)


def mera_from_atomic(value: int) -> str:
    return f"{Decimal(value) / Decimal(ATOMIC_PER_MERA):.8f}"


def core_path() -> Path:
    override = os.environ.get("MERA_CORE_PATH")
    if override:
        return Path(override)
    candidates = [ROOT / "mera_core", ROOT / "mera_core.exe", ROOT / "build" / "mera_core", ROOT / "build" / "Debug" / "mera_core.exe", ROOT / "build" / "Release" / "mera_core.exe"]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("mera_core executable not found; build cpp/mera_core.cpp first")


def core(*args: str, check: bool = True) -> dict[str, str]:
    cp = core_path()
    env = os.environ.copy()
    env.setdefault("MERA_NETWORK", "devnet")
    r = subprocess.run([str(cp), *args], cwd=ROOT, text=True, capture_output=True, env=env)
    if check and r.returncode != 0:
        msg = (r.stderr or r.stdout or "core command failed").strip()
        raise RuntimeError(msg)
    data: dict[str, str] = {}
    for line in (r.stdout or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k] = v
    return data


class Wallet:
    def __init__(self, private_key: Ed25519PrivateKey | None = None):
        self.private_key = private_key or Ed25519PrivateKey.generate()

    @property
    def public_bytes(self) -> bytes:
        return self.private_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    @property
    def public_hex(self) -> str:
        return self.public_bytes.hex()

    @property
    def address(self) -> str:
        return "MERA1" + sha256_bytes(self.public_bytes)[:40]

    def sign(self, message: str) -> str:
        return self.private_key.sign(message.encode("utf-8")).hex()

    def encrypted_dict(self, password: str) -> dict[str, Any]:
        salt = secrets.token_bytes(16)
        nonce = secrets.token_bytes(12)
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=WALLET_KDF_ITERATIONS)
        key = kdf.derive(password.encode("utf-8"))
        raw_private = self.private_key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        ciphertext = AESGCM(key).encrypt(nonce, raw_private, b"MERA-WALLET-V1")
        return {
            "version": 1,
            "address": self.address,
            "public_key": self.public_hex,
            "kdf": "PBKDF2-HMAC-SHA256",
            "iterations": WALLET_KDF_ITERATIONS,
            "salt": salt.hex(),
            "nonce": nonce.hex(),
            "ciphertext": ciphertext.hex(),
        }

    @staticmethod
    def from_file(path: Path, password: str) -> "Wallet":
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("version") != 1:
            raise ValueError("unsupported wallet version")
        salt = bytes.fromhex(d["salt"])
        nonce = bytes.fromhex(d["nonce"])
        kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=int(d["iterations"]))
        key = kdf.derive(password.encode("utf-8"))
        raw = AESGCM(key).decrypt(nonce, bytes.fromhex(d["ciphertext"]), b"MERA-WALLET-V1")
        w = Wallet(Ed25519PrivateKey.from_private_bytes(raw))
        if w.address != d["address"] or w.public_hex != d["public_key"]:
            raise ValueError("wallet integrity check failed")
        return w


def save_wallet(w: Wallet, password: str) -> None:
    ensure_dirs()
    WALLET_FILE.write_text(json.dumps(w.encrypted_dict(password), indent=2), encoding="utf-8")


def wallet_password(confirm: bool = False) -> str:
    env = os.environ.get("MERAWALLET_PASSWORD")
    if env is not None:
        return env
    p = getpass.getpass("Mera wallet password: ")
    if confirm:
        q = getpass.getpass("Repeat password: ")
        if p != q:
            raise ValueError("passwords do not match")
    return p


def load_wallet() -> Wallet:
    if not WALLET_FILE.exists():
        w = Wallet()
        save_wallet(w, wallet_password(confirm=True))
        return w
    return Wallet.from_file(WALLET_FILE, wallet_password())


@dataclass
class Challenge:
    challenge_id: str
    description: str
    dataset_kind: str
    dataset_sha256: str
    n_samples: int
    n_features: int
    generator_seed: int
    generator_noise: float
    split_rule: str
    train_fraction: float
    metric: str
    baseline_kind: str
    baseline_nmse: float
    baseline_model_sha256: str | None
    max_epochs: int = 1000
    verifier: str = "MLabChain-linear-v3"
    lswu_N0: float = LSWU_N0
    lswu_t0: float = LSWU_T0
    lswu_eta: float = LSWU_ETA
    lswu_wN: float = LSWU_WN
    lswu_wt: float = LSWU_WT
    lswu_wQ: float = LSWU_WQ
    schema_version: int = SCHEMA_VERSION
    manifest_sha256: str = ""

    def without_hash(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("manifest_sha256", None)
        return d

    def compute_hash(self) -> str:
        return sha256_text(canonical_json(self.without_hash()))

    def to_dict(self) -> dict[str, Any]:
        d = self.without_hash()
        d["manifest_sha256"] = self.compute_hash()
        return d


def synthesize_dataset(n_samples: int, n_features: int, seed: int, noise: float):
    rng = random.Random(seed)
    w_true = [rng.uniform(-1.0, 1.0) for _ in range(n_features)]
    b_true = rng.uniform(-1.0, 1.0)
    X: list[list[float]] = []
    y: list[float] = []
    for _ in range(n_samples):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        target = b_true + sum(w_true[j] * x[j] for j in range(n_features)) + rng.gauss(0.0, noise)
        X.append(x)
        y.append(target)
    return X, y


def dataset_commitment(X: list[list[float]], y: list[float]) -> str:
    h = hashlib.sha256()
    for x in X:
        h.update((",".join(f"{v:.12f}" for v in x) + "\n").encode())
    h.update(b"---\n")
    for v in y:
        h.update((f"{v:.12f}\n").encode())
    return h.hexdigest()


def split_dataset(X, y, fraction: float):
    k = int(len(X) * fraction)
    return X[:k], y[:k], X[k:], y[k:]


def train_linear_regression(X, y, n_features: int, lr: float, epochs: int):
    w = [0.0] * n_features
    b = 0.0
    for _ in range(epochs):
        for i, row in enumerate(X):
            pred = b + sum(w[j] * row[j] for j in range(n_features))
            err = pred - y[i]
            for j in range(n_features):
                w[j] -= lr * err * row[j]
            b -= lr * err
    return w, b


def mse(X, y, w, b):
    if not y:
        return math.nan
    return sum((b + sum(w[j] * X[i][j] for j in range(len(w))) - y[i]) ** 2 for i in range(len(y))) / len(y)


def variance(y):
    m = sum(y) / len(y)
    return sum((v - m) ** 2 for v in y) / len(y)


def r2(X, y, w, b):
    v = variance(y)
    return 1.0 - (mse(X, y, w, b) / v) if v > 0 else 0.0


def compute_lswu(n_train, wall_time_seconds, nmse_model, nmse_baseline, *, N0=LSWU_N0, t0=LSWU_T0, eta=LSWU_ETA, wN=LSWU_WN, wt=LSWU_WT, wQ=LSWU_WQ):
    D = max(math.log1p(n_train / N0), 1e-12)
    T = max(math.log1p(max(wall_time_seconds, 0.0) / t0), 1e-12)
    I = nmse_baseline / nmse_model if nmse_model > 0 and nmse_baseline > 0 else 1.0
    I_eta = I ** eta
    Q = max(I_eta / (1.0 + I_eta), 1e-12)
    return {"N": n_train, "t": wall_time_seconds, "I": I, "D": D, "T": T, "Q": Q, "LSWU": D ** wN * T ** wt * Q ** wQ,
            "N0": N0, "t0": t0, "eta": eta, "w_N": wN, "w_t": wt, "w_Q": wQ}


def ops_for(n_train: int, n_features: int, epochs: int) -> int:
    return int(epochs) * int(n_train) * (3 * int(n_features) + 4)


def work_reward_atomic(ops: int, nmse: float, baseline: float) -> int:
    nmse_s = max(1, int(round(nmse * 1_000_000_000)))
    base_s = max(1, int(round(baseline * 1_000_000_000)))
    if nmse_s >= base_s:
        return 0
    quality_ppm = min(1_000_000, ((base_s - nmse_s) * 1_000_000) // base_s)
    reward = (int(ops) * quality_ppm * ATOMIC_PER_MERA) // (OPS_PER_MERA * 1_000_000)
    return min(reward, MAX_WORK_REWARD_ATOMIC)


def build_challenge(args: argparse.Namespace) -> Challenge:
    X, y = synthesize_dataset(args.n_samples, args.n_features, args.seed, args.noise)
    Xtr, ytr, Xte, yte = split_dataset(X, y, args.train_fraction)
    _ = Xtr, ytr
    base_nmse = mse(Xte, yte, [0.0] * args.n_features, sum(yte) / len(yte)) / variance(yte)
    ch = Challenge(
        challenge_id=args.id,
        description=f"Deterministic synthetic linear regression, N={args.n_samples}, F={args.n_features}, noise={args.noise}, seed={args.seed}",
        dataset_kind="synthetic_linear",
        dataset_sha256=dataset_commitment(X, y),
        n_samples=args.n_samples,
        n_features=args.n_features,
        generator_seed=args.seed,
        generator_noise=args.noise,
        split_rule="first_fraction",
        train_fraction=args.train_fraction,
        metric="NMSE",
        baseline_kind="mean_predictor",
        baseline_nmse=base_nmse,
        baseline_model_sha256=None,
        max_epochs=args.max_epochs,
    )
    Path(args.output).write_text(json.dumps(ch.to_dict(), indent=2), encoding="utf-8")
    return ch


def load_challenge(path: str | Path) -> Challenge:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded = d.get("manifest_sha256")
    d.pop("manifest_sha256", None)
    # Compatible with v2 manifests, while writing v3 for new challenges.
    d.setdefault("max_epochs", 1000)
    d.setdefault("verifier", "MLabChain-linear-v3")
    d.setdefault("schema_version", SCHEMA_VERSION)
    ch = Challenge(**d)
    if recorded and recorded != ch.compute_hash():
        raise ValueError("challenge manifest hash mismatch")
    return ch


def materialize_challenge(ch: Challenge):
    if ch.dataset_kind != "synthetic_linear":
        raise ValueError("this reference build supports only synthetic_linear challenges")
    X, y = synthesize_dataset(ch.n_samples, ch.n_features, ch.generator_seed, ch.generator_noise)
    if dataset_commitment(X, y) != ch.dataset_sha256:
        raise RuntimeError("dataset commitment mismatch")
    return X, y


def run_training(ch: Challenge, config: dict[str, Any]) -> dict[str, Any]:
    if config.get("model_type", "linear-regression") != "linear-regression":
        raise ValueError("v0.2 supports deterministic linear-regression only")
    epochs = int(config.get("epochs", 30))
    if not 1 <= epochs <= ch.max_epochs:
        raise ValueError(f"epochs must be in [1,{ch.max_epochs}]")
    lr = float(config.get("learning_rate", 0.01))
    X, y = materialize_challenge(ch)
    Xtr, ytr, Xte, yte = split_dataset(X, y, ch.train_fraction)
    t0 = time.perf_counter()
    w, b = train_linear_regression(Xtr, ytr, ch.n_features, lr, epochs)
    wall = time.perf_counter() - t0
    m_train = mse(Xtr, ytr, w, b)
    m_test = mse(Xte, yte, w, b)
    v_test = variance(yte)
    nmse = m_test / v_test if v_test > 0 else math.inf
    arch = "\n".join([
        "model_type=linear-regression",
        "framework=mlabchain-native",
        f"n_features={ch.n_features}",
        "optimizer=SGD",
        f"learning_rate={lr}",
        f"epochs={epochs}",
        "deterministic=true",
        "verifier=MLabChain-linear-v3",
    ]) + "\n"
    model_bytes = pickle.dumps({"w": w, "b": b, "n_features": ch.n_features}, protocol=4)
    lswu = compute_lswu(len(Xtr), wall, nmse, ch.baseline_nmse, N0=ch.lswu_N0, t0=ch.lswu_t0, eta=ch.lswu_eta, wN=ch.lswu_wN, wt=ch.lswu_wt, wQ=ch.lswu_wQ)
    ops = ops_for(len(Xtr), ch.n_features, epochs)
    reward = work_reward_atomic(ops, nmse, ch.baseline_nmse)
    return {
        "model_bytes": model_bytes,
        "model_sha256": sha256_bytes(model_bytes),
        "arch_text": arch,
        "arch_sha256": sha256_text(arch),
        "config_sha256": sha256_text(canonical_json(config)),
        "training": {"n_train": len(Xtr), "n_test": len(Xte), "epochs": epochs, "learning_rate": lr, "wall_time_seconds": wall, "wall_ms": int(round(wall * 1000))},
        "metrics": {"mse_train": m_train, "mse_test": m_test, "var_test": v_test, "nmse_model": nmse, "nmse_baseline": ch.baseline_nmse, "r2_test": r2(Xte, yte, w, b)},
        "ops": ops,
        "reward_atomic": reward,
        "lswu": lswu,
    }


def transfer_message(sender, recipient, amount, fee, nonce, public_key):
    return f"TRANSFER|{sender}|{recipient}|{amount}|{fee}|{nonce}|{public_key}"


def ml_message(sender, challenge_id, manifest, dataset, model, arch, ops, n_train, n_features, epochs, nmse_scaled, baseline_scaled, reward, wall_ms, lswu_micro, nonce, public_key):
    return "|".join(["ML_WORK", sender, challenge_id, manifest, dataset, model, arch, str(ops), str(n_train), str(n_features), str(epochs), str(nmse_scaled), str(baseline_scaled), str(reward), str(wall_ms), str(lswu_micro), str(nonce), public_key])


def core_nonce(address: str) -> int:
    return int(core("nonce", "--address", address)["NONCE"])


def save_training_artifacts(result: dict[str, Any], challenge: Challenge, config: dict[str, Any]) -> tuple[Path, Path, Path]:
    ensure_dirs()
    stem = f"model_{result['model_sha256'][:16]}"
    model = MODEL_DIR / f"{stem}.pkl"
    arch = MODEL_DIR / f"{stem}.txt"
    proof = PROOF_DIR / f"{stem}.json"
    model.write_bytes(result["model_bytes"])
    arch.write_text(result["arch_text"], encoding="utf-8")
    proof.write_text(json.dumps({"schema_version": SCHEMA_VERSION, "challenge": challenge.to_dict(), "config": config, "result": {k:v for k,v in result.items() if k != "model_bytes"}}, indent=2), encoding="utf-8")
    return model, arch, proof


def command_create_wallet(_: argparse.Namespace) -> None:
    if WALLET_FILE.exists() and not _.force:
        raise RuntimeError(f"wallet already exists: {WALLET_FILE}; use --force to replace")
    w = Wallet()
    save_wallet(w, wallet_password(confirm=True))
    core("init")
    print(f"Wallet: {w.address}")
    print(f"Public key: {w.public_hex}")
    print(f"Encrypted file: {WALLET_FILE}")


def command_challenge_create(args: argparse.Namespace) -> None:
    ensure_dirs()
    ch = build_challenge(args)
    print(f"Challenge: {ch.challenge_id}")
    print(f"Manifest: {Path(args.output).resolve()}")
    print(f"Manifest SHA256: {ch.compute_hash()}")
    print(f"Dataset SHA256: {ch.dataset_sha256}")
    print(f"Baseline NMSE: {ch.baseline_nmse:.9f}")


def command_mine(args: argparse.Namespace) -> None:
    w = load_wallet()
    core("init")
    if not args.challenge or not args.config:
        raise ValueError("--challenge and --config are required for ML mining")
    ch = load_challenge(args.challenge)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    result = run_training(ch, config)
    model, arch, proof = save_training_artifacts(result, ch, config)
    m = result["metrics"]
    t = result["training"]
    l = result["lswu"]
    nonce = core_nonce(w.address) + 1
    nmse_scaled = max(1, int(round(m["nmse_model"] * 1_000_000_000)))
    baseline_scaled = max(1, int(round(m["nmse_baseline"] * 1_000_000_000)))
    lswu_micro = max(0, int(round(l["LSWU"] * 1_000_000)))
    msg = ml_message(w.address, ch.challenge_id, ch.compute_hash(), ch.dataset_sha256, result["model_sha256"], result["arch_sha256"], result["ops"], t["n_train"], ch.n_features, t["epochs"], nmse_scaled, baseline_scaled, result["reward_atomic"], t["wall_ms"], lswu_micro, nonce, w.public_hex)
    sig = w.sign(msg)
    out = core("submit-ml", "--sender", w.address, "--challenge-id", ch.challenge_id, "--manifest-sha256", ch.compute_hash(), "--dataset-sha256", ch.dataset_sha256, "--model-sha256", result["model_sha256"], "--arch-sha256", result["arch_sha256"], "--ops", str(result["ops"]), "--n-train", str(t["n_train"]), "--n-features", str(ch.n_features), "--epochs", str(t["epochs"]), "--nmse-scaled", str(nmse_scaled), "--baseline-scaled", str(baseline_scaled), "--work-reward", str(result["reward_atomic"]), "--wall-ms", str(t["wall_ms"]), "--lswu-micro", str(lswu_micro), "--nonce", str(nonce), "--public-key", w.public_hex, "--signature", sig, "--note", f"proof={proof.name}")
    # Mera monetary issuance is blocked until the C++ core has accepted the exact deterministic reward.
    mine = core("mine", "--difficulty", str(args.difficulty), "--producer", w.address)
    print("\nML work accepted and block sealed.")
    print(f"Challenge       : {ch.challenge_id}")
    print(f"Model SHA256    : {result['model_sha256']}")
    print(f"Verified ops    : {result['ops']:,}")
    print(f"NMSE            : {m['nmse_model']:.9f}")
    print(f"LSWU (audit)    : {l['LSWU']:.6f}")
    print(f"Mera reward     : {mera_from_atomic(result['reward_atomic'])} MERA")
    print(f"TXID            : {out.get('TXID','?')}")
    print(f"Block height    : {mine.get('HEIGHT','?')}")
    print(f"Block hash      : {mine.get('BLOCK_HASH','?')}")
    print(f"Model artifact  : {model}")
    print(f"Proof artifact  : {proof}")


def command_transfer(args: argparse.Namespace) -> None:
    w = load_wallet()
    amount = atomic_from_mera(args.amount)
    fee = atomic_from_mera(args.fee) if args.fee else MIN_FEE_ATOMIC
    nonce = core_nonce(w.address) + 1
    msg = transfer_message(w.address, args.to, amount, fee, nonce, w.public_hex)
    sig = w.sign(msg)
    out = core("submit-transfer", "--sender", w.address, "--recipient", args.to, "--amount", str(amount), "--fee", str(fee), "--nonce", str(nonce), "--public-key", w.public_hex, "--signature", sig, "--note", args.note)
    if args.mine:
        mine = core("mine", "--difficulty", str(args.difficulty), "--producer", w.address)
        print(f"Block: {mine.get('HEIGHT')} {mine.get('BLOCK_HASH')}")
    print(f"TXID: {out['TXID']}")


def command_faucet(args: argparse.Namespace) -> None:
    amount = atomic_from_mera(args.amount)
    core("faucet", "--recipient", args.address, "--amount", str(amount))
    out = core("mine", "--difficulty", str(args.difficulty), "--producer", args.address)
    print(f"Devnet mint: {mera_from_atomic(amount)} MERA")
    print(f"Block: {out['HEIGHT']} {out['BLOCK_HASH']}")


def command_balance(args: argparse.Namespace) -> None:
    a = args.address or load_wallet().address
    out = core("balance", "--address", a)
    print(f"{a}: {out['BALANCE_MERA']} MERA")


def command_status(_: argparse.Namespace) -> None:
    out = core("status")
    for k, v in out.items():
        print(f"{k}: {v}")


def command_validate(_: argparse.Namespace) -> None:
    out = core("validate")
    for k, v in out.items():
        print(f"{k}: {v}")


def command_show_tx(args: argparse.Namespace) -> None:
    out = core("show-tx", "--txid", args.tx_hash)
    for k, v in out.items(): print(f"{k}: {v}")


def command_verify_ml(args: argparse.Namespace) -> None:
    if not args.tx_hash:
        raise ValueError("--tx-hash is required")
    tx = core("show-tx", "--txid", args.tx_hash)
    ch_files = list(CHALLENGE_DIR.glob("*.json"))
    challenge_path = next((p for p in ch_files if json.loads(p.read_text(encoding="utf-8")).get("challenge_id") == tx.get("CHALLENGE_ID")), None)
    if challenge_path is None:
        raise RuntimeError(f"challenge manifest {tx.get('CHALLENGE_ID')} is not available locally")
    ch = load_challenge(challenge_path)
    cfg_path = None
    for p in PROOF_DIR.glob("*.json"):
        try:
            d=json.loads(p.read_text(encoding="utf-8"))
            if d.get("challenge",{}).get("challenge_id")==ch.challenge_id and d.get("result",{}).get("model_sha256")==tx.get("MODEL_SHA256"):
                cfg_path=p; break
        except Exception:
            pass
    if cfg_path is None:
        raise RuntimeError("matching local proof/config artifact not found")
    proof = json.loads(cfg_path.read_text(encoding="utf-8"))
    result = run_training(ch, proof["config"])
    recorded_model = MODEL_DIR / f"model_{tx['MODEL_SHA256'][:16]}.pkl"
    checks = {
        "manifest_hash": tx["MANIFEST_SHA256"] == ch.compute_hash(),
        "dataset_hash": tx["DATASET_SHA256"] == ch.dataset_sha256,
        "model_hash": result["model_sha256"] == tx["MODEL_SHA256"] and recorded_model.exists() and hash_file(recorded_model) == tx["MODEL_SHA256"],
        "arch_hash": result["arch_sha256"] == tx["ARCH_SHA256"],
        "ops": str(result["ops"]) == tx["OPS"],
        "reward": result["reward_atomic"] == int(tx["WORK_REWARD_ATOMIC"]),
        "nmse": int(round(result["metrics"]["nmse_model"] * 1_000_000_000)) == int(tx["NMSE_SCALED"]),
        "baseline": int(round(result["metrics"]["nmse_baseline"] * 1_000_000_000)) == int(tx["BASELINE_SCALED"]),
    }
    print("ML verification:")
    for k,v in checks.items(): print(f"  {'OK' if v else 'FAIL':4s} {k}")
    print(f"STATUS: {'VERIFIED' if all(checks.values()) else 'MISMATCH'}")


def command_demo(_: argparse.Namespace) -> None:
    ensure_dirs()
    if not WALLET_FILE.exists():
        w = Wallet(); save_wallet(w, os.environ.get("MERAWALLET_PASSWORD", "demo-password"))
        print(f"Created demo wallet: {w.address}")
    core("init")
    w = Wallet.from_file(WALLET_FILE, os.environ.get("MERAWALLET_PASSWORD", "demo-password"))
    # Fund devnet account once, then run a small deterministic ML contribution.
    if int(core("balance", "--address", w.address)["BALANCE_ATOMIC"]) == 0:
        core("faucet", "--recipient", w.address, "--amount", str(10 * ATOMIC_PER_MERA))
        core("mine", "--difficulty", "2", "--producer", w.address)
    ch = Challenge("MERA-DEMO-001", "Mera proof-of-useful-ML demonstration", "synthetic_linear", "", 200, 4, 7, 0.1, "first_fraction", 0.8, "NMSE", "mean_predictor", 1.0, None, 100)
    X,y=synthesize_dataset(ch.n_samples,ch.n_features,ch.generator_seed,ch.generator_noise)
    ch.dataset_sha256=dataset_commitment(X,y)
    _,_,Xte,yte=split_dataset(X,y,ch.train_fraction)
    ch.baseline_nmse=mse(Xte,yte,[0.0]*ch.n_features,sum(yte)/len(yte))/variance(yte)
    path=CHALLENGE_DIR/f"{ch.challenge_id}.json"; path.write_text(json.dumps(ch.to_dict(),indent=2),encoding="utf-8")
    cfg={**DEFAULT_CONFIG,"epochs":30}
    proof=run_training(ch,cfg)
    model,arch,pr=save_training_artifacts(proof,ch,cfg)
    nonce=core_nonce(w.address)+1
    ms=max(1,int(round(proof['metrics']['nmse_model']*1e9))); bs=max(1,int(round(proof['metrics']['nmse_baseline']*1e9)))
    lm=max(0,int(round(proof['lswu']['LSWU']*1e6)))
    sig=w.sign(ml_message(w.address,ch.challenge_id,ch.compute_hash(),ch.dataset_sha256,proof['model_sha256'],proof['arch_sha256'],proof['ops'],proof['training']['n_train'],ch.n_features,proof['training']['epochs'],ms,bs,proof['reward_atomic'],proof['training']['wall_ms'],lm,nonce,w.public_hex))
    core("submit-ml","--sender",w.address,"--challenge-id",ch.challenge_id,"--manifest-sha256",ch.compute_hash(),"--dataset-sha256",ch.dataset_sha256,"--model-sha256",proof['model_sha256'],"--arch-sha256",proof['arch_sha256'],"--ops",str(proof['ops']),"--n-train",str(proof['training']['n_train']),"--n-features",str(ch.n_features),"--epochs",str(proof['training']['epochs']),"--nmse-scaled",str(ms),"--baseline-scaled",str(bs),"--work-reward",str(proof['reward_atomic']),"--wall-ms",str(proof['training']['wall_ms']),"--lswu-micro",str(lm),"--nonce",str(nonce),"--public-key",w.public_hex,"--signature",sig,"--note",f"demo={pr.name}")
    core("mine","--difficulty","2","--producer",w.address)
    print("Demo complete.")
    command_status(argparse.Namespace())


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="mlabchain",description="MLabChain scientific-work ledger with Mera native asset")
    s=p.add_subparsers(dest="cmd",required=True)
    w=s.add_parser("create-wallet"); w.add_argument("--force",action="store_true"); w.set_defaults(func=command_create_wallet)
    c=s.add_parser("challenge-create"); c.add_argument("--id",required=True); c.add_argument("--output",required=True); c.add_argument("--n-samples",type=int,default=400); c.add_argument("--n-features",type=int,default=4); c.add_argument("--seed",type=int,default=42); c.add_argument("--noise",type=float,default=0.1); c.add_argument("--train-fraction",type=float,default=0.8); c.add_argument("--max-epochs",type=int,default=1000); c.set_defaults(func=command_challenge_create)
    m=s.add_parser("mine"); m.add_argument("--challenge",required=True); m.add_argument("--config",required=True); m.add_argument("--difficulty",type=int,default=DEFAULT_DIFFICULTY); m.set_defaults(func=command_mine)
    t=s.add_parser("transfer"); t.add_argument("--to",required=True); t.add_argument("--amount",required=True); t.add_argument("--fee",default=None); t.add_argument("--note",default=""); t.add_argument("--mine",action="store_true"); t.add_argument("--difficulty",type=int,default=DEFAULT_DIFFICULTY); t.set_defaults(func=command_transfer)
    f=s.add_parser("faucet"); f.add_argument("--address",required=True); f.add_argument("--amount",required=True); f.add_argument("--difficulty",type=int,default=2); f.set_defaults(func=command_faucet)
    b=s.add_parser("balance"); b.add_argument("--address"); b.set_defaults(func=command_balance)
    s0=s.add_parser("status"); s0.set_defaults(func=command_status)
    v=s.add_parser("validate"); v.set_defaults(func=command_validate)
    x=s.add_parser("show-tx"); x.add_argument("--tx-hash",required=True); x.set_defaults(func=command_show_tx)
    q=s.add_parser("verify-ml"); q.add_argument("--tx-hash",required=True); q.set_defaults(func=command_verify_ml)
    d=s.add_parser("demo"); d.set_defaults(func=command_demo)
    return p


def main():
    args=parser().parse_args()
    ensure_dirs()
    try: args.func(args)
    except KeyboardInterrupt: print("Cancelled.")
    except Exception as e: print(f"ERROR: {e}"); raise SystemExit(1)


if __name__ == "__main__": main()
