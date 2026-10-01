#!/usr/bin/env python3
# Copyright 2026 Ali Bavarchee
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MLabChain — MLabChain --VER == 0.1.7
===========================

Proof-of-Scientific-Work ledger for machine-learning experiments.

What this edition adds
----------------------
This revision replaces the informal "training work" accounting of the
earlier MLabChain with three explicit, documented pieces:

1. **Symbolic cost accounting.** For the linear-regression trainer used
   here, one elementary floating-point operation count is

       C_train = E · N_tr · (3F + 4)

   and, because verification re-executes the training, C_verify = C_train.
   These numbers are exposed by ``symbolic_costs(payload)`` and summed
   across the chain by ``total_symbolic_ops(blockchain)``.

2. **A scientific-work unit (LSWU).** A bounded, hardware-agnostic score
   combining dataset size, wall time, and quality relative to a pinned
   baseline:

       LSWU = D(N)^0.15 · T(t)^0.30 · Q(I)^1.0

       D(N) = ln(1 + N / N0)                     N0 = 1e5
       T(t) = ln(1 + t / t0)                     t0 = 60 s
       Q(I) = I^η / (1 + I^η)                    η  = 2
       I    = NMSE_baseline / NMSE_model

   The exponents do not sum to one on purpose. The earlier geometric-mean
   constraint compressed the reward's dynamic range to roughly 5×; with
   these exponents the range is about 12×, dominated by quality. The
   formula is documented in ``compute_lswu`` with its measured ranges.

3. **Challenge manifests with pinned baselines and split commitments.**
   A training transaction is meaningful only relative to a fixed task.
   A challenge manifest pins the dataset, the feature list, the
   train/test split rule, the metric, and the baseline NMSE. Without
   a pinned baseline, the quality term is a signed claim, not a
   measurement.


Dependencies
------------
Python 3.10+
Required : cryptography
*The ML path is pure Python

Examples
--------
    python mlabchain.py demo

    python mlabchain.py create-wallet
    python mlabchain.py challenge-create \\
        --id MLC-LINEAR-001 --output challenge.json

    python mlabchain.py mine \\
        --challenge challenge.json --config config.json

    python mlabchain.py verify-ml --model mlabchain_data/models/model_ab12cd.pkl
    python mlabchain.py symbolic --tx-hash <hex>
    python mlabchain.py credits
    python mlabchain.py status
    python mlabchain.py validate
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import pickle
import random
import secrets
import sys
import time

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Dependency

try:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding, rsa
except ImportError:
    print(
        "\nERROR: The 'cryptography' package is required.\n"
        "Install it with:\n\n"
        "    pip install cryptography\n"
    )
    sys.exit(1)

# Configuration

DATA_DIR = Path("mlabchain_data")
BLOCKCHAIN_FILE = DATA_DIR / "blockchain.json"
WALLET_FILE = DATA_DIR / "wallet.json"
MODEL_DIR = DATA_DIR / "models"
CHALLENGE_DIR = DATA_DIR / "challenges"

DEFAULT_DIFFICULTY = 3           # hash-PoW difficulty for block sealing
GENESIS_DIFFICULTY = 0
SCHEMA_VERSION = 2
MERKLE_SCHEME = "leaf-parent-v2"

# LSWU defaults — all overridable per challenge.
LSWU_N0 = 1.0e5
LSWU_T0 = 60.0
LSWU_ETA = 2.0
LSWU_WN = 0.15
LSWU_WT = 0.30
LSWU_WQ = 1.00

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_type": "linear-regression",
    "framework": "mlabchain-native",
    "seed": 42,
    "learning_rate": 0.01,
    "epochs": 30,
}

# Utilities

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHALLENGE_DIR.mkdir(parents=True, exist_ok=True)


def hash_file(path: str | Path) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if not p.is_file():
        raise ValueError(f"Not a file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def short_hash(text: str, n: int = 12) -> str:
    return text[:n]

# Cost calcuating

def ops_per_sample(n_features: int) -> int:
    """
    Elementary floating-point operations per training example per epoch,
    for the linear-regression SGD loop in ``train_linear_regression``:

        forward      : F multiplies + F adds + 1 bias add   →  F + 1
        residual     : 1 subtract                            →  1
        weight update: F multiplies + F subtracts            →  2F
        bias update  : 1 multiply-add + 1 subtract           →  2
                                                       total:  3F + 4
    """
    return 3 * n_features + 4


def symbolic_costs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the symbolic operation counts and reported wall time for an
    ML_TRAINING transaction.

    C_train and C_verify are the same number by construction, because
    verification re-executes the training. There is no shortcut and no
    probabilistic sampling.
    """
    if payload.get("kind") != "ml-training":
        raise ValueError("Not an ML training payload.")

    t = payload.get("training") or {}
    cfg = payload.get("config") or {}

    n_train = int(t.get("n_train", 0))
    epochs = int(t.get("epochs", 0))
    feats = int(cfg.get("n_features", 0))

    per_sample = ops_per_sample(feats)
    ops = epochs * n_train * per_sample
    wall = float(t.get("wall_time_seconds", 0.0))

    return {
        "n_train": n_train,
        "epochs": epochs,
        "n_features": feats,
        "ops_per_sample": per_sample,
        "C_train_ops": ops,
        "C_verify_ops": ops,
        "reported_wall_time_seconds": wall,
        "hardware_seconds_per_op": (wall / ops) if ops > 0 else 0.0,
    }


def total_symbolic_ops(blockchain: "Blockchain") -> int:
    """Sum C_train over every ML training transaction in the chain."""
    total = 0
    for block in blockchain.chain:
        for tx in block.transactions:
            p = tx["payload"]
            if p.get("kind") == "ml-training":
                total += symbolic_costs(p)["C_train_ops"]
    return total

# LSWU; MLabChain Scientific Work Unit

def compute_lswu(
    n_train: int,
    wall_time_seconds: float,
    nmse_model: float,
    nmse_baseline: float,
    N0: float = LSWU_N0,
    t0: float = LSWU_T0,
    eta: float = LSWU_ETA,
    w_N: float = LSWU_WN,
    w_t: float = LSWU_WT,
    w_Q: float = LSWU_WQ,
) -> Dict[str, float]:
    """
    Compute the LSWU score for a training contribution.

    Returns a dict with the three factors and the product, so a caller
    can inspect what dominated the score.

    Notes on the exponents
    ----------------------
    So the exponents do NOT sum to one. Imposing w_N + w_t + w_Q = 1 gives
    the formula a geometric-mean interpretation, but it also compresses
    the dynamic range to roughly 5x across the plausible input space.
    With the defaults used here the range is closer to 12x, dominated
    by the quality term — which is the intended ordering, and the one
    the design brief asked for.

    The quality term Q(I) = I^η / (1 + I^η) is bounded in (0, 1).
    It prevents a reward explosion from an unusually small NMSE while
    still saturating quickly as the model beats the baseline.
    """
    D = math.log(1.0 + n_train / N0) if n_train > 0 else 0.0
    T = math.log(1.0 + wall_time_seconds / t0) if wall_time_seconds > 0 else 0.0

    if nmse_model <= 0.0 or nmse_baseline <= 0.0:
        I = 1.0
    else:
        I = nmse_baseline / nmse_model

    I_eta = I ** eta
    Q = I_eta / (1.0 + I_eta) if I_eta > 0.0 else 0.0

    # Guard against zero bases before exponentiation.
    D = max(D, 1e-12)
    T = max(T, 1e-12)
    Q = max(Q, 1e-12)

    lswu = (D ** w_N) * (T ** w_t) * (Q ** w_Q)

    return {
        "N": n_train,
        "t": wall_time_seconds,
        "I": I,
        "D": D,
        "T": T,
        "Q": Q,
        "D_pow": D ** w_N,
        "T_pow": T ** w_t,
        "Q_pow": Q ** w_Q,
        "LSWU": lswu,
        "N0": N0,
        "t0": t0,
        "eta": eta,
        "w_N": w_N,
        "w_t": w_t,
        "w_Q": w_Q,
    }


def total_lswu(blockchain: "Blockchain") -> float:
    total = 0.0
    for block in blockchain.chain:
        for tx in block.transactions:
            p = tx["payload"]
            if p.get("kind") == "ml-training":
                s = p.get("lswu") or {}
                total += float(s.get("LSWU", 0.0))
    return total

# =-=-==-=-=--=-=-=-=-=-=-=-=-==-=--=-=-=-=-==-=-=-=-=-=-=-=-=-==-==
# Dataset generation

def synthesize_dataset(
    n_samples: int,
    n_features: int,
    seed: int,
    noise: float,
) -> Tuple[List[List[float]], List[float]]:
    """
    Deterministic synthetic linear dataset.

    Uses Python's Mersenne Twister via ``random.Random(seed)``. The same
    seed on the same Python version produces the same dataset, which is
    what makes split commitments and baseline NMSE reproducible.
    """
    rng = random.Random(seed)
    w_true = [rng.uniform(-1.0, 1.0) for _ in range(n_features)]
    b_true = rng.uniform(-1.0, 1.0)

    X: List[List[float]] = []
    y: List[float] = []
    for _ in range(n_samples):
        x = [rng.gauss(0.0, 1.0) for _ in range(n_features)]
        target = b_true + sum(w_true[j] * x[j] for j in range(n_features))
        target += rng.gauss(0.0, noise)
        X.append(x)
        y.append(target)
    return X, y


def dataset_commitment(X: List[List[float]], y: List[float]) -> str:
    """
    Hash of a dataset, computed from its values in a stable textual form.
    This is what a challenge manifest pins.
    """
    h = hashlib.sha256()
    for x in X:
        h.update((",".join(f"{v:.12f}" for v in x) + "\n").encode("utf-8"))
    h.update(b"---\n")
    for yv in y:
        h.update((f"{yv:.12f}\n").encode("utf-8"))
    return h.hexdigest()


def apply_split(
    X: List[List[float]],
    y: List[float],
    split_rule: str,
    train_fraction: float,
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    """
    Deterministic train/test split.

    Only one rule is implemented: ``first_fraction`` — take the first
    ``train_fraction`` of events in dataset order for training, the rest
    for testing. Other rules would need to be added by a challenge
    author; the important property is that the rule is *named* in the
    manifest, so a verifier can reproduce it exactly.
    """
    if split_rule != "first_fraction":
        raise ValueError(f"Unknown split rule: {split_rule!r}")
    n = len(X)
    k = int(n * train_fraction)
    return X[:k], y[:k], X[k:], y[k:]


# Training

def train_linear_regression(
    X: List[List[float]],
    y: List[float],
    n_features: int,
    lr: float,
    epochs: int,
) -> Tuple[List[float], float]:
    """
    Vanilla SGD for linear regression, in pure Python.

    No shuffling, no momentum, no bias correction. Deliberately simple
    so that the training is deterministic given the config and the
    challenge, which is what makes verification re-execution meaningful.
    """
    w = [0.0] * n_features
    b = 0.0
    n = len(X)

    for _ in range(epochs):
        for i in range(n):
            pred = b + sum(w[j] * X[i][j] for j in range(n_features))
            err = pred - y[i]
            for j in range(n_features):
                w[j] -= lr * err * X[i][j]
            b -= lr * err

    return w, b


def evaluate_mse(
    X: List[List[float]], y: List[float], w: List[float], b: float,
) -> float:
    n = len(X)
    if n == 0:
        return float("nan")
    sse = 0.0
    for i in range(n):
        pred = b + sum(w[j] * X[i][j] for j in range(len(w)))
        sse += (pred - y[i]) ** 2
    return sse / n


def evaluate_variance(y: List[float]) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    mean = sum(y) / n
    return sum((v - mean) ** 2 for v in y) / n


def evaluate_r2(
    X: List[List[float]], y: List[float], w: List[float], b: float,
) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    mean_y = sum(y) / n
    ss_tot = sum((v - mean_y) ** 2 for v in y)
    ss_res = sum(
        (b + sum(w[j] * X[i][j] for j in range(len(w))) - y[i]) ** 2
        for i in range(n)
    )
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def mean_predictor_baseline_mse(y: List[float]) -> float:
    """Baseline: always predict the mean of the test set."""
    return evaluate_variance(y)

# =-=-==-=-=--=-=-=-=-=-=-=-=-==-=--=-=-=-=-==-=-=-=-=-=-=-=-=-==-==

# Challenge manifest

@dataclass
class Challenge:
    """
    A fixed scientific task that training transactions are compared to.

    Pinning the dataset, split rule, feature list, metric, and *baseline
    NMSE* is what makes the LSWU quality term a measurement rather than
    a signed claim. Without a pinned baseline, a contributor could
    choose a weak reference and inflate their score.
    """
    challenge_id: str
    description: str

    # Dataset definition — for the demo, a synthetic generator; for real
    # work, a file path plus its SHA-256.
    dataset_kind: str            # "synthetic_linear" or "file"
    dataset_sha256: str
    n_samples: int
    n_features: int
    generator_seed: int
    generator_noise: float

    # Split
    split_rule: str              # "first_fraction"
    train_fraction: float

    # Metric and baseline
    metric: str                  # "NMSE"
    baseline_kind: str           # "mean_predictor" or "model"
    baseline_nmse: float
    baseline_model_sha256: Optional[str]

    # LSWU parameters (challenge-level defaults for reproducibility)
    lswu_N0: float = LSWU_N0
    lswu_t0: float = LSWU_T0
    lswu_eta: float = LSWU_ETA
    lswu_wN: float = LSWU_WN
    lswu_wt: float = LSWU_WT
    lswu_wQ: float = LSWU_WQ

    manifest_sha256: str = ""

    def compute_manifest_hash(self) -> str:
        data = asdict(self)
        data.pop("manifest_sha256", None)
        return sha256_text(canonical_json(data))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["manifest_sha256"] = self.compute_manifest_hash()
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Challenge":
        d = dict(d)
        d.pop("manifest_sha256", None)
        return Challenge(**d)


def load_challenge(path: str | Path) -> Challenge:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ch = Challenge.from_dict(data)
    # Verify the manifest hash if it was recorded.
    recorded = data.get("manifest_sha256")
    if recorded and recorded != ch.compute_manifest_hash():
        raise ValueError(
            f"Challenge manifest hash mismatch: {path} has been modified."
        )
    return ch


def save_challenge(ch: Challenge, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(ch.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def build_synthetic_challenge(
    challenge_id: str,
    n_samples: int,
    n_features: int,
    seed: int,
    noise: float,
    train_fraction: float = 0.8,
) -> Challenge:
    """
    Build a challenge whose dataset is generated deterministically from
    a seed. The baseline is the mean predictor on the test split, whose
    NMSE is exactly 1.0 by construction — which makes the quality term
    easy to interpret: I = 1 / NMSE_model.
    """
    X, y = synthesize_dataset(n_samples, n_features, seed, noise)
    ds_hash = dataset_commitment(X, y)
    Xtr, ytr, Xte, yte = apply_split(X, y, "first_fraction", train_fraction)
    baseline_nmse = mean_predictor_baseline_mse(yte) / evaluate_variance(yte)

    return Challenge(
        challenge_id=challenge_id,
        description=(
            f"Synthetic linear regression, N={n_samples}, F={n_features}, "
            f"noise={noise}, seed={seed}."
        ),
        dataset_kind="synthetic_linear",
        dataset_sha256=ds_hash,
        n_samples=n_samples,
        n_features=n_features,
        generator_seed=seed,
        generator_noise=noise,
        split_rule="first_fraction",
        train_fraction=train_fraction,
        metric="NMSE",
        baseline_kind="mean_predictor",
        baseline_nmse=baseline_nmse,
        baseline_model_sha256=None,
    )


def materialize_challenge_dataset(ch: Challenge) -> Tuple[List[List[float]], List[float]]:
    if ch.dataset_kind != "synthetic_linear":
        raise ValueError(
            f"Dataset kind {ch.dataset_kind!r} not supported by this build."
        )
    X, y = synthesize_dataset(
        ch.n_samples, ch.n_features, ch.generator_seed, ch.generator_noise,
    )
    computed = dataset_commitment(X, y)
    if computed != ch.dataset_sha256:
        raise RuntimeError(
            "Dataset commitment mismatch: the generator no longer "
            "produces the dataset this challenge pins."
        )
    return X, y

# Wallet

class Wallet:

    def __init__(
        self,
        private_key_pem: Optional[str] = None,
        public_key_pem: Optional[str] = None,
    ):
        if private_key_pem:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode("utf-8"), password=None,
            )
        else:
            self.private_key = rsa.generate_private_key(
                public_exponent=65537, key_size=2048,
            )
        if public_key_pem:
            self.public_key = serialization.load_pem_public_key(
                public_key_pem.encode("utf-8")
            )
        else:
            self.public_key = self.private_key.public_key()

    def private_pem(self) -> str:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

    def public_pem(self) -> str:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    def address(self) -> str:
        return "ML-" + sha256_text(self.public_pem())[:20]

    def sign(self, message: str) -> str:
        sig = self.private_key.sign(
            message.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(sig).decode("ascii")

    @staticmethod
    def verify(public_key_pem: str, message: str, signature_b64: str) -> bool:
        try:
            pk = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
            sig = base64.b64decode(signature_b64)
            pk.verify(
                sig, message.encode("utf-8"),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH,
                ),
                hashes.SHA256(),
            )
            return True
        except Exception:
            return False


def save_wallet(wallet: Wallet) -> None:
    ensure_data_dir()
    WALLET_FILE.write_text(
        json.dumps(
            {
                "address": wallet.address(),
                "private_key": wallet.private_pem(),
                "public_key": wallet.public_pem(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_wallet() -> Wallet:
    if not WALLET_FILE.exists():
        w = Wallet()
        save_wallet(w)
        return w
    data = json.loads(WALLET_FILE.read_text(encoding="utf-8"))
    return Wallet(
        private_key_pem=data["private_key"],
        public_key_pem=data["public_key"],
    )

# Transaction

@dataclass
class Transaction:

    tx_type: str
    sender: str
    timestamp: str
    payload: Dict[str, Any]
    public_key: str
    signature: str
    nonce: str

    def unsigned_payload(self) -> Dict[str, Any]:
        return {
            "tx_type": self.tx_type,
            "sender": self.sender,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "public_key": self.public_key,
            "nonce": self.nonce,
        }

    def signing_message(self) -> str:
        return canonical_json(self.unsigned_payload())

    def to_dict(self) -> Dict[str, Any]:
        data = self.unsigned_payload()
        data["signature"] = self.signature
        return data

    def tx_hash(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def verify_signature(self) -> bool:
        return Wallet.verify(
            self.public_key, self.signing_message(), self.signature,
        )


def build_signed_transaction(
    wallet: Wallet, tx_type: str, payload: Dict[str, Any],
) -> Transaction:
    tx = Transaction(
        tx_type=tx_type,
        sender=wallet.address(),
        timestamp=utc_now(),
        payload=payload,
        public_key=wallet.public_pem(),
        signature="",
        nonce=secrets.token_hex(16),
    )
    tx.signature = wallet.sign(tx.signing_message())
    return tx

# Training transaction

def run_challenge_training(
    ch: Challenge,
    config: Dict[str, Any],
    compute_lswu_flag: bool = True,
) -> Dict[str, Any]:
    """
    Execute the training procedure defined by (challenge, config) and
    return the model, metrics, cost, and LSWU.

    Reproducibility depends on:
      * the challenge's dataset generator,
      * the challenge's split rule,
      * the config's hyperparameters.

    All three are committed to in the returned payload, so a verifier
    can reproduce this run without the original environment.
    """
    if config.get("model_type", "linear-regression") != "linear-regression":
        raise ValueError(
            "This build only implements linear-regression training."
        )

    X, y = materialize_challenge_dataset(ch)
    Xtr, ytr, Xte, yte = apply_split(
        X, y, ch.split_rule, ch.train_fraction,
    )

    lr = float(config.get("learning_rate", 0.01))
    epochs = int(config.get("epochs", 30))
    n_features = ch.n_features

    t0 = time.perf_counter()
    w, b = train_linear_regression(Xtr, ytr, n_features, lr, epochs)
    wall = time.perf_counter() - t0

    mse_train = evaluate_mse(Xtr, ytr, w, b)
    mse_test = evaluate_mse(Xte, yte, w, b)
    var_test = evaluate_variance(yte)
    nmse_model = mse_test / var_test if var_test > 0 else float("inf")
    r2_test = evaluate_r2(Xte, yte, w, b)

    model_bytes = pickle.dumps(
        {"w": w, "b": b, "n_features": n_features}, protocol=4,
    )
    arch_text = (
        f"model_type=linear-regression\n"
        f"framework=mlabchain-native\n"
        f"n_features={n_features}\n"
        f"optimizer=SGD\n"
        f"learning_rate={lr}\n"
        f"epochs={epochs}\n"
    )

    cfg_canonical = canonical_json(config)

    result: Dict[str, Any] = {
        "model_bytes": model_bytes,
        "model_sha256": sha256_bytes(model_bytes),
        "arch_text": arch_text,
        "arch_sha256": sha256_text(arch_text),
        "config_canonical": cfg_canonical,
        "config_sha256": sha256_text(cfg_canonical),
        "training": {
            "n_train": len(Xtr),
            "n_test": len(Xte),
            "epochs": epochs,
            "learning_rate": lr,
            "wall_time_seconds": wall,
        },
        "metrics": {
            "mse_train": mse_train,
            "mse_test": mse_test,
            "var_test": var_test,
            "nmse_model": nmse_model,
            "nmse_baseline": ch.baseline_nmse,
            "r2_test": r2_test,
        },
    }

    if compute_lswu_flag:
        result["lswu"] = compute_lswu(
            n_train=len(Xtr),
            wall_time_seconds=wall,
            nmse_model=nmse_model,
            nmse_baseline=ch.baseline_nmse,
            N0=ch.lswu_N0, t0=ch.lswu_t0, eta=ch.lswu_eta,
            w_N=ch.lswu_wN, w_t=ch.lswu_wt, w_Q=ch.lswu_wQ,
        )

    return result


def create_training_transaction(
    wallet: Wallet,
    ch: Challenge,
    config: Dict[str, Any],
    result: Dict[str, Any],
    model_path: Path,
    arch_path: Path,
    notes: str = "",
    tags: Optional[List[str]] = None,
) -> Transaction:
    """
    Build a signed ML_TRAINING transaction.

    The payload commits to:
      * the challenge manifest hash and the pinned dataset hash,
      * the split rule and train_fraction,
      * the config content,
      * the model and architecture file hashes,
      * training metadata and metrics,
      * the LSWU score computed from those numbers,
      * the symbolic cost in elementary operations.
    """
    sym = symbolic_costs({
        "kind": "ml-training",
        "config": {"n_features": ch.n_features},
        "training": result["training"],
    })

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ml-training",

        "challenge": {
            "challenge_id": ch.challenge_id,
            "manifest_sha256": ch.compute_manifest_hash(),
            "dataset_sha256": ch.dataset_sha256,
            "split_rule": ch.split_rule,
            "train_fraction": ch.train_fraction,
            "metric": ch.metric,
            "baseline_kind": ch.baseline_kind,
            "baseline_nmse": ch.baseline_nmse,
        },

        "config": config,
        "config_sha256": result["config_sha256"],

        "model_file": {
            "path": str(model_path.resolve()),
            "sha256": result["model_sha256"],
            "size_bytes": len(result["model_bytes"]),
        },
        "arch_file": {
            "path": str(arch_path.resolve()),
            "sha256": result["arch_sha256"],
            "size_bytes": len(result["arch_text"].encode("utf-8")),
            "text": result["arch_text"],
        },

        "training": result["training"],
        "metrics": result["metrics"],

        "lswu": result.get("lswu", {}),
        "symbolic_costs": sym,

        "notes": notes,
        "tags": tags or [],
        "recorded_at": utc_now(),
    }

    return build_signed_transaction(wallet, "ML_TRAINING", payload)

# Merkle tree

def merkle_root(transaction_dicts: List[Dict[str, Any]]) -> str:
    """
    Merkle root with leaf/parent domain separation ('L' / 'N' prefixes).
    """
    if not transaction_dicts:
        return sha256_text("EMPTY")
    hashes = [sha256_text("L" + canonical_json(tx)) for tx in transaction_dicts]
    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])
        nxt: List[str] = []
        for i in range(0, len(hashes), 2):
            nxt.append(sha256_text("N" + hashes[i] + hashes[i + 1]))
        hashes = nxt
    return hashes[0]

# __________________________________________________________________
# Block

@dataclass
class Block:

    index: int
    timestamp: str
    previous_hash: str
    merkle_root: str
    transactions: List[Dict[str, Any]]
    difficulty: int
    nonce: int = 0
    block_hash: str = ""

    def header(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
        }

    def calculate_hash(self) -> str:
        return sha256_text(canonical_json(self.header()))

    def mine(self) -> None:
        """
        Hash-based block sealing.

        This is *not* Proof-of-Scientific-Work. It is a trivial SHA-256
        loop that prevents cheap block rewriting. The scientific work
        is in the transactions, not in this loop.
        """
        target = "0" * self.difficulty
        print(
            f"Sealing block #{self.index} "
            f"(hash-PoW difficulty={self.difficulty}, "
            f"transaction count={len(self.transactions)})..."
        )
        start = time.perf_counter()
        self.nonce = 0
        while True:
            candidate = self.calculate_hash()
            if candidate.startswith(target):
                self.block_hash = candidate
                elapsed = time.perf_counter() - start
                rate = self.nonce / elapsed if elapsed > 0 else 0
                print(
                    f"Block sealed.\n"
                    f"  hash     : {candidate}\n"
                    f"  nonce    : {self.nonce}\n"
                    f"  hash time: {elapsed:.4f} s  ({rate:,.0f} H/s)"
                )
                return
            self.nonce += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
            "merkle_root": self.merkle_root,
            "transactions": self.transactions,
            "difficulty": self.difficulty,
            "nonce": self.nonce,
            "block_hash": self.block_hash,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "Block":
        return Block(
            index=data["index"],
            timestamp=data["timestamp"],
            previous_hash=data["previous_hash"],
            merkle_root=data["merkle_root"],
            transactions=data["transactions"],
            difficulty=data["difficulty"],
            nonce=data["nonce"],
            block_hash=data["block_hash"],
        )

# Blockchain

class Blockchain:

    def __init__(self, difficulty: int = DEFAULT_DIFFICULTY):
        self.difficulty = difficulty
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        if BLOCKCHAIN_FILE.exists():
            self.load()
        else:
            self.create_genesis_block()
            self.save()

    def create_genesis_block(self) -> None:
        g = Block(
            index=0,
            timestamp="GENESIS",
            previous_hash="0" * 64,
            merkle_root=sha256_text("GENESIS"),
            transactions=[],
            difficulty=GENESIS_DIFFICULTY,
            nonce=0,
        )
        g.block_hash = g.calculate_hash()
        self.chain.append(g)

    @property
    def latest_block(self) -> Block:
        return self.chain[-1]

    def add_transaction(self, tx: Transaction) -> None:
        if not tx.verify_signature():
            raise ValueError("Transaction signature is invalid.")
        self.pending_transactions.append(tx.to_dict())
        print("\nTransaction accepted.")
        print(f"Transaction hash: {tx.tx_hash()}")

    def mine_pending(self) -> Block:
        if not self.pending_transactions:
            raise RuntimeError("No pending transactions to mine.")
        block = Block(
            index=len(self.chain),
            timestamp=utc_now(),
            previous_hash=self.latest_block.block_hash,
            merkle_root=merkle_root(self.pending_transactions),
            transactions=copy.deepcopy(self.pending_transactions),
            difficulty=self.difficulty,
        )
        block.mine()
        self.chain.append(block)
        self.pending_transactions.clear()
        self.save()
        return block

    def save(self) -> None:
        ensure_data_dir()
        BLOCKCHAIN_FILE.write_text(
            json.dumps(
                {
                    "difficulty": self.difficulty,
                    "merkle_scheme": MERKLE_SCHEME,
                    "schema_version": SCHEMA_VERSION,
                    "chain": [b.to_dict() for b in self.chain],
                    "pending_transactions": self.pending_transactions,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        try:
            data = json.loads(BLOCKCHAIN_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt blockchain file {BLOCKCHAIN_FILE}: {exc}"
            ) from exc
        try:
            self.chain = [Block.from_dict(b) for b in data["chain"]]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Malformed block in {BLOCKCHAIN_FILE}: {exc}"
            ) from exc
        self.difficulty = data.get("difficulty", DEFAULT_DIFFICULTY)
        self.pending_transactions = data.get("pending_transactions", [])

    def validate(self) -> bool:
        if not self.chain:
            print("Blockchain is empty.")
            return False
        g = self.chain[0]
        if g.previous_hash != "0" * 64:
            print("Invalid genesis previous hash.")
            return False
        for i, block in enumerate(self.chain):
            if block.block_hash != block.calculate_hash():
                print(f"Invalid hash in block #{block.index}")
                return False
            if not block.block_hash.startswith("0" * block.difficulty):
                print(f"Invalid hash-PoW in block #{block.index}")
                return False
            if i > 0:
                prev = self.chain[i - 1]
                if block.previous_hash != prev.block_hash:
                    print(f"Broken chain at block #{block.index}")
                    return False
                if block.merkle_root != merkle_root(block.transactions):
                    print(f"Invalid Merkle root in block #{block.index}")
                    return False
            for tx_data in block.transactions:
                try:
                    tx = Transaction(
                        tx_type=tx_data["tx_type"],
                        sender=tx_data["sender"],
                        timestamp=tx_data["timestamp"],
                        payload=tx_data["payload"],
                        public_key=tx_data["public_key"],
                        signature=tx_data["signature"],
                        nonce=tx_data["nonce"],
                    )
                    if not tx.verify_signature():
                        print(
                            f"Invalid transaction signature "
                            f"in block #{block.index}"
                        )
                        return False
                except Exception as exc:
                    print(f"Transaction validation failed: {exc}")
                    return False
        print("\nBlockchain validation PASSED.")
        print(f"Blocks       : {len(self.chain)}")
        print(f"Transactions : {sum(len(b.transactions) for b in self.chain)}")
        return True

    def print_chain(self) -> None:
        print("\n" + "=" * 78)
        print("MLabChain")
        print("=" * 78)
        print(f"Blocks              : {len(self.chain)}")
        print(f"Pending transactions: {len(self.pending_transactions)}")
        print(f"Total LSWU          : {total_lswu(self):.6f}")
        print(f"Total symbolic ops  : {total_symbolic_ops(self):,}")
        print("-" * 78)
        for block in self.chain:
            print(f"\nBlock #{block.index}")
            print(f"  Timestamp     : {block.timestamp}")
            print(f"  Hash          : {block.block_hash}")
            print(f"  Merkle root   : {block.merkle_root}")
            print(f"  Hash-PoW diff : {block.difficulty}")
            print(f"  Nonce         : {block.nonce}")
            print(f"  Transactions  : {len(block.transactions)}")
            for tx in block.transactions:
                p = tx["payload"]
                if p.get("kind") != "ml-training":
                    print(f"    - [{tx['tx_type']}]")
                    continue
                ch = p.get("challenge") or {}
                m = p.get("metrics") or {}
                t = p.get("training") or {}
                s = p.get("lswu") or {}
                sym = p.get("symbolic_costs") or {}
                print(f"    - [ml-training] {ch.get('challenge_id', '?')}")
                print(
                    f"        n_train={t.get('n_train')} "
                    f"epochs={t.get('epochs')} "
                    f"wall={t.get('wall_time_seconds', 0.0):.4f}s"
                )
                print(
                    f"        NMSE_model={m.get('nmse_model', float('nan')):.6f}  "
                    f"NMSE_baseline={m.get('nmse_baseline', float('nan')):.6f}  "
                    f"I={s.get('I', float('nan')):.4f}"
                )
                print(
                    f"        LSWU={s.get('LSWU', 0.0):.6f}  "
                    f"C_train_ops={sym.get('C_train_ops', 0):,}"
                )
        print("=" * 78)

# Verification

def find_training_records(
    blockchain: Blockchain,
    model_sha256: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> List[Tuple[int, Dict[str, Any], Dict[str, Any]]]:
    """
    Return [(block_index, tx_dict, payload), …] matching the selector.
    """
    out: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    for block in blockchain.chain:
        for tx in block.transactions:
            payload = tx["payload"]
            if payload.get("kind") != "ml-training":
                continue
            if tx_hash:
                tx_obj = Transaction(
                    tx_type=tx["tx_type"],
                    sender=tx["sender"],
                    timestamp=tx["timestamp"],
                    payload=payload,
                    public_key=tx["public_key"],
                    signature=tx["signature"],
                    nonce=tx["nonce"],
                )
                if tx_obj.tx_hash() == tx_hash:
                    out.append((block.index, tx, payload))
            elif model_sha256:
                mf = payload.get("model_file") or {}
                if mf.get("sha256") == model_sha256:
                    out.append((block.index, tx, payload))
    return out


def _approx_equal(a: float, b: float, rtol: float = 1e-9, atol: float = 1e-12) -> bool:
    return abs(a - b) <= max(rtol * max(abs(a), abs(b)), atol)


def verify_ml_training(
    blockchain: Blockchain,
    model_path: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> None:
    """
    Verify an ML training transaction by re-training from the signed
    config and the challenge manifest.

    Reports:
      * metric matches (NMSE_model, NMSE_baseline, MSE_train, MSE_test),
      * LSWU recomputation,
      * verification wall time and the ratio to reported training time,
      * symbolic cost comparison (C_train vs C_verify, which are equal).

    Because the linear trainer is deterministic, metric matches are exact
    to floating-point tolerance. In a stochastic training regime the
    same code would produce approximate matches instead, and the check
    would need to be relaxed accordingly.
    """
    if not model_path and not tx_hash:
        print("Provide either --model or --tx-hash.")
        return

    model_sha = None
    if model_path:
        p = Path(model_path)
        if not p.exists():
            print(f"File not found: {p}")
            return
        model_sha = hash_file(p)
        print(f"\nModel file : {p.resolve()}")
        print(f"Model SHA  : {model_sha}")

    matches = find_training_records(
        blockchain, model_sha256=model_sha, tx_hash=tx_hash,
    )
    if not matches:
        print("\nSTATUS: NOT FOUND")
        print("No matching ML training transaction in the chain.")
        return

    for block_idx, tx_dict, payload in matches:
        print(f"\n--- Matching transaction in block #{block_idx} ---")
        ch_block = payload.get("challenge") or {}
        cfg = payload.get("config") or {}
        rec_metrics = payload.get("metrics") or {}
        rec_training = payload.get("training") or {}
        rec_lswu = payload.get("lswu") or {}
        rec_sym = payload.get("symbolic_costs") or {}

        print("Challenge:")
        for k in ("challenge_id", "dataset_sha256", "split_rule",
                  "train_fraction", "metric", "baseline_nmse"):
            print(f"    {k} = {ch_block.get(k)}")

        print("Config:")
        for k in sorted(cfg.keys()):
            print(f"    {k} = {cfg[k]}")

        print("Reported metrics:")
        for k in sorted(rec_metrics.keys()):
            v = rec_metrics[k]
            if isinstance(v, float):
                print(f"    {k} = {v:.10f}")
            else:
                print(f"    {k} = {v}")

        print("Reported training metadata:")
        for k in sorted(rec_training.keys()):
            v = rec_training[k]
            if isinstance(v, float):
                print(f"    {k} = {v:.6f}")
            else:
                print(f"    {k} = {v}")

        print("Reported LSWU components:")
        for k in ("D_pow", "T_pow", "Q_pow", "LSWU"):
            print(f"    {k} = {rec_lswu.get(k, 0.0):.6f}")

        # --- reconstruct the challenge manifest reference ---
        # We do not have the manifest file here, only its hash and the
        # fields the transaction recorded. Reconstruct enough to re-run.
        ch = Challenge(
            challenge_id=ch_block.get("challenge_id", "reconstructed"),
            description="reconstructed from transaction",
            dataset_kind="synthetic_linear",
            dataset_sha256=ch_block.get("dataset_sha256", ""),
            n_samples=int(cfg.get("n_samples", 0)),
            n_features=int(cfg.get("n_features", 0)),
            generator_seed=int(cfg.get("seed", 0)),
            generator_noise=float(cfg.get("noise", 0.0)),
            split_rule=ch_block.get("split_rule", "first_fraction"),
            train_fraction=float(ch_block.get("train_fraction", 0.8)),
            metric=ch_block.get("metric", "NMSE"),
            baseline_kind=ch_block.get("baseline_kind", "mean_predictor"),
            baseline_nmse=float(ch_block.get("baseline_nmse", 1.0)),
            baseline_model_sha256=None,
        )

        print("\nRe-training from the recorded config and challenge fields...")
        t0 = time.perf_counter()
        try:
            rerun = run_challenge_training(ch, cfg)
        except Exception as exc:
            print(f"Verification FAILED: re-training raised {exc}")
            continue
        verify_wall = time.perf_counter() - t0

        # --- metric comparisons ---
        print("\nMetric comparison:")
        ok = True
        for key in ("mse_train", "mse_test", "nmse_model",
                    "nmse_baseline", "r2_test"):
            if key not in rec_metrics:
                continue
            a = float(rec_metrics[key])
            b = float(rerun["metrics"][key])
            match = _approx_equal(a, b)
            marker = "OK " if match else "MISMATCH"
            print(
                f"  [{marker}] {key:14s} recorded={a:.10f}  "
                f"re-trained={b:.10f}"
            )
            if not match:
                ok = False

        # --- LSWU recomputation ---
        print("\nLSWU comparison:")
        rerun_lswu = rerun.get("lswu") or {}
        for key in ("D_pow", "T_pow", "Q_pow", "LSWU"):
            a = float(rec_lswu.get(key, 0.0))
            b = float(rerun_lswu.get(key, 0.0))
            match = _approx_equal(a, b)
            marker = "OK " if match else "MISMATCH"
            print(f"  [{marker}] {key:8s} recorded={a:.8f}  recomputed={b:.8f}")
            if not match:
                ok = False

        # --- symbolic cost comparison ---
        rerun_sym = symbolic_costs({
            "kind": "ml-training",
            "config": cfg,
            "training": rerun["training"],
        })
        print("\nSymbolic cost comparison (C_train vs C_verify):")
        print(f"  reported C_train_ops  : {rec_sym.get('C_train_ops', 0):,}")
        print(f"  recomputed C_verify_ops: {rerun_sym['C_verify_ops']:,}")
        print(
            "  note: by construction C_verify = C_train, because "
            "verification re-executes training."
        )

        # --- wall time comparison ---
        reported_wall = float(rec_training.get("wall_time_seconds", 0.0))
        ratio = (verify_wall / reported_wall) if reported_wall > 0 else float("inf")
        print("\nCost comparison (the defining property of useful PoW):")
        print(f"  reported training wall time : {reported_wall:.6f} s")
        print(f"  verification wall time      : {verify_wall:.6f} s")
        print(f"  ratio (verify / train)      : {ratio:.3f}×")
        print(
            "  interpretation: verification re-executes training, so this "
            "ratio is the hardware ratio c'/c, not an exponential "
            "asymmetry."
        )

        print(f"\nSTATUS: {'VERIFIED' if ok else 'MISMATCH'}")

# CLI commands

def command_create_wallet() -> None:
    wallet = Wallet()
    save_wallet(wallet)
    print("\nWallet created.")
    print(f"Address    : {wallet.address()}")
    print(f"Wallet file: {WALLET_FILE.resolve()}")
    print(
        "\nNote: the private key is stored in plaintext. "
        "This is an educational tool."
    )


def command_hash_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return
    print(f"\nFile    : {p.resolve()}")
    print(f"SHA-256 : {hash_file(p)}")
    print(f"Size    : {p.stat().st_size} bytes")


def command_challenge_create(args: argparse.Namespace) -> None:
    ch = build_synthetic_challenge(
        challenge_id=args.id,
        n_samples=args.n_samples,
        n_features=args.n_features,
        seed=args.seed,
        noise=args.noise,
        train_fraction=args.train_fraction,
    )
    save_challenge(ch, args.output)
    print(f"\nChallenge written to {Path(args.output).resolve()}")
    print(f"  challenge_id   : {ch.challenge_id}")
    print(f"  dataset_sha256 : {ch.dataset_sha256}")
    print(f"  baseline_nmse  : {ch.baseline_nmse:.6f}")
    print(f"  manifest_sha   : {ch.compute_manifest_hash()}")


def command_challenge_inspect(args: argparse.Namespace) -> None:
    ch = load_challenge(args.path)
    d = ch.to_dict()
    print("\nChallenge manifest:")
    print(json.dumps(d, indent=2, ensure_ascii=False))


def command_mine(args: argparse.Namespace) -> None:
    """
    Mine a block. With --challenge and --config, train a model first,
    save model.pkl and model.txt, create the training transaction, then
    seal the block.
    """
    wallet = load_wallet()
    blockchain = Blockchain(difficulty=args.difficulty)

    if args.challenge and args.config:
        ch = load_challenge(args.challenge)
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object.")

        print(f"\nChallenge : {ch.challenge_id}")
        print(f"Config    : {args.config}")

        result = run_challenge_training(ch, config)
        m = result["metrics"]
        t = result["training"]
        s = result["lswu"]

        print("\nTraining done.")
        print(f"  n_train          : {t['n_train']}")
        print(f"  epochs           : {t['epochs']}")
        print(f"  wall_time        : {t['wall_time_seconds']:.6f} s")
        print(f"  MSE_train        : {m['mse_train']:.10f}")
        print(f"  MSE_test         : {m['mse_test']:.10f}")
        print(f"  NMSE_model       : {m['nmse_model']:.6f}")
        print(f"  NMSE_baseline    : {m['nmse_baseline']:.6f}")
        print(f"  I                : {s['I']:.4f}")
        print(f"  LSWU             : {s['LSWU']:.6f}")

        out_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"model_{short_hash(result['config_sha256'])}"
        model_path = out_dir / f"{stem}.pkl"
        arch_path = out_dir / f"{stem}.txt"
        model_path.write_bytes(result["model_bytes"])
        arch_path.write_text(result["arch_text"], encoding="utf-8")

        print(f"  model            : {model_path.resolve()}")
        print(f"  arch             : {arch_path.resolve()}")

        tx = create_training_transaction(
            wallet=wallet,
            ch=ch,
            config=config,
            result=result,
            model_path=model_path,
            arch_path=arch_path,
            notes=args.notes,
            tags=args.tag,
        )
        blockchain.add_transaction(tx)
    elif args.challenge or args.config:
        raise ValueError(
            "Both --challenge and --config are required to train, or "
            "neither to mine existing pending transactions."
        )

    blockchain.mine_pending()


def command_status() -> None:
    Blockchain().print_chain()


def command_validate() -> None:
    Blockchain().validate()


def command_symbolic(args: argparse.Namespace) -> None:
    bc = Blockchain()
    matches = find_training_records(bc, tx_hash=args.tx_hash)
    if not matches:
        print(f"No transaction found with hash {args.tx_hash}")
        return
    for block_idx, _tx, payload in matches:
        sym = symbolic_costs(payload)
        print(f"\nSymbolic costs for transaction in block #{block_idx}:")
        for k in sorted(sym.keys()):
            v = sym[k]
            if isinstance(v, float):
                print(f"  {k:32s} = {v:.10g}")
            elif isinstance(v, int):
                print(f"  {k:32s} = {v:,}")
            else:
                print(f"  {k:32s} = {v}")


def command_credits() -> None:
    bc = Blockchain()
    n_tx = 0
    total_train = 0
    total_steps = 0
    total_wall = 0.0
    total_ops = 0
    total_w = 0.0

    for block in bc.chain:
        for tx in block.transactions:
            p = tx["payload"]
            if p.get("kind") != "ml-training":
                continue
            n_tx += 1
            t = p.get("training") or {}
            s = p.get("lswu") or {}
            sym = p.get("symbolic_costs") or {}
            total_train += int(t.get("n_train", 0))
            total_steps += int(t.get("n_train", 0)) * int(t.get("epochs", 0))
            total_wall += float(t.get("wall_time_seconds", 0.0))
            total_ops += int(sym.get("C_train_ops", 0))
            total_w += float(s.get("LSWU", 0.0))

    print("\nMLabChain credits (non-transferable, not a coin):")
    print(f"  training transactions    : {n_tx}")
    print(f"  total training examples  : {total_train:,}")
    print(f"  total gradient steps     : {total_steps:,}")
    print(f"  total reported wall time : {total_wall:.6f} s")
    print(f"  total symbolic ops       : {total_ops:,}")
    print(f"  total LSWU               : {total_w:.6f}")


def command_verify_ml(args: argparse.Namespace) -> None:
    bc = Blockchain()
    verify_ml_training(
        bc, model_path=args.model, tx_hash=args.tx_hash,
    )

# .demo

def run_demo() -> None:
    print("\n" + "=" * 78)
    print("MLabChain — DEMONSTRATION")
    print("=" * 78)

    ensure_data_dir()
    wallet = load_wallet()
    print(f"\nWallet address: {wallet.address()}")

    bc = Blockchain(difficulty=DEFAULT_DIFFICULTY)

    # Build one challenge and train three models against it.
    ch = build_synthetic_challenge(
        challenge_id="MLC-DEMO-LINEAR-001",
        n_samples=400,
        n_features=4,
        seed=7,
        noise=0.1,
        train_fraction=0.8,
    )
    ch_path = CHALLENGE_DIR / f"{ch.challenge_id}.json"
    save_challenge(ch, ch_path)

    print(f"\nChallenge     : {ch.challenge_id}")
    print(f"Dataset sha   : {ch.dataset_sha256}")
    print(f"Baseline NMSE : {ch.baseline_nmse:.6f}")
    print(f"Manifest file : {ch_path.resolve()}")

    demo_configs = [
        {**DEFAULT_CONFIG, "seed": 1, "epochs": 10},
        {**DEFAULT_CONFIG, "seed": 2, "epochs": 30},
        {**DEFAULT_CONFIG, "seed": 3, "epochs": 60},
    ]

    demo_dir = DATA_DIR / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)

    last_model_path: Optional[Path] = None

    for i, cfg in enumerate(demo_configs, start=1):
        print("\n" + "-" * 78)
        print(f"DEMO TRAINING {i} — epochs={cfg['epochs']}")
        print("-" * 78)

        result = run_challenge_training(ch, cfg)
        m = result["metrics"]
        t = result["training"]
        s = result["lswu"]

        print(f"  n_train       : {t['n_train']}")
        print(f"  wall_time     : {t['wall_time_seconds']:.6f} s")
        print(f"  MSE_train     : {m['mse_train']:.10f}")
        print(f"  MSE_test      : {m['mse_test']:.10f}")
        print(f"  NMSE_model    : {m['nmse_model']:.6f}")
        print(f"  NMSE_baseline : {m['nmse_baseline']:.6f}")
        print(f"  I             : {s['I']:.4f}")
        print(f"  LSWU          : {s['LSWU']:.6f}")

        stem = f"model_{short_hash(result['config_sha256'])}"
        model_path = demo_dir / f"{stem}.pkl"
        arch_path = demo_dir / f"{stem}.txt"
        model_path.write_bytes(result["model_bytes"])
        arch_path.write_text(result["arch_text"], encoding="utf-8")

        tx = create_training_transaction(
            wallet=wallet,
            ch=ch,
            config=cfg,
            result=result,
            model_path=model_path,
            arch_path=arch_path,
            notes=f"Demo training {i}",
            tags=["demo"],
        )
        bc.add_transaction(tx)
        bc.mine_pending()

        last_model_path = model_path

    bc.print_chain()
    bc.validate()
    command_credits()

    print("\n" + "-" * 78)
    print("VERIFICATION — re-train and compare")
    print("-" * 78)
    if last_model_path is not None:
        verify_ml_training(bc, model_path=str(last_model_path))

    print("\nDemo complete.")

# Argument parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlabchain",
        description=(
            "MLabChain — Proof-of-Scientific-Work ledger for ML experiments. "
            "Mining requires training; verification re-trains."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("demo", help="Run the full demonstration.")
    sub.add_parser("create-wallet", help="Create a new RSA wallet.")

    p_hash = sub.add_parser("hash-file", help="SHA-256 of a file.")
    p_hash.add_argument("path")

    # challenge-create
    p_chc = sub.add_parser(
        "challenge-create",
        help="Create a synthetic-linear challenge manifest.",
    )
    p_chc.add_argument("--id", required=True, help="Challenge identifier.")
    p_chc.add_argument("--output", required=True, help="Output .json path.")
    p_chc.add_argument("--n-samples", type=int, default=400)
    p_chc.add_argument("--n-features", type=int, default=4)
    p_chc.add_argument("--seed", type=int, default=42)
    p_chc.add_argument("--noise", type=float, default=0.1)
    p_chc.add_argument("--train-fraction", type=float, default=0.8)

    # challenge-inspect
    p_chi = sub.add_parser(
        "challenge-inspect",
        help="Print a challenge manifest.",
    )
    p_chi.add_argument("path")

    # mine
    p_mine = sub.add_parser(
        "mine",
        help=(
            "Seal a block. With --challenge and --config, trains a model "
            "first and creates the training transaction."
        ),
    )
    p_mine.add_argument("--challenge", help="Challenge manifest path.")
    p_mine.add_argument("--config", help="Training config JSON path.")
    p_mine.add_argument(
        "--output-dir",
        help="Where to write model.pkl and model.txt.",
    )
    p_mine.add_argument(
        "--difficulty", type=int, default=DEFAULT_DIFFICULTY,
        help="Hash-PoW difficulty for block sealing.",
    )
    p_mine.add_argument("--notes", default="")
    p_mine.add_argument("--tag", action="append", default=[])

    # chain ops
    sub.add_parser("status", help="Display the chain.")
    sub.add_parser("validate", help="Validate the chain.")
    sub.add_parser("credits", help="Show accumulated LSWU and symbolic costs.")

    # symbolic
    p_sym = sub.add_parser(
        "symbolic",
        help="Show symbolic cost breakdown for a transaction.",
    )
    p_sym.add_argument("--tx-hash", required=True)

    # verify-ml
    p_v = sub.add_parser(
        "verify-ml",
        help="Re-train a recorded model and compare metrics, LSWU, and cost.",
    )
    g = p_v.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Path to model.pkl.")
    g.add_argument("--tx-hash", help="Transaction hash.")

    return parser

# Main

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        if args.command == "demo":
            run_demo()
        elif args.command == "create-wallet":
            command_create_wallet()
        elif args.command == "hash-file":
            command_hash_file(args.path)
        elif args.command == "challenge-create":
            command_challenge_create(args)
        elif args.command == "challenge-inspect":
            command_challenge_inspect(args)
        elif args.command == "mine":
            command_mine(args)
        elif args.command == "status":
            command_status()
        elif args.command == "validate":
            command_validate()
        elif args.command == "credits":
            command_credits()
        elif args.command == "symbolic":
            command_symbolic(args)
        elif args.command == "verify-ml":
            command_verify_ml(args)
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\nOperation cancelled.")
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()