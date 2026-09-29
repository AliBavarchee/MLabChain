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
MLabChain --VER == 0.0.9
=========

Proof-of-Training blockchain for machine-learning experiments.

The idea
--------
Bitcoin's "work" is a SHA-256 hash with leading zeros. It is deliberately
meaningless: a hash is a hash, nobody wants it, and that uselessness is
what makes the system trustless. MLabChain asks what happens if we replace
that meaningless work with *training an ML model*. Mining a block then
requires producing a real artifact — a model — and the block's proof is
the training metadata: config, model hash, architecture, training-set
size, wall time, validation MSE, and a model metric.

The consequence, stated plainly
-------------------------------
In Bitcoin, verification is O(1): one hash, one check. In MLabChain,
verification means **re-training with the same config and checking that
the outputs match**. Verification cost therefore scales with training
cost:

    verification cost  ≈  computation cost

This is the property the whole design is built around. It is also the
property that makes the scheme unsuitable as a production blockchain:
a decentralized network needs cheap verification, and cheap verification
is exactly what useful mining destroys. MLabChain is a demonstration of
that trade-off, not a proposal for a currency.

What this is *not*
------------------
  * Not a tradeable coin. No network, no exchange, no liquidity, no
    consensus. The "credit" the chain accumulates is a local, non-
    transferable counter equal to the training work it records.
  * Not a claim that ML training is verifiable in general. Real training
    is non-deterministic (random seeds, GPU scheduling, framework
    versions). MLabChain's demo training is deterministic on purpose,
    so verification is exact. The docstring of ``run_training`` marks
    this assumption clearly.
  * Not a replacement for MLOps tooling, model registries, or the
    academic literature on proof-of-learning.

What it does do
---------------
  * Trains a small linear-regression model deterministically from a
    ``config.json``.
  * Saves the model to ``model.pkl`` and the architecture description
    to ``model.txt``.
  * Records the training as a signed transaction: config content, model
    hash, arch hash, training-set size, wall time, MSE on train and
    test, R^2 on test.
  * Mines those transactions into a block with the same PoW / Merkle /
    RSA machinery used elsewhere in the LabChain family.
  * Lets anyone re-run the training and check the metrics, measuring
    the verification cost and comparing it to the reported training cost.

Examples
--------
    python mlabchain.py demo
    python mlabchain.py create-wallet

    python mlabchain.py mine --config config.json
    python mlabchain.py mine --config config.json --output-dir models/

    python mlabchain.py verify-ml --model models/model_ab12cd.pkl
    python mlabchain.py status
    python mlabchain.py validate
    python mlabchain.py credits
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import pickle
import random
import secrets
import sys
import time

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

DEFAULT_DIFFICULTY = 3
GENESIS_DIFFICULTY = 0

SCHEMA_VERSION = 1
MERKLE_SCHEME = "leaf-parent-v2"

DEFAULT_CONFIG: Dict[str, Any] = {
    "model_type": "linear-regression",
    "framework": "mlabchain-native",
    "n_samples": 400,
    "n_features": 4,
    "seed": 313,
    "noise": 0.1,
    "learning_rate": 0.01,
    "epochs": 30,
    "train_fraction": 0.8,
}

# Utilities

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def hash_file(path: str | Path) -> str:
    """Streaming SHA-256 of a file, 1 MiB chunks."""

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

# ML — deterministic training

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
    what makes the demo's training reproducible.
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


def train_linear_regression(
    X: List[List[float]],
    y: List[float],
    n_features: int,
    lr: float,
    epochs: int,
) -> Tuple[List[float], float]:
    """
    Vanilla SGD for linear regression in pure Python.

    No shuffling, no momentum, no bias correction. Deliberately simple
    so the training is deterministic and easy to re-run during
    verification.
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
    X: List[List[float]],
    y: List[float],
    w: List[float],
    b: float,
) -> float:
    n = len(X)
    if n == 0:
        return float("nan")
    sse = 0.0
    for i in range(n):
        pred = b + sum(w[j] * X[i][j] for j in range(len(w)))
        sse += (pred - y[i]) ** 2
    return sse / n


def evaluate_r2(
    X: List[List[float]],
    y: List[float],
    w: List[float],
    b: float,
) -> float:
    n = len(y)
    if n == 0:
        return float("nan")
    mean_y = sum(y) / n
    ss_tot = sum((yv - mean_y) ** 2 for yv in y)
    ss_res = sum(
        (b + sum(w[j] * X[i][j] for j in range(len(w))) - y[i]) ** 2
        for i in range(n)
    )
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train a model from a config. Deterministic.

    Returns a dict with:
      * ``model_bytes``       — pickled model weights
      * ``model_sha256``      — SHA-256 of those bytes
      * ``arch_text``         — human-readable architecture description
      * ``arch_sha256``       — SHA-256 of that text
      * ``metrics``           — mse_train, mse_test, r2_test
      * ``training_metadata`` — n_train, n_test, epochs, learning_rate,
                                wall_time_seconds
      * ``config_sha256``     — SHA-256 of the canonical config

    Determinism assumption
    ----------------------
    This implementation is deterministic given the config: same seed,
    same Python version, same result. Real ML training is *not*
    deterministic in general. The exactness of the verification below
    depends on this assumption, which the docstring marks honestly.
    """

    n_samples = int(config["n_samples"])
    n_features = int(config["n_features"])
    seed = int(config["seed"])
    noise = float(config.get("noise", 0.1))
    lr = float(config["learning_rate"])
    epochs = int(config["epochs"])
    train_fraction = float(config.get("train_fraction", 0.8))

    X, y = synthesize_dataset(n_samples, n_features, seed, noise)
    n_train = int(n_samples * train_fraction)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    t0 = time.perf_counter()
    w, b = train_linear_regression(X_train, y_train, n_features, lr, epochs)
    wall_time = time.perf_counter() - t0

    mse_train = evaluate_mse(X_train, y_train, w, b)
    mse_test = evaluate_mse(X_test, y_test, w, b)
    r2_test = evaluate_r2(X_test, y_test, w, b)

    model_bytes = pickle.dumps(
        {"w": w, "b": b, "n_features": n_features},
        protocol=4,
    )

    arch_text = (
        f"model_type=linear-regression\n"
        f"framework=mlabchain-native\n"
        f"n_features={n_features}\n"
        f"optimizer=SGD\n"
        f"learning_rate={lr}\n"
        f"epochs={epochs}\n"
    )

    config_canonical = canonical_json(config)

    return {
        "model_bytes": model_bytes,
        "model_sha256": sha256_bytes(model_bytes),
        "arch_text": arch_text,
        "arch_sha256": sha256_text(arch_text),
        "config_sha256": sha256_text(config_canonical),
        "metrics": {
            "mse_train": mse_train,
            "mse_test": mse_test,
            "r2_test": r2_test,
        },
        "training_metadata": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "epochs": epochs,
            "learning_rate": lr,
            "wall_time_seconds": wall_time,
        },
    }

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
        digest = hashlib.sha256(self.public_pem().encode("utf-8")).hexdigest()
        return "ML-" + digest[:20]

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

def create_training_transaction(
    wallet: Wallet,
    config: Dict[str, Any],
    training_result: Dict[str, Any],
    model_path: Path,
    arch_path: Path,
    notes: str = "",
    tags: Optional[List[str]] = None,
) -> Transaction:
    """
    Build a signed transaction describing one training run.

    The payload records:
      * the config content itself (signed, so it cannot be changed),
      * the config file's hash and path,
      * the model file's hash and path,
      * the architecture file's hash, path, and text content,
      * training metadata: n_train, n_test, epochs, learning rate,
        reported wall time in seconds,
      * metrics: mse_train, mse_test, r2_test.

    A verifier re-trains from the config content and checks that the
    metrics (and, in a deterministic world, the model bytes) match.
    """

    payload = {
        "schema_version": SCHEMA_VERSION,
        "kind": "ml-training",
        "framework": config.get("framework", "mlabchain-native"),
        "model_type": config.get("model_type", "linear-regression"),

        "config": config,
        "config_sha256": training_result["config_sha256"],
        "config_file": {
            "path": str(Path(model_path).parent / "config.json"),
        },

        "model_file": {
            "path": str(model_path.resolve()),
            "sha256": training_result["model_sha256"],
            "size_bytes": len(training_result["model_bytes"]),
        },

        "arch_file": {
            "path": str(arch_path.resolve()),
            "sha256": training_result["arch_sha256"],
            "size_bytes": len(training_result["arch_text"].encode("utf-8")),
            "text": training_result["arch_text"],
        },

        "training": training_result["training_metadata"],
        "metrics": training_result["metrics"],

        "notes": notes,
        "tags": tags or [],
        "recorded_at": utc_now(),
    }

    return build_signed_transaction(wallet, "ML_TRAINING", payload)


# Merkle tree

def merkle_root(transaction_dicts: List[Dict[str, Any]]) -> str:
    """
    Merkle root with leaf/parent domain separation ('L' / 'N' prefixes).

    Eliminates the leaf-vs-internal-node collision class that made the
    original Bitcoin merkle scheme ambiguous (CVE-2012-2459).
    """

    if not transaction_dicts:
        return sha256_text("EMPTY")

    hashes = [sha256_text("L" + canonical_json(tx)) for tx in transaction_dicts]

    while len(hashes) > 1:
        if len(hashes) % 2 != 0:
            hashes.append(hashes[-1])
        next_level: List[str] = []
        for i in range(0, len(hashes), 2):
            next_level.append(
                sha256_text("N" + hashes[i] + hashes[i + 1])
            )
        hashes = next_level

    return hashes[0]


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
        target = "0" * self.difficulty
        print(f"Mining block #{self.index} (hash difficulty={self.difficulty})...")
        start = time.perf_counter()
        self.nonce = 0
        while True:
            candidate = self.calculate_hash()
            if candidate.startswith(target):
                self.block_hash = candidate
                elapsed = time.perf_counter() - start
                rate = self.nonce / elapsed if elapsed > 0 else 0
                print(
                    f"Block mined.\n"
                    f"  hash    : {candidate}\n"
                    f"  nonce   : {self.nonce}\n"
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
        genesis = Block(
            index=0,
            timestamp="GENESIS",
            previous_hash="0" * 64,
            merkle_root=sha256_text("GENESIS"),
            transactions=[],
            difficulty=GENESIS_DIFFICULTY,
            nonce=0,
        )
        genesis.block_hash = genesis.calculate_hash()
        self.chain.append(genesis)

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

        genesis = self.chain[0]
        if genesis.previous_hash != "0" * 64:
            print("Invalid genesis previous hash.")
            return False

        for i, block in enumerate(self.chain):
            if block.block_hash != block.calculate_hash():
                print(f"Invalid hash in block #{block.index}")
                return False
            if not block.block_hash.startswith("0" * block.difficulty):
                print(f"Invalid Proof-of-Work in block #{block.index}")
                return False
            if i > 0:
                previous = self.chain[i - 1]
                if block.previous_hash != previous.block_hash:
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

    def training_credit(self) -> Dict[str, Any]:
        """
        Aggregate the training work recorded in the chain.

        'Credit' is a non-transferable counter equal to
        sum(n_train * epochs) over every ML_TRAINING transaction, plus
        the total reported wall time. It exists to make the "useful
        work" visible. It is not a coin, not a balance, and cannot be
        transferred or spent.
        """

        total_steps = 0
        total_wall = 0.0
        total_train = 0
        n_tx = 0

        for block in self.chain:
            for tx in block.transactions:
                p = tx["payload"]
                if p.get("kind") != "ml-training":
                    continue
                t = p.get("training", {})
                total_steps += int(t.get("n_train", 0)) * int(t.get("epochs", 0))
                total_wall += float(t.get("wall_time_seconds", 0.0))
                total_train += int(t.get("n_train", 0))
                n_tx += 1

        return {
            "training_transactions": n_tx,
            "total_training_examples": total_train,
            "total_gradient_steps": total_steps,
            "total_reported_wall_seconds": total_wall,
        }

    def print_chain(self) -> None:
        print("\n" + "=" * 78)
        print("MLabChain")
        print("=" * 78)
        print(f"Blocks              : {len(self.chain)}")
        print(f"Pending transactions: {len(self.pending_transactions)}")
        print(f"Merkle scheme       : {MERKLE_SCHEME}")
        print("-" * 78)

        for block in self.chain:
            print(f"\nBlock #{block.index}")
            print(f"  Timestamp     : {block.timestamp}")
            print(f"  Hash          : {block.block_hash}")
            print(f"  Previous hash : {block.previous_hash}")
            print(f"  Merkle root   : {block.merkle_root}")
            print(f"  Difficulty    : {block.difficulty}")
            print(f"  Nonce         : {block.nonce}")
            print(f"  Transactions  : {len(block.transactions)}")

            for tx in block.transactions:
                p = tx["payload"]
                if p.get("kind") == "ml-training":
                    mt = p.get("model_type", "?")
                    fw = p.get("framework", "?")
                    t = p.get("training", {})
                    m = p.get("metrics", {})
                    print(f"    - [ml-training] {fw} / {mt}")
                    print(
                        f"        n_train={t.get('n_train')} "
                        f"n_test={t.get('n_test')} "
                        f"epochs={t.get('epochs')}"
                    )
                    wall = t.get("wall_time_seconds")
                    if wall is not None:
                        print(f"        wall_time={wall:.4f} s")
                    print(
                        f"        mse_train={m.get('mse_train'):.6f}  "
                        f"mse_test={m.get('mse_test'):.6f}  "
                        f"r2_test={m.get('r2_test'):.6f}"
                    )
                else:
                    print(f"    - [{tx['tx_type']}]")

        print("=" * 78)

# ML verification

def find_training_records(
    blockchain: Blockchain,
    model_sha256: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> List[Tuple[int, Dict[str, Any]]]:
    """
    Return [(block_index, payload), …] matching the given selector.

    ``model_sha256`` matches the recorded model file hash.
    ``tx_hash`` matches the transaction hash.
    """

    out: List[Tuple[int, Dict[str, Any]]] = []

    for block in blockchain.chain:
        for tx in block.transactions:
            payload = tx["payload"]
            if payload.get("kind") != "ml-training":
                continue
            if tx_hash:
                # Recompute the transaction hash from the dict we have.
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
                    out.append((block.index, payload))
            elif model_sha256:
                mf = payload.get("model_file") or {}
                if mf.get("sha256") == model_sha256:
                    out.append((block.index, payload))

    return out


def _approx_equal(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def verify_ml_training(
    blockchain: Blockchain,
    model_path: Optional[str] = None,
    tx_hash: Optional[str] = None,
) -> None:
    """
    Verify an ML training transaction by re-training from its config.

    This is the *point* of MLabChain: verification is not a cheap hash
    check, it is the same computation the miner originally did. The
    function measures its own training time and reports it alongside
    the reported wall time from the transaction, making the
    ``verification cost ≈ computation cost`` relation visible.
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

    for block_idx, payload in matches:
        print(f"\n--- Matching transaction in block #{block_idx} ---")

        config = payload.get("config") or {}
        recorded_metrics = payload.get("metrics") or {}
        recorded_training = payload.get("training") or {}

        print("Recorded config:")
        for k in sorted(config.keys()):
            print(f"    {k} = {config[k]}")
        print("Recorded metrics:")
        for k in sorted(recorded_metrics.keys()):
            print(f"    {k} = {recorded_metrics[k]}")
        print("Recorded training metadata:")
        for k in sorted(recorded_training.keys()):
            v = recorded_training[k]
            if isinstance(v, float):
                print(f"    {k} = {v:.6f}")
            else:
                print(f"    {k} = {v}")

        # --- re-train ---
        print("\nRe-training from the recorded config...")
        t0 = time.perf_counter()
        try:
            rerun = run_training(config)
        except Exception as exc:
            print(f"Verification FAILED: re-training raised {exc}")
            continue
        verify_wall = time.perf_counter() - t0

        # --- compare metrics ---
        new_metrics = rerun["metrics"]
        ok = True
        for key in ("mse_train", "mse_test", "r2_test"):
            if key not in recorded_metrics:
                continue
            a = float(recorded_metrics[key])
            b = float(new_metrics[key])
            match = _approx_equal(a, b)
            marker = "OK " if match else "MISMATCH"
            print(
                f"  [{marker}] {key:10s}  recorded={a:.12f}  "
                f"re-trained={b:.12f}"
            )
            if not match:
                ok = False

        # --- compare model bytes ---
        if model_path:
            recomputed_model_sha = rerun["model_sha256"]
            recorded_model_sha = (payload.get("model_file") or {}).get("sha256")
            if recomputed_model_sha == recorded_model_sha:
                print(f"  [OK ] model bytes   match recorded SHA-256 exactly.")
            else:
                print(
                    f"  [WARN] model bytes differ from recorded SHA-256.\n"
                    f"         recorded : {recorded_model_sha}\n"
                    f"         rerun    : {recomputed_model_sha}\n"
                    f"         (This can happen across Python/pickle versions "
                    f"even when metrics match.)"
                )

        # --- cost comparison ---
        reported = float(recorded_training.get("wall_time_seconds", 0.0))
        ratio = (verify_wall / reported) if reported > 0 else float("inf")

        print("\nCost comparison (the point of the exercise):")
        print(f"  reported training wall time : {reported:.6f} s")
        print(f"  verification  wall time     : {verify_wall:.6f} s")
        print(f"  ratio (verify / train)      : {ratio:.3f}×")

        print(
            f"\nSTATUS: {'VERIFIED' if ok else 'MISMATCH'}"
        )

# .Demo

def run_demo() -> None:
    print("\n" + "=" * 78)
    print("MLabChain — DEMONSTRATION")
    print("=" * 78)

    ensure_data_dir()
    wallet = load_wallet()
    print(f"\nWallet address: {wallet.address()}")

    blockchain = Blockchain(difficulty=DEFAULT_DIFFICULTY)

    # Three small training runs with different configs so the chain has
    # something to show. Deliberately tiny, so the demo runs in a second.
    demo_configs = [
        {
            **DEFAULT_CONFIG,
            "n_samples": 300,
            "n_features": 3,
            "seed": 1,
            "epochs": 25,
        },
        {
            **DEFAULT_CONFIG,
            "n_samples": 400,
            "n_features": 4,
            "seed": 2,
            "epochs": 30,
        },
        {
            **DEFAULT_CONFIG,
            "n_samples": 500,
            "n_features": 5,
            "seed": 3,
            "epochs": 35,
        },
    ]

    for i, cfg in enumerate(demo_configs, start=1):
        print("\n" + "-" * 78)
        print(f"DEMO TRAINING {i} — n_samples={cfg['n_samples']} "
              f"n_features={cfg['n_features']} epochs={cfg['epochs']}")
        print("-" * 78)

        result = run_training(cfg)

        # Write files under mlabchain_data/demo/
        demo_dir = DATA_DIR / "demo"
        demo_dir.mkdir(parents=True, exist_ok=True)
        stem = f"model_{short_hash(result['config_sha256'])}"
        model_path = demo_dir / f"{stem}.pkl"
        arch_path = demo_dir / f"{stem}.txt"

        model_path.write_bytes(result["model_bytes"])
        arch_path.write_text(result["arch_text"], encoding="utf-8")

        # Also write the config for reference (this is the config that
        # a user would normally already have on disk).
        config_path = demo_dir / f"{stem}.config.json"
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        m = result["metrics"]
        t = result["training_metadata"]
        print(f"  mse_train = {m['mse_train']:.6f}")
        print(f"  mse_test  = {m['mse_test']:.6f}")
        print(f"  r2_test   = {m['r2_test']:.6f}")
        print(f"  wall_time = {t['wall_time_seconds']:.6f} s")
        print(f"  model     = {model_path}")
        print(f"  arch      = {arch_path}")

        tx = create_training_transaction(
            wallet=wallet,
            config=cfg,
            training_result=result,
            model_path=model_path,
            arch_path=arch_path,
            notes=f"Demo training {i}",
            tags=["demo", "linear-regression"],
        )
        blockchain.add_transaction(tx)
        blockchain.mine_pending()

    # --- report ---
    blockchain.print_chain()
    blockchain.validate()

    print("\n" + "-" * 78)
    print("CHAIN TRAINING CREDIT (non-transferable)")
    print("-" * 78)
    credit = blockchain.training_credit()
    for k in sorted(credit.keys()):
        v = credit[k]
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
        else:
            print(f"  {k} = {v}")

    # --- verify the last model by re-training ---
    print("\n" + "-" * 78)
    print("VERIFICATION — re-train and compare (this is the point)")
    print("-" * 78)
    last_cfg = demo_configs[-1]
    last_result = run_training(last_cfg)
    demo_dir = DATA_DIR / "demo"
    stem = f"model_{short_hash(last_result['config_sha256'])}"
    last_model = demo_dir / f"{stem}.pkl"

    verify_ml_training(blockchain, model_path=str(last_model))

    print("\nDemo complete.")

# CLI commands

def command_create_wallet() -> None:
    wallet = Wallet()
    save_wallet(wallet)
    print("\nWallet created.")
    print(f"Address    : {wallet.address()}")
    print(f"Wallet file: {WALLET_FILE.resolve()}")
    print(
        "\nNote: the private key is stored in plaintext. "
        "This is an educational tool — do not reuse this wallet "
        "for anything of value."
    )


def command_hash_file(path: str) -> None:
    p = Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return
    print(f"\nFile    : {p.resolve()}")
    print(f"SHA-256 : {hash_file(p)}")
    print(f"Size    : {p.stat().st_size} bytes")


def command_mine(args: argparse.Namespace) -> None:
    """
    Mine a block.

    If ``--config`` is given, train a model from it first, save the
    resulting model.pkl and model.txt, create the training transaction,
    then mine. Otherwise, mine any pending transactions.
    """

    wallet = load_wallet()
    blockchain = Blockchain(difficulty=args.difficulty)

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config not found: {cfg_path}")

        config = json.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("Config must be a JSON object.")

        print("\nTraining from config:")
        for k in sorted(config.keys()):
            print(f"    {k} = {config[k]}")

        result = run_training(config)

        m = result["metrics"]
        t = result["training_metadata"]
        print("\nTraining done.")
        print(f"  mse_train = {m['mse_train']:.6f}")
        print(f"  mse_test  = {m['mse_test']:.6f}")
        print(f"  r2_test   = {m['r2_test']:.6f}")
        print(f"  wall_time = {t['wall_time_seconds']:.6f} s")

        # Where to write model + arch?
        out_dir = Path(args.output_dir) if args.output_dir else MODEL_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"model_{short_hash(result['config_sha256'])}"

        model_path = out_dir / f"{stem}.pkl"
        arch_path = out_dir / f"{stem}.txt"

        model_path.write_bytes(result["model_bytes"])
        arch_path.write_text(result["arch_text"], encoding="utf-8")

        print(f"  model     = {model_path.resolve()}")
        print(f"  arch      = {arch_path.resolve()}")

        tx = create_training_transaction(
            wallet=wallet,
            config=config,
            training_result=result,
            model_path=model_path,
            arch_path=arch_path,
            notes=args.notes,
            tags=args.tag,
        )
        blockchain.add_transaction(tx)

    blockchain.mine_pending()


def command_status() -> None:
    Blockchain().print_chain()


def command_validate() -> None:
    Blockchain().validate()


def command_credits() -> None:
    bc = Blockchain()
    credit = bc.training_credit()
    print("\nChain training credit (non-transferable, not a coin):")
    for k in sorted(credit.keys()):
        v = credit[k]
        if isinstance(v, float):
            print(f"  {k} = {v:.6f}")
        else:
            print(f"  {k} = {v}")


def command_verify_ml(args: argparse.Namespace) -> None:
    bc = Blockchain()
    verify_ml_training(
        bc,
        model_path=args.model,
        tx_hash=args.tx_hash,
    )

# Argument parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlabchain",
        description=(
            "MLabChain — proof-of-training blockchain for ML experiments. "
            "Mining requires training; verification re-trains."
        ),
    )
    sub = parser.add_subparsers(dest="command")

    # demo
    sub.add_parser("demo", help="Run the full demonstration.")

    # create-wallet
    sub.add_parser("create-wallet", help="Create a new RSA wallet.")

    # hash-file
    p_hash = sub.add_parser("hash-file", help="SHA-256 of a file.")
    p_hash.add_argument("path")

    # mine
    p_mine = sub.add_parser(
        "mine",
        help=(
            "Mine a block. With --config, trains a model first, "
            "writes model.pkl and model.txt, and mines the training."
        ),
    )
    p_mine.add_argument(
        "--config",
        help="Path to a config.json describing the training run.",
    )
    p_mine.add_argument(
        "--output-dir",
        help="Where to write model.pkl and model.txt (default: mlabchain_data/models/).",
    )
    p_mine.add_argument(
        "--difficulty", type=int, default=DEFAULT_DIFFICULTY,
        help="SHA-256 PoW difficulty for the block.",
    )
    p_mine.add_argument("--notes", default="")
    p_mine.add_argument("--tag", action="append", default=[])

    # chain ops
    sub.add_parser("status", help="Display the chain.")
    sub.add_parser("validate", help="Validate the chain.")
    sub.add_parser("credits", help="Show accumulated training credit.")

    # verify-ml
    p_verify = sub.add_parser(
        "verify-ml",
        help=(
            "Re-train a recorded model from its config and compare "
            "metrics. Reports verification wall time versus reported "
            "training wall time."
        ),
    )
    g = p_verify.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="Path to a model.pkl to verify.")
    g.add_argument("--tx-hash", help="Transaction hash to verify.")

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
        elif args.command == "mine":
            command_mine(args)
        elif args.command == "status":
            command_status()
        elif args.command == "validate":
            command_validate()
        elif args.command == "credits":
            command_credits()
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