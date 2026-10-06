<p align="center">
  <img src="materials/mlabchain.png" alt="MLabChain" width="290"/>
</p>

<h1 align="center">MLabChain</h1>

<p align="center">
  <em>Proof-of-Scientific-Work ledger for machine-learning experiments.</em>
</p>

<p align="center">
  <img src="materials/MERA.png" alt="Mera" width="222"/>
  &nbsp;&nbsp;
  <strong>Mera</strong> — the unit MLabChain records.
</p>

# MLabChain + Mera v0.2

A hybrid Python/C++ implementation that upgrades the original local
MLabChain proof-of-scientific-work ledger into a transferable **MERA** asset
model.

The uploaded v0.1.7 code already committed challenge manifests, dataset/model/
architecture hashes, LSWU, symbolic operation counts, Ed25519-style signed
transactions, and Merkle-linked blocks. The original monetary accounting was
still explicitly non-transferable credits. The v0.2 redesign makes the state
machine monetary while keeping ML work as the origin of new Mera.

## Architecture

```text
                 ┌──────────────────────────────┐
                 │         mlabchain.py         │
                 │ Python scientific layer      │
                 │                              │
                 │ • deterministic ML trainer   │
                 │ • challenges/manifests       │
                 │ • artifact hashing            │
                 │ • LSWU scientific metric      │
                 │ • encrypted wallet            │
                 │ • transaction construction    │
                 └──────────────┬───────────────┘
                                │ command boundary
                                ▼
                 ┌──────────────────────────────┐
                 │          mera_core            │
                 │ C++ monetary/consensus core  │
                 │                              │
                 │ • Ed25519 verification       │
                 │ • SQLite state                │
                 │ • balances + nonces           │
                 │ • signed tx validation        │
                 │ • deterministic reward math   │
                 │ • Merkle roots                │
                 │ • SHA-256 block PoW           │
                 │ • supply-cap enforcement      │
                 └──────────────────────────────┘

                 optional exchange representation
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │ contracts/MeraScientific.sol │
                 │ ERC-20 settlement token      │
                 │ 8 decimals / 100M cap        │
                 └──────────────────────────────┘
```

## What is actually improved

### 1. Mera is transferable

Accounts now have balances and monotonically increasing nonces. `TRANSFER`
transactions are signed by the sender. The C++ state engine rejects invalid
signatures, stale nonces, insufficient balances, duplicate transactions and
supply-cap violations.

### 2. Mera issuance is tied to useful ML work

The previous LSWU score remains a scientific accounting metric. Monetary
issuance is a separate deterministic rule:

```text
ops = epochs × n_train × (3 × n_features + 4)

quality_ppm = clamp((baseline_NMSE - model_NMSE) / baseline_NMSE, 0, 1) × 1e6

reward = min(50 MERA,
             ops × quality_ppm × 1e8 / (1e7 × 1e6))
```

The C++ implementation actually uses integer fixed-point NMSE values, so the
reward transition is not dependent on cross-platform floating-point rounding.

### 3. Wall time is no longer a minting input

Wall time is recorded in the ML certificate because it is scientifically
interesting, but it does not create extra money merely because a miner reports
a slower run.

### 4. Supply is explicit

The reference monetary policy uses:

- maximum supply: **100,000,000 MERA**
- 8 decimal places
- 1 MERA = 100,000,000 atomic units
- max ML reward: 50 MERA per proof
- fees: burned in the reference state machine

These are protocol parameters, not a claim about future market value.

### 5. Better wallet security

The Python wallet uses Ed25519 and encrypts its private key with AES-GCM using a
PBKDF2-HMAC-SHA256 derived key. The original file stored its private key in
plaintext; v0.2 removes that default.

### 6. Persistent transactional state

The C++ layer uses SQLite rather than a single mutable JSON chain file. Block
insertion and account-state transitions occur inside a database transaction.

## Build

### Windows / PowerShell

Install CMake, a C++17 compiler, OpenSSL, SQLite3 and Boost headers (Boost.Multiprecision is header-only). With vcpkg, a practical
setup is to provide OpenSSL, SQLite3 and Boost to CMake via the vcpkg toolchain.

```powershell
.\build.ps1
```

The executable can also be built manually:

```powershell
cmake -S cpp -B build
cmake --build build --config Release
```

Then point Python at the resulting executable when necessary:

```powershell
$env:MERA_CORE_PATH = "$PWD\build\Release\mera_core.exe"
```

### Linux/macOS

```bash
./build.sh
```

## Demo

Set an environment password so the demo is non-interactive:

```powershell
$env:MERAWALLET_PASSWORD = "demo-password"
python .\mlabchain.py demo
```

The demo creates a devnet wallet, funds it with devnet-only Mera, generates a
challenge, trains a deterministic linear model, submits an ML-work transaction,
seals a block, and prints the resulting balance/state.

## Typical use

Create a wallet:

```bash
python mlabchain.py create-wallet
```

Create a scientific challenge:

```bash
python mlabchain.py challenge-create \
  --id MERA-LINEAR-001 \
  --output mera_data/challenges/MERA-LINEAR-001.json \
  --n-samples 400 \
  --n-features 4 \
  --seed 42 \
  --noise 0.1 \
  --train-fraction 0.8
```

Mine ML work:

```bash
python mlabchain.py mine \
  --challenge mera_data/challenges/MERA-LINEAR-001.json \
  --config config.json
```

Transfer Mera:

```bash
python mlabchain.py transfer \
  --to MERA1... \
  --amount 0.25 \
  --mine
```

Inspect and verify:

```bash
python mlabchain.py balance
python mlabchain.py status
python mlabchain.py validate
python mlabchain.py show-tx --tx-hash <txid>
python mlabchain.py verify-ml --tx-hash <txid>
```

## Tradability: what the code does and does not provide

A transferable native asset is necessary but not sufficient for external
trading. An asset becomes practically tradable only when there is a public
network, wallets, market infrastructure and a venue/pool willing to quote it.

The included `MeraScientific.sol` is an ERC-20 representation intended for an
EVM deployment. Once independently audited, deployed, and supplied with
liquidity, it is technically compatible with ERC-20 tooling. OpenZeppelin
provides the ERC-20 implementation and capped-supply extension used by this
contract.

**Important:** the contract is intentionally *not* presented as a trustless
bridge. Its mint authority must be replaced by an audited bridge, multisig,
DAO, or another explicit cross-chain issuance mechanism before production use.

## Remaining fundamental limitations

### Permissionless ML verification

A hash proves that bytes match a commitment; it does not prove that the bytes
were honestly produced by the claimed training process. The Python verifier can
re-run this deterministic linear-regression challenge, but general stochastic
PyTorch/TensorFlow/JAX workloads are not automatically consensus-safe.

### Distributed consensus

The reference C++ node is still a single-node state engine with local SQLite
persistence. It does not yet implement a permissionless P2P gossip layer,
fork-choice, validator set, finality, peer discovery, NAT handling, chain sync,
or DoS protection. Therefore it should not be treated as a public L1 merely
because it has signatures and block hashes.

### Economic attack surfaces

Challenge selection, task difficulty, duplicated work, overfitting to public
benchmarks, data leakage, hardware asymmetry, and sybil identities all need
explicit protocol rules before a public monetary network can safely pay for
ML work.

### Token symbol collision

`MERA` is not a globally unique ticker. Current web results include unrelated
projects/tokens using the MERA name/symbol, so a public listing should use a
clearly disambiguated project identity and verify ticker availability before
deployment.

## Recommended next protocol stage

The strongest next step is **Mera v0.3: a multi-node testnet** with a
consensus-defined deterministic ML VM, a real P2P layer, block proposal and
fork-choice rules, peer synchronization, signed blocks, checkpoint/finality,
and replayable ML proofs. Only after that should an ERC-20 bridge or exchange
integration be considered production infrastructure.
