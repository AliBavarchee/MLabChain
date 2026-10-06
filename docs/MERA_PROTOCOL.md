# Mera protocol v0.2 (reference)

## Asset

| Parameter | Value |
|---|---:|
| Asset | MERA |
| Atomic units | 100,000,000 per MERA |
| Decimals | 8 |
| Maximum supply | 100,000,000 MERA |
| Minimum transfer fee | 1 atomic unit |
| Max ML reward / proof | 50 MERA |
| Consensus seal | SHA-256 block PoW (reference/devnet) |

The PoW nonce is only a block-sealing mechanism. It is not the source of the
monetary value. Monetary issuance comes from accepted ML-work certificates.

## Monetary proof

For a deterministic challenge, the C++ core uses fixed-point NMSE values:

`nmse_scaled = round(NMSE * 1e9)`

`baseline_scaled = round(baseline_NMSE * 1e9)`

The quality improvement factor is

`quality_ppm = clamp(floor((baseline_scaled - nmse_scaled) * 1e6 / baseline_scaled), 0, 1e6)`.

No reward is issued unless the recorded model beats the pinned baseline.

For the reference trainer,

`ops = epochs * n_train * (3*n_features + 4)`

and

`reward_atomic = min(50 MERA, floor(ops * quality_ppm * 100,000,000 /
(10,000,000 * 1,000,000)))`.

The computation uses integer arithmetic in the C++ state engine. This avoids
floating-point disagreement between nodes for the monetary transition.

## Why wall time is not money

Wall-clock time depends on hardware, load, thermal state, compiler behavior and
other environment details. Mera records wall time and LSWU for scientific
provenance, but a miner cannot increase issuance merely by claiming that a run
took longer.

## Transaction types

`TRANSFER`
: Moves MERA between account addresses with an account nonce and a fee.

`ML_WORK`
: Commits the challenge, dataset, model, architecture, operation count,
  fixed-point metric values, and reward. The transaction is signed by the
  contributor's Ed25519 key.

`FAUCET`
: Devnet-only mint used to bootstrap testing. It is disabled on non-devnet
  configurations.

## Verification boundary

The C++ core can independently verify signatures, account state transitions,
reward arithmetic, hashes, Merkle roots, block PoW, and supply limits.

The reference Python layer independently re-runs the deterministic linear
regression and checks the model and architecture hashes. This is stronger than
trusting a self-reported score, but it is not sufficient for a permissionless
mainnet because arbitrary ML training cannot be made cheap and deterministic
for every validator simply by storing an artifact hash.

A production Mera network should move ML verification into one of these
protocol primitives:

1. a deterministic ML VM with a consensus-defined instruction set and exact
   numerical semantics;
2. succinct proofs (SNARK/STARK/zkML-style) of the training computation;
3. an optimistic verification/fraud-proof system; or
4. a defined validator/verifier committee with economic penalties.

Until that layer exists, this repository should be treated as a reference
implementation/testnet rather than a production L1 security claim.
