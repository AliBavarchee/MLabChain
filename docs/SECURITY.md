# Security notes

This repository is a reference implementation and should be treated as a
research/testnet project.

## Fixed in v0.2

- Private keys are not stored as plaintext by the Python wallet; the private
  Ed25519 key is encrypted with AES-GCM under a PBKDF2-HMAC-SHA256 key.
- Account nonces prevent the same signed account transaction from being
  replayed in the same chain state.
- The C++ state engine checks sender/public-key binding and Ed25519 signatures.
- Monetary rewards use integer fixed-point NMSE and integer arithmetic.
- Maximum supply and per-proof reward limits are enforced by the C++ core.
- SQLite state transitions during block creation are atomic.
- Block hashes and Merkle roots are recomputed during validation.

## Still not solved

A model SHA-256 hash is not a proof that the model was honestly trained. The
reference verifier re-executes one deterministic linear trainer, but that is
not a general proof system for arbitrary ML programs.

The reference node also does not yet implement a production permissionless
P2P network, peer discovery, fork-choice, finality, anti-Sybil economics,
chain synchronization, authenticated block proposals, or network-level DoS
controls.

The ERC-20 contract is not a trustless bridge. Its mint authority is a trust
boundary until replaced with an audited bridge/issuer mechanism.

Do not use the devnet faucet on a public network. Do not reuse a development
wallet or development private key for anything of monetary value.
