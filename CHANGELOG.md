# Changelog

## Mera v0.2-devnet — 2026-10-02

- Split the system into a Python scientific layer and C++ monetary/state core.
- Replaced RSA wallet design with Ed25519.
- Encrypted wallet private keys with AES-GCM + PBKDF2-HMAC-SHA256.
- Replaced mutable JSON chain state with transactional SQLite state.
- Added account balances and nonces.
- Added signed `TRANSFER` transactions.
- Added signed `ML_WORK` transactions that create Mera only from deterministic
  work arithmetic and improvement over a pinned baseline.
- Moved monetary reward arithmetic to integer fixed-point values in C++.
- Kept LSWU and wall time as scientific/provenance metrics instead of direct
  monetary inputs.
- Added 100M MERA hard cap and 50 MERA per-ML-proof cap.
- Added devnet-only faucet for testing.
- Added replay-based chain validation and block/Merkle checks.
- Added an ERC-20-compatible Mera contract template with capped supply,
  burnability and permit support.

## Intentionally deferred

- Permissionless P2P gossip and peer discovery.
- Fork choice and finality.
- Trustless arbitrary-ML proof verification.
- Production bridge design.
- Exchange/DEX deployment and liquidity.
