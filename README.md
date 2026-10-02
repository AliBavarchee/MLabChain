<p align="center">
  <img src="matterials/mlabchain.png" alt="MLabChain" width="222"/>
</p>

<h1 align="center">MLabChain</h1>

<p align="center">
  <em>Proof-of-Scientific-Work ledger for machine-learning experiments.</em>
</p>

<p align="center">
  <img src="matterials/MERA.png" alt="Mera" width="111"/>
  &nbsp;&nbsp;
  <strong>Mera</strong> — the unit MLabChain records.
</p>

---

MLabChain currently is a small, local blockchain that records the provenance of
machine-learning experiments. Every training run becomes a signed
transaction containing the config, the model, the architecture, the
training metadata, and the quality metrics. Verification means
re-training from the recorded config and checking that the numbers
match.

That is the whole idea. One person, one laptop, no network, no GPU
required, no service to depend on.

---

## What Mera is

Mera (plural: Mera) is the unit MLabChain records. One Mera is one unit
of **verified scientific work**: training a model against a pinned
challenge, on a pinned dataset, with a pinned split, and beating a
pinned baseline by a documented amount.

Mera is computed from three things:

- **how much data** was used,
- **how much wall time** the training actually consumed,
- **how much better** the resulting model is than the challenge's
  baseline.

Each of these enters through a bounded, hardware-agnostic formula
(LabChain Scientific Work Unit, LSWU):

```
Mera =  D(N)^0.15  ·  T(t)^0.30  ·  Q(I)^1.00
```

with

```
D(N) = ln(1 + N / N0)                N0 = 100 000
T(t) = ln(1 + t / t0)                t0 = 60 seconds
Q(I) = I^η / (1 + I^η)               η  = 2
I    = NMSE_baseline / NMSE_model
```

The exponents do **not** sum to one. That is deliberate: summing to
one gives a geometric mean, which compresses the reward's dynamic range
to about 5× across the plausible input space. The version above spans
about 12× and is dominated by the quality term, which is the intended
ordering.

What Mera *is*: a way to say "I did this much verified scientific
training work, against this challenge, with this result, and here is
the signed record." That is worth something to you. It is not worth
anything to a market, and it should never be described as if it were.

If you came here looking for an investment opportunity, this is not
one, and the rest of the README will be more useful to you if you close
the tab now.

---

## The actual idea

Bitcoin burns electricity to compute hashes with leading zeros. The
hashes are deliberately meaningless — that is what makes the system
trustless, because nobody wants a hash and so nobody will produce one
except to mine.

MLabChain asks a different question. What if the work you did to earn
the reward *was itself the useful thing*? What if mining a block meant
training a model, and the proof of that work was the model?

The answer, developed in detail across the project's documentation, is:
you can make mining produce something useful, or you can make mining
secure and decentralised, but so far nobody has made it do both. The
properties that make Bitcoin work — expensive to produce, cheap to
verify, worthless to anyone but the miner — are exactly the properties
useful mining has to give up.

MLabChain sits on the "useful" side of that line. It is not a
production blockchain. It is a **provenance ledger**: it records what
you trained, when, against what, and with what result, in a form you
can verify later by re-running the training and checking that the
numbers agree.

The trade-off is measured and reported: `verify-ml` prints the ratio
between the verification wall time and the reported training wall time.
For deterministic training on the same hardware, that ratio is about 1.
For Bitcoin, the equivalent ratio is about 2^20. That difference is
not a bug — it is the definition of useful work.

---

## Install

Python 3.10 or newer.

```
pip install cryptography
```

That is the only required dependency. Everything else — hashing,
Merkle trees, block sealing, JSON persistence, the CLI, and the ML
trainer — is standard library.

---

## Try it

```
python mlabchain.py demo
```

This creates a challenge, trains three models against it, records each
training as a signed transaction, seals three blocks, validates the
chain, prints the accumulated Mera, and verifies the last model by
re-training and comparing metrics and cost.

Takes about two seconds.

Then:

```
python mlabchain.py status
python mlabchain.py validate
python mlabchain.py credits
```

---

## Real usage

Create a wallet once:

```
python mlabchain.py create-wallet
```

Create a challenge:

```
python mlabchain.py challenge-create \
    --id MLC-LINEAR-001 \
    --output challenge.json \
    --n-samples 1000 \
    --n-features 5 \
    --seed 42
```

The challenge pins:

- the dataset (here, a synthetic linear generator with a fixed seed
  and its SHA-256),
- the split rule (`first_fraction`) and the training fraction,
- the metric (`NMSE`),
- the baseline (the mean predictor, whose NMSE on the test split is
  exactly 1.0 by construction),
- the LSWU scoring parameters.

Inspect it:

```
python mlabchain.py challenge-inspect challenge.json
```

Write a training config:

```
cat > config.json <<'JSON'
{
  "model_type": "linear-regression",
  "framework": "mlabchain-native",
  "learning_rate": 0.01,
  "epochs": 50
}
JSON
```

Train, record, and seal in one step:

```
python mlabchain.py mine --challenge challenge.json --config config.json
```

Verify by re-training:

```
python mlabchain.py verify-ml --model mlabchain_data/models/model_ab12cd34ef56.pkl
```

The verifier prints every recorded metric next to its recomputed value,
recomputes Mera factor-by-factor, recomputes the symbolic operation
count, and prints the verification-to-training wall-time ratio.

---

## The scoring formula in more detail

### Why normalised MSE

Raw MSE has the units of the target variable squared. A model with
MSE = 0.001 on a problem with variance 0.01 is *not* better than a
model with MSE = 0.1 on a problem with variance 10. Normalising by the
test-set variance makes the number dimensionless and comparable across
problems:

```
NMSE = MSE / Var(y_test)
```

### Why a baseline-relative quality term

Comparing NMSE to a pinned baseline makes the score depend on the
*problem*, not on the units. The baseline must be pinned by the
challenge — if a contributor chooses their own baseline, they can
inflate the score by choosing a weak one, and no amount of re-training
will catch it because the baseline is a signed number, not a
measurement.

The challenge supports two baseline kinds:

- `mean_predictor` — the baseline predicts the mean of the test set.
  Its NMSE is exactly 1.0 by construction, so `I = 1 / NMSE_model`.
  This is what the demo uses.
- `model` — a reference model, identified by hash, whose NMSE is
  recorded in the challenge manifest. Not implemented in this build.

### Why logarithms

If time rewards were linear, someone could sleep their process and
inflate the score. Log scale gives diminishing returns, so an hour of
training earns more than a minute, but not sixty times more. The same
applies to dataset size.

### What Mera is not sensitive to

Hardware. A PC and a supercomputer that solve the same challenge to
the same quality level earn similar Mera, even though their wall times
differ by an order of magnitude. The log time term compresses the
difference; the quality term dominates the ordering. This is the
property the design is built around: Mera rewards *verified scientific
work*, not *hardware*.

---

## Proof-of-Scientific-Work, plainly stated

The README has used the phrase "Proof-of-Scientific-Work." It deserves
a straight definition, because it is not a proof in the cryptographic
sense.

**What it means here:** the chain records a signed claim that a
specific training was performed against a specific pinned challenge
with a specific config, producing a specific model and specific
metrics. Anybody with the challenge manifest and the config can
re-execute the training and check that the recorded metrics match.

**What it does not mean:** that verification is cheap. Re-executing
training costs what training costs. For a large model on a large
dataset, a full audit can take hours or days. The chain does not
pretend otherwise.

The chain is a **tamper-evident record of claims**. Verification is
re-execution, performed by whoever wants to audit a specific
transaction. There is no automatic, cheap, per-transaction check — the
whole point of useful work is that no such check exists.

---

## Commands

| Command | Purpose |
|---|---|
| `demo` | Full end-to-end demonstration. |
| `create-wallet` | Generate a new RSA wallet. |
| `hash-file PATH` | SHA-256 of a file. |
| `challenge-create` | Create a synthetic-linear challenge manifest. |
| `challenge-inspect PATH` | Print a challenge manifest. |
| `mine` | Seal a block. With `--challenge` and `--config`, trains first. |
| `status` | Print the chain, per-transaction Mera and operation counts. |
| `validate` | Check hashes, block sealing, Merkle roots, signatures. |
| `credits` | Show accumulated Mera and symbolic operation totals. |
| `symbolic --tx-hash HEX` | Show the symbolic cost breakdown of one transaction. |
| `verify-ml --model PATH` | Re-train and compare metrics, Mera, and cost. |
| `verify-ml --tx-hash HEX` | Same, addressed by transaction hash. |

---

## What is recorded in a transaction

Each `ML_TRAINING` transaction commits to:

- the **challenge** it was run against: challenge ID, manifest SHA-256,
  dataset SHA-256, split rule, train fraction, metric, baseline kind,
  baseline NMSE;
- the **config** in full, plus its canonical hash;
- the **model file**: path, SHA-256, size;
- the **architecture file**: path, SHA-256, size, and text content;
- the **training metadata**: number of training examples, number of
  test examples, epochs, learning rate, reported wall time in seconds;
- the **metrics**: MSE on train and test, test-set variance, NMSE of
  the model, NMSE of the baseline, R² on the test set;
- the **Mera score** with all its components (`D`, `T`, `Q`, `I`, and
  each powered factor);
- the **symbolic cost**: elementary operation count `C_train_ops`,
  which equals `C_verify_ops` by construction;
- free-form notes and tags;
- a timestamp and an RSA signature.

---

## Verification semantics

`verify-ml` re-executes the training from the transaction's config and
challenge fields, then:

1. Compares each recorded metric to the recomputed one, to floating-point
   tolerance.
2. Recomputes Mera factor by factor and compares.
3. Recomputes the symbolic operation count and prints the comparison.
4. Measures its own wall time and prints the ratio to the reported
   training time, with the interpretation spelled out.

Because the trainer is deterministic (linear regression with no
shuffling, seeded dataset generation, fixed split), metric matches are
exact. In a stochastic training regime the same code would produce
approximate matches, and the check would need to be relaxed — the
docstring of `run_challenge_training` marks this assumption.

Three outcomes are possible:

- **VERIFIED** — every metric and every Mera component matches.
- **MISMATCH** — a metric or a Mera component diverges.
- **NOT FOUND** — no matching transaction in the chain.

---

## What is in the box

```
mlabchain.py              the whole thing, one file
README.md                 this document
LICENSE.txt               Apache 2.0
requirements.txt          cryptography
materials/
├── mlabchain.png         project logo
└── MERA.png              Mera logo
mlabchain_data/
├── blockchain.json       the chain
├── wallet.json           your RSA keypair (plaintext)
├── challenges/           challenge manifests
├── models/               trained models and architecture files
└── demo/                 artifacts created by `demo`
```

The chain file is human-readable JSON. You can `cat` it, `jq` it, diff
it, grep it.

---

## Honest limitations

- **The private key is plaintext.** `wallet.json` is not encrypted.
  Do not reuse this wallet for anything of value.
- **This is a single-writer chain.** Anyone with write access to
  `mlabchain_data/` can rewrite the whole thing from genesis. Validation
  gives internal consistency, not protection from a determined attacker.
- **The wall-time field is a claim.** A miner can report a larger number
  than they actually spent. Re-training catches this indirectly, by
  showing that the true cost is lower, but only if someone actually
  re-trains.
- **The baseline must be pinned by the challenge.** If a contributor
  chooses their own baseline, the quality term becomes a signed claim.
  The challenge manifest machinery exists specifically to prevent this.
- **Verification costs what training costs.** Do not expect cheap
  per-transaction checks. There are none, and there cannot be any,
  given what the system is trying to do.
- **Only linear regression is implemented.** The symbolic cost formula
  `3F + 4` is specific to that trainer. Any other model class would
  need its own op-per-sample count.
- **Only synthetic datasets are implemented.** Real file-backed datasets
  (ROOT, CSV, HDF5) are future work.
- **No schema migration.** Bumping the schema means a new chain or
  manual conversion.
- **Not audited.** Use it on things you can afford to lose.

---

## On the name "Mera"

The unit is called Mera because it needed a name that was short, easy
to type, and not already taken by another scientific unit. It is used
in the sense of "one measure of verified training work," the same way
"joule" means one measure of energy.

Mera 0.1.7 is not a ticker symbol, not an asset, and not something that trades
anywhere. If you find a project elsewhere that uses a similar name for
a financial instrument, it is not this project and it has nothing to do
with this project.

---

## What I would build next

- **File-backed datasets.** A `dataset_kind="file"` that hashes a local
  ROOT or CSV file and parses a specified column range, so challenges
  can be defined against real data.
- **Non-linear models.** A small neural network in pure Python, with
  its own operation-count function, so the LSWU formula has more than
  one trainer to apply to.
- **Real baselines.** Consume `baseline_kind="model"`, loading a
  reference model from a hash-addressed store.
- **Cross-hardware normalisation.** A benchmark embedded in the
  challenge, so that the wall-time ratio can be interpreted on a
  consistent scale across machines.
- **A `status --query` filter** by challenge ID, tag, or metric range.
- **Export to Markdown or LaTeX**, for pasting a training-record table
  into a paper's supplementary material.

None of these are hard. They are just not written yet.

---

## Why this exists

I work with machine-learning models on physics data. I have trained
models, published results, and then six months later been unable to
reconstruct exactly which config, which split, and which data produced
the numbers in a plot. Git helps with the code. Nothing helps with the
trained weights, the metrics, or the exact hyperparameters.

MLabChain fixes that, for me, on my laptop, without asking anyone's
permission. It gives me a signed record of every training run I care
about, and it lets me verify the record later by re-running the
training and checking the numbers.

That is a small thing. It is worth a small thing.

If you find a bug, open an issue. If you want a new model class, send
a PR. If you want to describe this as a coin, please don't — the README
has been careful to explain why, and the code has been careful to make
it impossible.

— someone who has spent too long looking for old configs

---

## License

Apache License 2.0. See `LICENSE.txt`.
