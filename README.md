# Hebbian Sandbox

**DataForge 2026 — Pathway Track submission**

## The claim

> A network with no dedicated memory component can still "remember" a recent
> pattern for several steps, purely because Hebbian updates temporarily
> strengthen the synapses that just fired together — and that memory decays
> and can be overwritten by interference, not by an explicit forget gate.

## Chosen topic

**Synaptic Plasticity as Short-Term Memory** (approved topic list).

## Intended learner and prerequisites

Data scientists and ML engineers familiar with basic neural network concepts
(neurons, weights, activations) but new to brain-inspired short-term memory
mechanisms. No prior knowledge of Hebbian learning or BDH is required.

## Learning objectives

After using this sandbox, a learner should be able to:
1. Explain how Hebbian updates can act as short-term memory without a
   dedicated memory module.
2. Predict how increasing delay or interference will affect recall.
3. Distinguish short-term synaptic memory from durable parameter learning.
4. Describe how BDH uses a related mechanism (synaptic plasticity as working
   memory) at model scale, with the same order-of-magnitude sparsity level.

**60-second test:** after one guided trial, a learner should be able to say,
in 1-2 sentences, why the network's recall gets worse as delay and
interference increase, without prompting.

## Architecture

Five-stage pipeline, plus a BDH connection module branching off the
encoding stage:

| Stage | Input | What happens | Output |
|---|---|---|---|
| 1. Pattern input (learner-facing) | Learner-chosen delay / interference / cue-completeness settings | Learner sets the three control parameters | Parameters passed to the engine |
| 2. Hebbian encoding | A random sparse pattern (k=10 of n=200 units active) | Co-active units get their synapse strengthened via the covariance Hebbian rule | Updated 200x200 synapse weight matrix |
| 3. Synapse weight state | Weight matrix, continuously | Weights decay (scalar shrink + background synaptic noise) over the chosen number of delay steps; optional distractor patterns are encoded (interference) | Current, possibly-corrupted weight matrix |
| 4. Recall | Decayed matrix + a partial cue (a fraction of the true active units, chosen by the learner) | One-shot readout: score every unit from the weights, force revealed units on, keep the top-k highest-scoring units active | Reconstructed pattern + accuracy (overlap on active units) |
| 5. Comparison output (learner-facing) | Reconstructed pattern + ground truth | Side-by-side display | The teaching moment |
| BDH module (branches off stage 2) | — | Shows the real mechanism from the Dragon Hatchling paper alongside the toy version | — |

- **Backend:** `hebbian_sparse.py` (the frozen verified teaching engine; SHA-256: `9a6287877a8f678dcae4eef451c26e74c181b867635f55499c9aff18a4aad4dd`) + `app.py` (FastAPI, exposing `/trial`, `/history`, and `/experiment`).
- **Calibration:** `calibrate.py` (finite-difference gradient descent used to calibrate decay and noise parameters).
- **Archive:** `archive/hebbian_engine.py` (archived Phase 1 dense +/-1 prototype, preserved for provenance).
- **Test Suite:** `tests/test_regression.py` (pytest regression suite validating determinism, decay curves, interference degradation, and API schema contracts).
- **Frontend:** `index.html` — a single-file interactive sandbox (no build step) that calls the backend live and renders the synapse heatmap, pattern grids, diagnostic feedback, and sweep charts from real computed data.

## What's live, precomputed, or illustrative

**Everything in this sandbox is live, computed on demand — nothing is
precomputed or animated.** Every trial randomly generates a fresh pattern,
genuinely encodes it into a weight matrix via the Hebbian rule, genuinely
decays/interferes with that matrix, and genuinely reconstructs it from a
partial cue. The weight heatmap, the three pattern grids, and the accuracy
number are all computed from that specific run, not looked up or faked.

## How to reproduce

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run automated regression tests
pytest tests/ -v

# Start the unified server
uvicorn app:app --port 8000

# Open your browser to http://127.0.0.1:8000
```

Sanity check (delay=0, distractors=0 must give exactly 1.0; delay=20,
distractors=20 should sit noticeably below 1.0 — exact numbers are verified by `tests/test_regression.py`):

```python
from hebbian_sparse import run_sparse_experiment
run_sparse_experiment(n_distractors=0)   # -> 1.000 at delay=0, ~0.667 at delay=20
run_sparse_experiment(n_distractors=20)  # -> lower than the above at every delay > 0 (0.629 at delay=20)
```

## Development history and known limitations

This engine went through three rounds of verification, and each round caught
a real bug that would have quietly invalidated the claim if shipped:

1. **Scalar decay alone is invisible to sign/top-k-based recall** — shrinking
   every weight by the same factor doesn't change which units score highest,
   so nothing appeared to forget. Fixed by adding background synaptic noise
   during decay, which is what actually erodes signal-to-noise over time.
2. **The accuracy metric was wrong for sparse patterns** — with only 5% of
   units active, "fraction of matching bits" gave a blank, useless output
   ~95% accuracy. Fixed by measuring overlap only on the true active units.
3. **The cue accidentally revealed the answer** — revealing a fraction of
   the whole (mostly-zero) vector also revealed the same fraction of the
   active units for free. Fixed by revealing a fraction of the active units
   specifically.

**Limitations we're disclosing directly, not hiding:**
- This is a small, standalone toy model (n=200 units) built for teaching
  purposes. It is **not** the official BDH implementation and should never
  be presented as one.
- Recall here is a one-shot readout, not the deeper recurrent inference
  dynamics a full sequence model would use — this was a deliberate choice
  so the decay/interference effects stay honestly visible rather than being
  masked by iterative cleanup (an earlier version with multi-step attractor
  dynamics did exactly that, and was rejected for it).
- This sandbox's "memory" is entirely within-session. It does not model how
  (or whether) useful short-term synaptic state might ever consolidate into
  durable, cross-session parameter learning — that remains an open question,
  including for BDH itself.

## Connection to Dragon Hatchling (BDH)

Primary source: Kosowski, Uznański, Chorowski, Stamirowska, Bartoszkiewicz.
"The Dragon Hatchling: The Missing Link between the Transformer and Models
of the Brain." arXiv:2509.26507 (2025).

- **Working memory = Hebbian synapses** (Abstract, Section 1.2): BDH's
  working memory during inference relies entirely on synaptic plasticity
  with Hebbian learning, at a potentiation timescale comparable to minutes
  of brain activity (roughly hundreds of tokens) — the same mechanism, and
  the same order of timescale, this sandbox's "memory delay" slider models.
- **Sparse, non-negative activation** (Section 6.2): BDH-GPU's positive
  activation vectors are reported at roughly 5% sparsity. This sandbox uses
  exactly that sparsity level (k=10 of n=200 units), not an arbitrary
  choice.
- **Monosemantic synapses** (Section 6.3): the paper reports that a
  synapse's in-context state localizes consistently on the same
  neuron-neuron connection across multiple prompts, letting individual
  synapses be read as concept-specific features.

This sandbox's toy network is an independent, simplified reimplementation
built for teaching. It is explicitly labeled as such throughout the app and
this README, and should never be represented as the official BDH model.

## Primary research papers (2022-2026) on the selected concept

1. Duan, Y., Jia, Z., Li, Q., Zhong, Y., Ma, K. (2023). "Hebbian and
   Gradient-based Plasticity Enables Robust Memory and Rapid Learning in
   RNNs." ICLR 2023. arXiv:2302.03235. — Directly relevant: compares local
   Hebbian plasticity against gradient-based plasticity for memory and
   associative-learning tasks in RNNs, and reports Hebbian plasticity as
   well-suited for memory/associative tasks specifically.
2. Behrouz, A., Zhong, P., Mirrokni, V. (2024/2025). "Titans: Learning to
   Memorize at Test Time." arXiv:2501.00663. Google Research. — Relevant
   context: a recent, prominent architecture built around test-time memory
   formation (rather than only pretraining-time learning), framing attention
   itself as a form of short-term memory versus a longer-term memory module.
3. Szelogowski, D. (2025). "Hebbian Memory-Augmented Recurrent Networks:
   Engram Neurons in Deep Learning." arXiv:2507.21474. — Directly relevant:
   introduces an explicit, differentiable memory matrix using Hebbian
   plasticity and sparse, attention-driven retrieval, motivated by the same
   biological engram/synaptic-plasticity framing this project uses.
4. Ellwood, I. T. (2024). "Short-term Hebbian learning can implement
   transformer-like attention." PLoS Computational Biology, 20(1), e1011843.
   https://doi.org/10.1371/journal.pcbi.1011843 — Directly relevant: provides
   a rigorous theoretical and biophysical proof demonstrating how short-term
   Hebbian synaptic potentiation implements computations functionally
   analogous to key-query attention in transformers. Directly reinforces the
   Hebbian-to-Transformer conceptual bridge presented in the BDH module.

## Public deployment & submission artifacts

- **Architecture:** Zero-framework single-page application served via FastAPI.
  Supports both local execution and standard cloud platforms without modification.
- **Production Server Startup:**
  ```bash
  uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
  ```
- **Deployment Manifests:**
  - `Procfile`: Configured for immediate deployment on Render, Railway, or Heroku.
  - `render.yaml`: Blueprint configuration for automated Render web service deployment.
- **Concept Summary Artifact:**
  A standalone, verified 1-page PDF summary (675 words) is generated in the artifacts directory (`concept_summary.pdf`) for judging review.

## Credits and licenses

- **numpy** (BSD-3-Clause), **FastAPI** (MIT), **uvicorn** (BSD-3-Clause) —
  standard open-source dependencies, unmodified.
- **Fonts:** Fraunces and IBM Plex Mono, both loaded from Google Fonts,
  licensed under the SIL Open Font License.
- All simulation code, the FastAPI backend, and the frontend in this
  repository were written for this project; no external code was copied in.

## AI assistance disclosure

This project was built with iterative assistance from AI agents.
**IMPLEMENTATION:** Antigravity / Gemini 3.6 High (Google DeepMind / Google) proposed initial implementations of the Hebbian memory engine, the FastAPI backend, and the frontend.
**INDEPENDENT REVIEW / RESEARCH:** Claude Sonnet 4.6 Thinking (Anthropic) acted as an independent reviewer and strategic research agent.

This generated code and logic was then **tested and verified at
every stage** rather than accepted on trust — this process caught and fixed
three real bugs (see "Development history" above) and one fabricated
citation during drafting, which was removed and replaced with primary
sources checked directly against arXiv. The team is responsible for, and
can explain and defend, every claim, citation, and line of code in this
submission.
