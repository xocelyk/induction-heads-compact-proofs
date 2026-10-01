# Compact proofs for a 2-layer induction head

Lower bounds on the accuracy of a small trained transformer, computed from its weights without running the model on every input, and checked against brute-force evaluation.

## Setup

**Model.** A 2-layer attention-only transformer: no LayerNorm, no MLP, one head per layer, d_model = 32, d_head = 16, learned token and position embeddings, unembedding read at the final position only.

**Task.** Vocabulary of d = 16 tokens, sequence length n = 6 (one run uses n = 8). Tokens x_0 … x_{n-2} are distinct; the final token repeats an earlier one, x_{n-1} = x_i for some i ≤ n-3. The target at the final position is x_{i+1}. Inputs are uniform over all such sequences, which gives 2,096,640 inputs for n = 6.

**Groups.** Inputs are partitioned by (q, i, a): the query token q = x_i = x_{n-1}, the match position i, and the answer a = x_{i+1}. That gives 16 · 15 · 4 = 960 groups of 1,716 inputs each, which vary only in the three distractor tokens. A group is *certified* if a lower bound on the logit margin (correct logit minus the largest wrong logit) is positive for every input in it. All groups are the same size, so the fraction of certified groups is a lower bound on accuracy.

## Method

The bound is computed from tables built from the weights (`proof_v1.build`). Within a group, the distractor tokens are relaxed and each attention pattern is replaced by a set of distributions that contains it.

- **v1** (`proof_v1.py`): layer-1 attention at each key position is bounded below on the previous token. Layer-2 attention is bounded below on the correct key (i+1). Linear quantities are bounded at the vertices of these polytopes.
- **v2** (`proof_v2.py`): every attention row is relaxed to {w in the simplex : l ≤ w ≤ u}, with per-position bounds l, u derived from score intervals. Linear functionals are bounded exactly over this polytope by a greedy fractional-knapsack LP.
- **v3** (`proof_v3.py`, current): v2, plus two refinements.
  - For each distractor key k, the tokens at positions (k-1, k) are enumerated jointly, so the key's residual, the key's layer-1 attention row and the query's layer-1 atoms all use the same tokens.
  - Layer-2 attention is relaxed to a box polytope from the per-key score gaps, and the logit margin is bounded by an LP over that box.

`sound.py` checks every intermediate bound against its true minimum over each group's inputs: the layer-2 score gaps, the layer-2 attention on the correct key, and the logit margin.

## Results

v3 bound, n = 6, 960 groups per seed. Every seed reaches 100% brute-force accuracy.

| seed | brute-force min margin | groups certified | accuracy lower bound | soundness violations |
|---|---|---|---|---|
| 0 | 18.59 | 900 / 960 | 0.9375 | 0 |
| 1 | 15.49 | 812 / 960 | 0.8458 | 0 |
| 2 | 15.44 | 928 / 960 | 0.9667 | 0 |
| 3 | 10.44 | 812 / 960 | 0.8458 | 0 |
| 4 | 15.06 | 952 / 960 | 0.9917 | 0 |
| 5 | 16.97 | 890 / 960 | 0.9271 | 0 |
| 6 | 15.03 | 900 / 960 | 0.9375 | 0 |

Brute force takes about 10–20 s per seed on a laptop CPU, and the v3 bound about 5–20 s. At n = 6 the proof is not cheaper than brute force. The comparison of interest is how the two scale.

**n = 8, seed 0** (`results/res_n8_s0.log`). The bound certifies 1093 / 1440 groups (accuracy lower bound 0.759) in about 70 s. Sampled accuracy is 1.0 on 400k inputs, and brute force over all 3.5 × 10⁸ inputs is estimated at about 1.2 h. The per-group soundness check on 6 random groups found no violations.

## Limitations

- The bounds are sound in exact arithmetic. Floating-point rounding is not yet accounted for, so they are not formal certificates.
- Soundness is checked empirically against brute force (exhaustively at n = 6, on sampled groups at n = 8). There is no machine-checked proof that the bound computation is correct.
- The model and task are small. Every model is 100% accurate, so each uncertified group reflects looseness in the bound.

## Usage

```bash
uv venv && uv pip install -r requirements.txt   # numpy, autograd

python src/model.py SEED STEPS                  # train; writes weights/params_s{SEED}.pkl
python src/brute.py   weights/params_s0.pkl     # exhaustive accuracy (n = 6)
python src/proof_v3.py weights/params_s0.pkl    # accuracy lower bound
python src/sound.py   weights/params_s0.pkl     # check every bound against brute force

N=8 python src/model.py 0 STEPS                 # length-8 task -> weights/params_n8_s0.pkl
N=8 python src/n8check.py weights/params_n8_s0.pkl 6
```

Task size is set by the environment variables `D` (vocabulary, default 16) and `N` (length, default 6).

## Layout

```
src/       model, brute force, proof versions v1–v3, soundness checks
weights/   trained parameters (pickled dicts of numpy arrays)
results/   logs per seed: brute force, bound, soundness
```
