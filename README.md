# Compact proofs for a 2-layer induction head

Lower bounds on the accuracy of a small trained transformer, computed from its weights without running the model on every input, with floating-point rounding accounted for, and checked against brute-force evaluation.

## Setup

**Model.** A 2-layer attention-only transformer: no LayerNorm, no MLP, one head per layer, d_model = 32, d_head = 16, learned token and position embeddings, unembedding read at the final position only.

**Task.** Vocabulary of d = 16 tokens, sequence length n = 6 (one run uses n = 8). Tokens x_0 … x_{n-2} are distinct; the final token repeats an earlier one, x_{n-1} = x_i for some i ≤ n-3. The target at the final position is x_{i+1}. Inputs are uniform over all such sequences, which gives 2,096,640 inputs for n = 6.

**Groups.** Inputs are partitioned by (q, i, a): the query token q = x_i = x_{n-1}, the match position i, and the answer a = x_{i+1}. That gives 16 · 15 · 4 = 960 groups of 1,716 inputs each, which vary only in the three distractor tokens. A group is *certified* if a lower bound on the logit margin (correct logit minus the largest wrong logit) is positive for every input in it. All groups are the same size, so the fraction of certified groups is a lower bound on accuracy.

## Method

The bound is computed from tables built from the weights (`proof_v1.build`). Within a group, the distractor tokens are relaxed and each attention pattern is replaced by a set of distributions that contains it.

- **v1** (`proof_v1.py`): layer-1 attention at each key position is bounded below on the previous token. Layer-2 attention is bounded below on the correct key (i+1). Linear quantities are bounded at the vertices of these polytopes.
- **v2** (`proof_v2.py`): every attention row is relaxed to {w in the simplex : l ≤ w ≤ u}, with per-position bounds l, u derived from score intervals. Linear functionals are bounded exactly over this polytope by a greedy fractional-knapsack LP.
- **v3** (`proof_v3.py`): v2, plus two refinements.
  - For each distractor key k, the tokens at positions (k-1, k) are enumerated jointly, so the key's residual, the key's layer-1 attention row and the query's layer-1 atoms all use the same tokens.
  - Layer-2 attention is relaxed to a box polytope from the per-key score gaps, and the logit margin is bounded by an LP over that box.
- **v4** (`proof_v4.py`, current): v3 with a different treatment of layer-1 attention rows. In v2 and v3 the attention weights range over a box independently of the tokens, although each weight is determined by the token at its position. A quantity Σ_m A[m] V_m, with A a layer-1 attention row, equals Σ_m exp(s_m(t_m)) V_m(t_m) / Σ_m exp(s_m(t_m)), where the score s_m and the value V_m depend only on the token t_m at position m. The minimum of this ratio over independent token choices per position is computed exactly (Dinkelbach's iteration for fractional programs). This is used for the final-position row in every layer-2 score gap and for all rows in the logit terms. Tokens fixed by the enumeration are also excluded from the other distractor positions.

`sound.py` checks every intermediate bound against its true minimum over each group's inputs: the layer-2 score gaps, the layer-2 attention on the correct key, and the logit margin.

### Why v4

In v3, the failing groups were examined by replacing parts of the bound with their true values, computed by brute force over the group. With the true layer-2 score gaps, all 60 failing groups of seed 0 at n = 6 certify, and 33 of a sample of 40 at n = 8. With the true range of layer-1 attention in place of the boxes, 28% and 15% certify. So the loss was in the score gaps, and it did not come from the boxes being wider than the true attention range: the final-position attention row does vary by 0.2–0.4 across a group. The loss came from letting the weights move within that range independently of the tokens. v4 removes that independence.

Applying the same exact minimisation to the key rows inside each score gap was also tried. At n = 8 it certified 4 more groups (1353 vs 1349) at about 6 times the cost, so it is not included.

### Floating point

`proof_v3_rigorous.py` and `proof_v4_rigorous.py` recompute the bounds in outward-rounded interval arithmetic (`interval.py`). A group counts as certified only if a floating-point number that is provably below the exact margin is positive.

- Every table built from the weights is an interval that contains its exact value. Each elementary operation is widened by one unit in the last place, and sums use the standard a-priori error bound for floating-point summation. Matrix products are written as elementwise products followed by sums, so nothing depends on BLAS.
- exp is computed from the basic operations (argument reduction and a Taylor series with a remainder bound), so nothing depends on the accuracy of the system math library.
- Each LP value is replaced by a dual bound, which is valid for any multiplier, so the result does not depend on the floating-point LP solution being optimal. Similarly, the ratio minimum in v4 is certified by a sign check at a value slightly below the floating-point estimate.

The certified statement concerns the real-valued function defined by the stored float64 weights, evaluated in exact arithmetic with a hard causal mask. `tests/test_interval.py` checks the interval primitives against exact rational arithmetic and 60-digit exp. `tests/check_rigorous_vs_float.py` checks that the rigorous and plain-float versions of a bound agree to rounding error on every group.

## Results

n = 6, 960 groups per seed. Every seed reaches 100% brute-force accuracy. The certified counts are from the rigorous versions; the plain-float versions certify the same groups in every case, and the two differ by less than 1e-8 in every margin.

| seed | brute-force min margin | v3 certified | v4 certified | v4 accuracy lower bound |
|---|---|---|---|---|
| 0 | 18.59 | 900 | 959 | 0.9990 |
| 1 | 15.49 | 812 | 957 | 0.9969 |
| 2 | 15.44 | 928 | 959 | 0.9990 |
| 3 | 10.44 | 812 | 937 | 0.9760 |
| 4 | 15.06 | 952 | 960 | 1.0000 |
| 5 | 16.97 | 890 | 957 | 0.9969 |
| 6 | 15.03 | 900 | 956 | 0.9958 |

`sound.py` finds 0 violations for every seed and every version. For v4 the smallest difference between a true value and its bound is 0 to four decimal places: some score-gap bounds are attained.

Approximate run times per seed on a laptop CPU at n = 6: brute force 10–20 s, v4 in plain float 9 s, v4 rigorous 50–60 s. At n = 6 the proof is not cheaper than brute force. The comparison of interest is how the two scale.

**n = 8, seed 0** (`results/v4_n8_s0.log`, 1440 groups).

| | certified | accuracy lower bound | time |
|---|---|---|---|
| v3 | 1093 | 0.759 | 30–90 s |
| v4, plain float | 1349 | 0.937 | about 40 s |
| v4, rigorous | 1349 | 0.937 | about 3 min |
| brute force (estimated) | | | 40–70 min |

Sampled accuracy is 1.0 on 400k inputs. The per-group soundness check on 6 random groups (240,240 inputs each) found no violations. Of the 91 groups v4 does not certify, 62 have the match at the first or last possible position (i = 0 or i = 5), and 37 share one query token.

## Limitations

- The rigorous bounds are statements about the exact-arithmetic function of the stored weights. They do not bound the rounding error of a particular floating-point forward pass. That error is many orders of magnitude smaller than the smallest certified margins (0.06 for v4 at n = 6), but it is not part of the certificate.
- The interval code assumes IEEE-754 round-to-nearest for single numpy operations. Its correctness is tested, not machine-checked, and the same holds for the derivation of the bound itself: soundness is checked against brute force (exhaustively at n = 6, on sampled groups at n = 8).
- The model and task are small, with one seed at n = 8. Every model is 100% accurate on the inputs evaluated, so each uncertified group reflects looseness in the bound.

## Usage

```bash
uv venv && uv pip install -r requirements.txt       # numpy, autograd

python src/model.py SEED STEPS                      # train; writes weights/params_s{SEED}.pkl
python src/brute.py weights/params_s0.pkl           # exhaustive accuracy (n = 6)
python src/proof_v4.py weights/params_s0.pkl        # accuracy lower bound, plain float
python src/proof_v4_rigorous.py weights/params_s0.pkl   # the same bound in interval arithmetic
python src/sound.py weights/params_s0.pkl proof_v4_rigorous   # check every bound against brute force

python tests/test_interval.py                       # interval primitives vs exact arithmetic
python tests/check_rigorous_vs_float.py weights/params_s0.pkl proof_v4

N=8 python src/model.py 0 STEPS                     # length-8 task -> weights/params_n8_s0.pkl
N=8 python src/n8check.py weights/params_n8_s0.pkl 6 proof_v4_rigorous
```

Task size is set by the environment variables `D` (vocabulary, default 16) and `N` (length, default 6). `sound.py`, `n8check.py` and `check_rigorous_vs_float.py` take the name of the proof module as their last argument.

## Layout

```
src/       model, brute force, proof versions v1–v4, interval arithmetic, soundness checks
tests/     interval arithmetic vs exact arithmetic; rigorous vs plain-float bounds
weights/   trained parameters (pickled dicts of numpy arrays)
results/   logs per seed: brute force, bounds, soundness
```
