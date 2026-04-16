<h1 align="center">DDTree</h1>

<p align="center">
  Official implementation of <strong>DDTree (Diffusion Draft Tree)</strong> from
  <em>Accelerating Speculative Decoding with Block Diffusion Draft Trees</em>.
</p>

<p align="center">
  <a href="https://liranringel.github.io/ddtree/">🌐 Project Page</a>
  &nbsp;|&nbsp;
  <a href="https://liranringel.github.io/ddtree/DDTree.pdf">📄 Paper</a>
</p>

## Setup

This codebase is intended for a CUDA-enabled PyTorch environment.

```bash
pip install -r requirements.txt
```

## Run Experiments

```bash
bash run_benchmark.sh
```

This produces benchmark outputs in `runs/` and logs in `logs/`.
The benchmark runner compares autoregressive decoding, DFlash, MDFlash, P-Express, P-Flash, P-Flash V2, P-Flash V3, P-Flash V4, P-Flash V5, P-Flash V6, and DDTree in the same sweep.

`run_benchmark.sh` also exposes `PEXPRESS_PERTURBATION_TEMPERATURE`, `PEXPRESS_POSITION_TEMPERATURE_DECAY`, `PFLASH_BRANCH_PRIOR_WEIGHT`, `PFLASH_MERGE_PREFIX_BRANCHES`, `PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT`, and the `PFLASH_V4_*`/`PFLASH_V5_*`/`PFLASH_V6_*` knobs as environment-variable overrides for tuning the perturbation sweep. P-Express, P-Flash, P-Flash V2, P-Flash V3, P-Flash V4, P-Flash V5, and P-Flash V6 all reuse the perturbation settings, while P-Flash adds a branch-prior penalty and can optionally merge branch proposals into a shared-prefix trie with a configurable support bonus. P-Flash V4 keeps a DDTree backbone and spends only the tail of the budget on normalized perturbation grafts. P-Flash V5 uses perturbation-batch agreement as a confidence gate for adaptive tree shape. P-Flash V6 uses agreement to adapt verifier block length and tree budget.

Set `MEASURE_BATCH_AGREEMENT=1` or pass `--measure-batch-agreement` to store perturbation-batch agreement diagnostics and have `getlog.sh` report their correlation with actual acceptance.

## Reproduce Paper Artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```
