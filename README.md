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
The benchmark runner compares autoregressive decoding, DFlash, MDFlash, P-Express, P-Flash, P-Flash V2, P-Flash V3, and DDTree in the same sweep.

`run_benchmark.sh` also exposes `PEXPRESS_PERTURBATION_TEMPERATURE`, `PEXPRESS_POSITION_TEMPERATURE_DECAY`, `PFLASH_BRANCH_PRIOR_WEIGHT`, `PFLASH_MERGE_PREFIX_BRANCHES`, and `PFLASH_PREFIX_SUPPORT_BONUS_WEIGHT` as environment-variable overrides for tuning the perturbation sweep. P-Express, P-Flash, P-Flash V2, and P-Flash V3 all reuse the perturbation settings, while P-Flash adds a branch-prior penalty and can optionally merge branch proposals into a shared-prefix trie with a configurable support bonus.

## Reproduce Paper Artifacts

Generate the plots:

```bash
python3 plot_results.py
```

Generate the LaTeX table:

```bash
python3 make_latex_table.py
```
