# Notice

This project is licensed under the Apache License, Version 2.0 (see `LICENSE`).

## Third-party dependencies not vendored here

- **verl** (https://github.com/verl-project/verl), licensed under Apache-2.0.
  Not vendored in this repository -- `grpo_training/setup/setup_env.sh` clones
  a pinned commit into `grpo_training/vendor/verl` and applies a small patch
  (`grpo_training/vendor_patches/verl_custom.patch`, see the accompanying
  `README.md` for the exact clone/checkout/apply steps). See verl's own
  repository for its license and notices.
- **giants** (https://github.com/joyheyueya/giants). Not vendored here --
  `scripts/prepare_rl_data.sh` clones it into `grpo_training/vendor/giants` to
  reuse its `scripts/prepare_rl_data.py` against the public GiantsBench-train
  dataset. See that repository for its license.
