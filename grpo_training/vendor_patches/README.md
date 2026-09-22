# verl customizations

Local patch set for our vendored verl at `grpo_training/vendor/verl`.

That verl directory is a **separate clone** of upstream verl (not a submodule),
and it is **gitignored** by this repo. So our local edits to it are not tracked
here directly — they are preserved as the patch in this folder
(`verl_custom.patch`) and re-applied on each machine.

- Upstream: https://github.com/verl-project/verl.git
- Base commit: `86e8123643c982343b11fa21dc1df96ab07ced3d`
- Patch covers 4 files (~54 lines): ray_trainer.py, reward_loop.py,
  fsdp_checkpoint_manager.py, transformer_impl.py

## How the patch works

The patch stores **only our changes**, not the verl code itself. To recreate our
customized verl you must (1) get verl to the exact base commit, then (2) apply
the patch on top. The patch will only apply cleanly against that base commit,
because git locates each change by matching the surrounding lines.

## Applying it

Run these from the repo root (`insight_synthesis/`).

### Case A — verl is NOT cloned yet (fresh machine)

```bash
# 1. clone upstream verl into the vendored location
git clone https://github.com/verl-project/verl.git grpo_training/vendor/verl

# 2. move it to the exact commit the patch was built against
cd grpo_training/vendor/verl
git checkout 86e8123643c982343b11fa21dc1df96ab07ced3d

# 3. (optional) dry-run: check it will apply cleanly, changes nothing
git apply --check ../vendor_patches/verl_custom.patch

# 4. apply our customizations
git apply ../vendor_patches/verl_custom.patch
```

### Case B — verl is already cloned

```bash
cd grpo_training/vendor/verl

# make sure there are no local changes first (stash or discard as needed)
git status

# move to the base commit, then apply
git checkout 86e8123643c982343b11fa21dc1df96ab07ced3d
git apply --check ../vendor_patches/verl_custom.patch   # optional dry-run
git apply ../vendor_patches/verl_custom.patch
```

## If the patch fails to apply

You will see `error: patch does not apply` if verl is not at the base commit,
or if it has drifted. Options:

```bash
# confirm you are on the right commit
git rev-parse HEAD    # should be 86e8123643c982343b11fa21dc1df96ab07ced3d

# try a 3-way merge (leaves conflict markers you resolve by hand)
git apply --3way ../vendor_patches/verl_custom.patch
```

## Updating the patch (after changing verl again)

Regenerate the patch from inside the verl clone and copy it back here:

```bash
cd grpo_training/vendor/verl
git diff > ../vendor_patches/verl_custom.patch
# then commit the updated patch in the insight_synthesis repo
```

If you update to a newer verl base commit, also update the "Base commit" line
above so future applies check out the right starting point.
