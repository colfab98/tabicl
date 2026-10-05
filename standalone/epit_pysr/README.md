# Standalone EPIT PySR discovery bundle

This directory is the deployable copy of the Git-tracked discovery workflow in
`scripts/epit_pipeline/run_pysr_discovery.py`. It contains only the 608
development rows from the frozen old-EPIT split and no final-test rows or
targets. The experiment evaluates the same 452 Fe/Ni development rows as the
strong hand-designed rule.

The search begins with all 24 elements plus temperature, chloride, and pH.
PySR's random-forest selector chooses physical inputs inside each context fold.
Test method is represented by learned additive categorical offsets, outside the
symbolic physical formula. No PREN term or corrosion formula is supplied.

`sv3000` is the CPU-cluster head node. First copy the bundle from `RZ-Dienste`
to the cluster home, which is mounted on the compute nodes. Then submit setup
and PySR through SLURM:

```bash
cp -a ~/RZ-Dienste/hpc-user/fcolanto/epit_pysr_20261005 ~/projects/epit_pysr
cd ~/projects/epit_pysr
sbatch bootstrap.sbatch
```

After that setup job finishes, submit the dry run:

```bash
sbatch run_pysr.sbatch
```

The full five-fold, 1,000-iteration search is:

```bash
sbatch --export=ALL,MODE=full run_pysr.sbatch
```

A full five-fold, three-seed discovery run is:

```bash
sbatch --export=ALL,MODE=full,NITERATIONS=5000,POPULATIONS=16,SEEDS=42:43:44,SELECT_K_FEATURES=12 run_pysr.sbatch
```

Use `squeue -u "$USER"` to check the queue. Job logs are written under
`~/tmp/`.

The included `wheelhouse/` and compressed `runtime/` were prepared on an online
Linux x86-64 host. `bootstrap_offline.sh` installs and verifies them without
network access.
