# EPIT 24-element v8 snapshot (2026-09-28)

This snapshot preserves the suspended 24-element composition schema, new target
rules, tests, generated development-fold rule-evaluation artifacts, and the
corresponding `main4_supplement.md` text. It was taken before the live tree was
returned to the source state frozen by the in-progress coefficient-variation
Optuna experiment.

The snapshot is intentionally stored in a normal, non-ignored repository path
so it can be committed and pushed. The canonical payload is
`epit24_v8_files.tar.gz`; `CONTENTS.txt` lists its 50 repository-relative
members.

## Integrity

From this directory, verify the archive before use:

```bash
sha256sum -c SHA256SUMS
tar -tzf epit24_v8_files.tar.gz
```

Expected archive SHA-256:

```text
3bf769da1a2b5abca67dbd72279eead0a199e0eb51fedfc7a824d5155480bbf0
```

## Safe restoration

Do not extract this archive over an active coefficient-variation run. First
finish that study and commit or otherwise preserve the then-current working
tree. The preferred workflow is to unpack into a temporary directory, review
the diff against the later code, and integrate it deliberately:

```bash
mkdir -p /tmp/epit24_v8_restore
tar -xzf epit24_v8_files.tar.gz -C /tmp/epit24_v8_restore
diff -ru /home/fcolanto/projects/tabicl /tmp/epit24_v8_restore
```

For an exact overwrite after that review, run from the repository root:

```bash
tar -xzf corrosion_datasets/analysis/epit_pipeline/snapshots/epit24_v8_20260928/epit24_v8_files.tar.gz
```

Extraction overwrites only the listed members; it does not delete other files.
In particular, the archived `src/tabicl/prior/dataset.py` contains both the
coefficient-variation support and the later v8 schema integration as they stood
at snapshot time.

## Snapshot status

At capture time, the v8 implementation had passed 290 tests with 2 skipped.
The `target_rules_v3` results are direct development-fold formula evaluations,
not a trained v8 model comparison. No v3 rule had been promoted to the active
synthetic generator, and the final test split was not used for rule selection.
