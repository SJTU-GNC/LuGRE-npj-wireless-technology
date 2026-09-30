# LuGRE cislunar C/N0 model — code

Version 1.0.0 accompanies the manuscript *AI-driven cislunar GNSS channel modelling from lunar observations*.

Code repository: [SJTU-GNC/LuGRE-npj-wireless-technology](https://github.com/SJTU-GNC/LuGRE-npj-wireless-technology).

Companion derived-data deposit: [Zenodo, version 1.0.0](https://doi.org/10.5281/zenodo.23055100). The upstream LuGRE mission DOI in the source inventory identifies the original observations, not this study's derived-data deposit.

Extract the companion data archive to `data/` and place this repository in a sibling `code/` directory. For example, clone the repository with `git clone https://github.com/SJTU-GNC/LuGRE-npj-wireless-technology.git code`. Public original observations, ephemerides, attitude products, antenna products and papers are not distributed here. Their sources and download requirements are documented in the data package's `data/PUBLIC_DATA_SOURCES.md` and `data/public_input_inventory.json`; [PUBLIC_DATA_SOURCES.md](PUBLIC_DATA_SOURCES.md) and `public_input_inventory.json` here provide the same input inventory for code-only readers.

## Start here

Use Python 3.12 and install `requirements-core.txt` in a separate environment. From this directory run:

```text
python reproduce_core.py --data ../data --out ../verification_output --refit
```

The command loads the version-locked final model, regenerates its predictions, refits the selected configuration on the original training + validation samples, and refits the training-only model used for validation. It checks the results against the released prediction table. It does not repeat model selection, modify the supplied data or download public inputs.

Verified reference RMSE values (dB-Hz):

| Evaluation | Physical baseline | Baseline + beta | Final model |
|---|---:|---:|---:|
| Internal test, 1,800 samples | 7.604879 | 2.159078 | 1.147083 |
| OP2/OP21/OP27/OP74, 4,112 samples | 8.420941 | 1.842255 | 1.635563 |

The training-only selected model gives validation RMSE 1.068969 dB-Hz on 1,897 samples. Validation is not an independent test. `data/verification/` records the original release checks; the command creates a separate new check.

The runner applies the same output smoothing and split isolation as the original implementation. The stored target is the processed one-minute trend, not an unprocessed 1-Hz sample. See [the data dictionary](../data/DATA_DICTIONARY.md) for field meanings. The final model has 44 inputs and was fitted to 9,894 eligible training + validation samples. Its SHA256 is `4ce4d4fd0344d1b500e182cb1e21852335e1a68cd856e20a6a86d3d23e8bceb8`.

## Original analysis and preprocessing code

`original_sources/` contains the selected original training, preprocessing, attribution, antenna, thermal, chronological and plotting sources, plus their local Python import dependencies. Scientific computations are retained. Reviewer-only wording in two thermal scripts was removed from the release copies; this is recorded in `SOURCE_MANIFEST.csv`. Original working-directory files were not changed.

These scripts were written for the author's directory layout. Prepare an isolated workspace before using them:

```text
python prepare_workspace.py --data ../data --workspace ../analysis_workspace
```

The target must be new or empty. This copies the released data and source files into a common layout, adapts known author-specific paths, removes internal planning-file hashing dependencies, and writes `PORTABILITY_CHANGES.json`. It does not change model parameters or experimental values. Install `requirements-analysis.txt` for the corresponding optional analyses. Run scripts only in this isolated workspace, because legacy scripts may overwrite their own output tables.

Representative entry points, after preparing the workspace:

| Analysis | Script relative to workspace |
|---|---|
| Original selected-model training from physical rows + downloaded telemetry | `script/train_cn0_trend_residual_tuned_no_leakage.py` |
| Original validation parameter/feature search | `script/tune_cn0_trend_residual_no_leakage.py` |
| Leave-one-operation-out | `script/run_cn0_leave_one_operation_out.py` |
| Grouped model-input attribution | `analysis/phase_compensation_attribution_v4/attrib_group_shap.py` |
| Antenna-reference sensitivity | `analysis/antenna_refit_v11/refit_antenna_references.py` |
| Temperature-refit sensitivity | `analysis/thermal_refit_v9/refit_temperature_models.py` |
| Temperature/correction time series | `analysis/thermal_timeseries_v15/build_diagnostics.py` |
| Early-operation calibration | `analysis/chronological_prediction_v13/run_chronological.py` |

The core verification above was executed successfully. The temperature-refit and chronological-calibration scripts were also rerun in an isolated workspace: their released metrics and prediction tables agreed exactly with the recomputed values. See `data/verification/release_verification.json`. All LOPO folds, attribution combinations, raw-data reconstruction and figure renderers were not rerun. Full raw reconstruction additionally requires the public files in the input manifest. In particular, the historic WGC downloader's default observer/target settings are not a complete recipe for the saved receiver-state product; use the recorded calculation settings, not the default unchanged. Final figure artwork includes manual layout edits and is not promised to be pixel-identical to a fresh plot.

## Version and release status

This release follows the original numerical branch used in the current manuscript, not the later coordinate-consistency or hyperparameter experiments (`v16`/`v17`). No runtime installation, credentials, correspondence, manuscript drafts or third-party PDF/kernel/observation binaries are included. Check `SHA256SUMS.txt` before loading joblib files; pickle/joblib files must only be loaded from trusted sources.

## Licence and citation

The author-contributed code is released under the [MIT License](LICENSE), copyright 2026 LuGRE cislunar channel modelling contributors. Cite this software using [CITATION.cff](CITATION.cff).

The authors have selected CC BY 4.0 for their own derived data in the companion data deposit. This choice does not transfer ownership of, or override the terms for, upstream third-party products, numerical grids or digitized source curves. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the source references in the data package. Third-party Python dependencies retain their own licences and are not bundled.
