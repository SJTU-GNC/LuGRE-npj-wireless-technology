# Third-party sources and licence boundaries

The MIT licence in this repository covers the author-contributed software. It does not relicense third-party dependencies, upstream data, source publications or source-derived numerical material. The authors' CC BY 4.0 choice for their own companion data likewise does not replace existing upstream rights. Original third-party observation archives, orbit/attitude products, SPICE kernels, antenna workbooks and publication PDFs are not bundled.

## Galileo GRAP numerical grid in the companion data

File relative to the companion data root: `table/external_reference/galileo_grap_eirp_grid.csv`.

Source: the European Commission Joint Research Centre's [Galileo Reference Antenna Pattern model](https://joint-research-centre.ec.europa.eu/scientific-activities/galileo-reference-antenna-pattern-model_en), developed with DG DEFIS and ESA. Copyright European Union. The source workbooks are `GRAP_File_E1_.xlsx` and `GRAP_File_E5a_.xlsx` in [GRAP_metadata.zip](https://joint-research-centre.ec.europa.eu/document/download/bbba68eb-d8fb-417e-acfd-1606de7816c4_en?filename=GRAP_metadata.zip).

The [European Commission legal notice](https://commission.europa.eu/legal-notice_en) licenses EU-owned website content under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) unless another notice applies. Its reuse conditions require attribution and identification of changes; third-party content and other excluded rights are not covered by that default. The source ZIP's four workbook XML packages were inspected on 30 September 2026 and contained no separate copyright or licence notice. The accompanying [JRC technical report](https://publications.jrc.ec.europa.eu/repository/bitstream/JRC135110/JRC135110_01.pdf), page 2, also explicitly specifies CC BY 4.0, subject to its stated exceptions.

Changes made by this study: E1/E5a EIRP and upper/lower 95% bound sheets were parsed and reshaped into a row-oriented CSV, with signal identifiers, angular coordinates, units and provenance fields. The transformation is implemented in `original_sources/script/prepare_gnss_transmit_physics_inputs.py`. The grid is a source-derived numerical representation, not a new independent antenna measurement; its upstream attribution and applicable licence remain with it. No endorsement by the European Union, JRC, DG DEFIS or ESA is implied.

Source citation: European Commission, Joint Research Centre; Menzione, F.; Sgammini, M.; Paonni, M. *Reconstruction of Galileo Constellation Antenna Pattern for Space Service Volume Applications*. Publications Office of the European Union (2024), JRC135110. [https://doi.org/10.2760/765842](https://doi.org/10.2760/765842).

## LuGRE observations and digitized publication curves

The upstream [LuGRE mission dataset](https://doi.org/10.5281/zenodo.16411687) is documented separately from this study's derived-data deposit. Its CC BY 4.0 attribution and source identity must be retained when reusing processed observations.

The companion data includes `data/external_reference/lugre_antenna/LuGRE_Fig3_gain_outer_envelope_digitized.csv`, thermal inputs under `analysis/thermal_sensitivity_v8/alex_temperature_inputs/`, and published-comparison digitizations. These are author-produced numerical extractions or transformations of figures in [GNSS Reception at the Moon: First Results of the Lunar GNSS Receiver Experiment (LuGRE)](https://doi.org/10.33012/navi.756). They are not the original publication images and are not raw calibrated telemetry. Retain the publication citation, figure identification and supplied digitization provenance. The authors' licence does not override any rights applicable to the original publication or third-party material in it.

## Other upstream inputs and dependencies

See [PUBLIC_DATA_SOURCES.md](PUBLIC_DATA_SOURCES.md), `public_input_inventory.json` and `REFERENCES.bib` for CODE/IGS, NASA/NAIF, GPS NAVCEN, ANTEX and lunar-gravity sources. Availability through a public URL is not by itself a new licence. Downloaded originals and installed Python dependencies remain subject to their respective provider terms and licences.
