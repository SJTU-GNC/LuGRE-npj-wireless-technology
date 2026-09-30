# Public input data and download links

Checked for this release on 30 September 2026. This document covers third-party inputs; the companion archive supplies the author-derived feature tables, fitted models and numerical results. Public original files are **not** bundled. The core-model verification can run from the supplied derived data without downloading them.

For raw-data reconstruction, `public_input_inventory.json` is the file-level specification: exact product names, dates, download URLs, local SHA256 values, target relative paths, SPICE load order and query settings. Target paths are relative to the isolated analysis workspace created by `code/prepare_workspace.py`, not to the unextracted ZIP.

| Input | Official source | Required subset / purpose |
|---|---|---|
| LuGRE flight observations | [Zenodo record, DOI 10.5281/zenodo.16411687](https://zenodo.org/records/16411687); [download LuGRE.zip](https://zenodo.org/records/16411687/files/LuGRE.zip?download=1) | Version 1. The manifest lists 21 `L0/TLM/TLM_RAW_*.txt` members used for observation processing. Raw IQ/MAT files are not needed for the main trend model. |
| GNSS precise orbits | [IGS MGEX products](https://igs.org/mgex/data-products/) | CODE `COD0MGXFIN` SP3 products for 15 January–16 March 2025; 61 daily files. Exact historical URLs and hashes are listed in the manifest. |
| GNSS transmitter attitudes | [IGS MGEX products](https://igs.org/mgex/data-products/) | CODE ORBEX `ATT.OBX`, 30-second products for the same 61 dates. |
| Spacecraft attitude and frame kernels | [NASA NAIF CLPS SPICE archive](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/); [archive documentation](https://naif.jpl.nasa.gov/pub/naif/pds/pds4/clps/clps_spice/document/spiceds_v004.html) | The manifest pins the 24 kernel files and load order used by the original implementation. A current named CLPS kernel set is not itself a version pin. |
| GPS transmit-antenna products | [USCG NAVCEN technical references](https://www.navcen.uscg.gov/gps-technical-references) | IIR/IIR-M, IIF and GPS III antenna products. The manifest specifies the required 18 files/archive products, including embedded workbooks. |
| Galileo reference antenna patterns | [European Commission JRC GRAP model](https://joint-research-centre.ec.europa.eu/scientific-activities/galileo-reference-antenna-pattern-model_en); [GRAP metadata ZIP](https://joint-research-centre.ec.europa.eu/document/download/bbba68eb-d8fb-417e-acfd-1606de7816c4_en?filename=GRAP_metadata.zip); [reference report](https://publications.jrc.ec.europa.eu/repository/handle/JRC135110) | GRAP Issue 1.0, May 2024; E1 and E5a workbooks. Processed numerical grids are supplied as derived inputs. |
| PRN/SVN/block mapping source | [GSC IGS20 2417 ANTEX snapshot](https://www.gsc-europa.eu/sites/default/files/sites/all/files/igs20_2417.atx.gz) | Needed only to recreate the supplied active mapping CSV. The filename differs from the author's local renamed copy; see the manifest. |
| LuGRE receive-gain and temperature curves | [LuGRE first-results paper](https://doi.org/10.33012/navi.756) | Published Fig. 3 supports gain-envelope digitization; Figs. 11–12 support thermal inputs. Supplied digitizations are derived curve estimates, not raw calibrated temperature telemetry. Publisher PDF/image URLs are in the manifest. |
| Lunar gravity | [NASA PGDA GRGM1200A product](https://pgda.gsfc.nasa.gov/products/50); [PDS coefficient file](https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.tab); [PDS label](https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-5-rdr-v1/grail_1001/shadr/gggrx_1200a_sha.lbl) | Required for the lunar dynamic-trajectory calculation, which uses degree/order 8. Cite Lemoine et al. (2014) and Goossens et al. (2016), as requested by the provider. |
| Historical broadcast-navigation seed | [BKG IGS data archive](https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2025/) | Optional 41 daily mixed-RINEX files for the older seed-geometry builder. Not needed when using the supplied final SP3 geometry features. |
| Receiver reference states | [NASA WebGeocalc](https://naif.jpl.nasa.gov/naif/webgeocalc.html) | Generated calculations, not a static downloadable dataset. Query settings and remaining provenance limits are recorded under `wgc_generated_inputs` in the manifest. |

## Access and version details

- LuGRE ZIP: 256,135,673 bytes, MD5 `cec32df1ca17cb95887762762c16629f`; CC BY 4.0. The correct published archive is `LuGRE.zip`, not the author's local subset name `TLM.zip`. The public record notes an OP23 metadata correction; this release preserves the manuscript's numerical branch.
- The current AIUB directory could not be opened during this audit. CODE product URLs/hashes are retained from the original download manifests. CDDIS mirrors may require NASA Earthdata authentication. No account credentials are included or required by the released core-model verification.
- Some publisher PDF/image endpoints returned access restrictions in this audit. Their official records were checked, and available local paper copies were used to inspect the relevant passages. These PDFs and images are not redistributed.
- The original local GRGM1200A file was truncated at high degree. Its header and the degree-8 coefficients actually used were checked against the complete public file and agree. The manifest records both hashes; a complete fresh download is not expected to have the truncated input's whole-file hash.
- WGC receiver-state queries used target `BGM1_LUGRE`, observer `EARTH`, frame `J2000`, aberration correction `NONE`. The historic downloader defaults to the reverse target/observer order and must not be run unchanged as a recipe for that saved product. The supplied final geometry table supports downstream physical-budget and residual-model work without the large 1-Hz WGC cache. Full raw-to-trajectory regeneration has not been validated by this release.
- A public download link does not establish unrestricted redistribution rights. Follow each provider's terms. No new licence is assigned to third-party files here.

## Figure 1 artwork sources

These are credits for the schematic layout and visual assets, not data sources for the numerical model; artwork is not included in these archives.

- Layout reference: [NASA Artemis I mission map](https://www.nasa.gov/image-detail/artemis-i-mission-map-nov/).
- Earth imagery: [NASA Blue Marble 2002](https://science.nasa.gov/resource/blue-marble-2002/).
- Moon imagery: [NASA/Goddard Space Flight Center full Moon image](https://www.flickr.com/photos/gsfc/5836482263/), adapted under [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/).

Formal source citations are provided in `REFERENCES.bib`. The author-contributed code is licensed under MIT at [SJTU-GNC/LuGRE-npj-wireless-technology](https://github.com/SJTU-GNC/LuGRE-npj-wireless-technology). The authors' original derived data are licensed under CC BY 4.0 at [Zenodo](https://doi.org/10.5281/zenodo.23055100). These licences do not replace upstream third-party terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
