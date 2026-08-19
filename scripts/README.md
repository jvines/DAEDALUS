# Reproduction scripts

Every script in this directory reproduces a result reported in the
`daedalus` methods paper. Each writes its outputs (`.npz`, `.json`)
alongside itself; those artefacts are regenerable and are not tracked.

Run from the project root, e.g. `python scripts/diabetes_paper.py`.

These scripts need more than the core package. Install the extra that
covers them:

```bash
pip install -e ".[reproduce]"
```

`real_data_chains.py` must be run before `real_data_confirmatory.py` and
`sdss_emission_lines.py`, which patch the `real_data_results.npz` it
produces. The HD 10180 scripts share one VizieR download, cached on first
use by `hd10180_lovis_labelled.load_harps()`.

Scripts that fetch data do so from the public archives named in the
paper's Data Availability statement (VizieR, MAST, SDSS); the diabetes
design matrix comes from `sklearn.datasets.load_diabetes`.

## Validation

| Script | Reproduces |
|---|---|
| `diabetes_paper.py` | JZS diabetes inclusion probabilities against the enumerated 1024-model reference (Section 4.1) |
| `diabetes_dummy_ablation.py` | Invariance of those probabilities to the dummy-coordinate prior width, `W` ∈ {0.1, 1, 10} (Section 4.1) |
| `insertion_index.py` | Insertion-index diagnostic of Fowlie, Handley & Su (2020), imported by the application scripts below (Section 4.2) |
| `insertion_index_validate.py` | Diagnostic's own calibration: a well-mixed reference run and a deliberately under-mixed run (Section 4.2) |
| `insertion_index_hd10180.py` | Insertion-index test applied to the HD 10180 labelled problem (Section 4.2) |
| `sbc_paper.py` | Simulation-based calibration at M = 400 trials (Appendix, SBC) |
| `sbc_diagnostic_nlive.py` | SBC inclusion bias at `n_live` = 800, showing it is not finite-live-point in origin (Appendix, SBC) |
| `sbc_rao_blackwell_validate.py` | Rao-Blackwellised inclusion estimator removing the empirical-frequency bias (Appendix, SBC) |
| `sbc_is_rb_validate.py` | Importance-sampled form of that estimator against the closed-form conditional logit (Appendix, SBC) |
| `polynomial_scaling.py` | High-`g` scaling sweep over (`K_max`, `K_true`) on polynomial-degree selection (Appendix, high-g) |
| `polynomial_scaling_extra.py` | `n_live` sweep at the hardest cell, `K_max` = 20, `g` = 21 (Appendix, Table: high-g extra) |

## Real-data applications

| Script | Reproduces |
|---|---|
| `hd10180_lovis_labelled.py` | HD 10180 labelled candidate-confirmation problem definition, imported by the scripts below (Section 5.1) |
| `hd10180_paper_run.py` | The reported labelled inclusions and the injected fake-eighth-candidate control at four test periods (Section 5.1) |
| `hd10180_fake_planet.py` | Fake-slot construction and the multi-seed discrimination control (Section 5.1) |
| `hd10180_fixed_dim_bf.py` | Fixed-dimensional Bayes-factor cross-check of the near-threshold candidate b (Section 5.1) |
| `hd10180_rb_b.py` | Rao-Blackwellised inclusion for candidate b on the labelled chain (Section 5.1) |
| `hd10180_de_campaign.py` | Labelled campaign and the blind wide-prior eight-slot search with GLS-informed births (Section 5.1, Appendix blind search) |
| `sx_car_reduce.py` | Reduction of SX Car TESS Sectors 63–64 SAP photometry (Section 5.2) |
| `sx_car_run.py` | Cepheid Fourier-harmonic selection and the recovered Simon–Lee parameters (Section 5.2, Appendix SX Car) |
| `kic_peak_bagging.py` | KIC 6603624 asteroseismic peak bagging, self-contained from the MAST fetch onward (Section 5.3) |
| `real_data_chains.py` | Driver producing `real_data_results.npz` for the real-data applications (Section 5) |
| `real_data_confirmatory.py` | KIC 6603624, AU Mic and SDSS chains at the dimension-aware chain budget (Sections 5.3–5.5) |
| `sdss_emission_lines.py` | SDSS star-forming galaxy emission-line detection with the DE within-model kernel (Section 5.5) |
