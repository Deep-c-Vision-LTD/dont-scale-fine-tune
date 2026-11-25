# Don't Scale, Fine-Tune

Code and results for:

> Hamit Soyel. **Don't Scale, Fine-Tune: One Hundred Fixation Maps Beat State-of-the-Art
> Saliency Models on Advertising Images.** IEEE Signal Processing Letters (submitted).

## The result in one paragraph

General-purpose saliency models have plateaued on advertising images: DeepGaze IIE (2021)
reaches CC 0.766 on the ADD1000 advertising fixation benchmark and its far larger 2025
successor DeepGazeMSDB reaches only CC 0.777, at roughly 50x the inference cost. The
strongest zero-shot model is the unified designed-media SUM (WACV 2025) at CC 0.836 with
its best domain token (e-commerce). Training **xSoyelNet** (a 3.7M-parameter ASPP
readout with a Gaussian finalizer on a frozen DINOv2-base backbone) on ADD1000's own
fixation maps reaches **CC 0.895±0.003** (5-fold CV) warm-started, and **CC 0.898±0.003
from a randomly initialized readout**: the decisive ingredient is benchmark-matched
in-domain data on strong frozen features, not saliency-specific pretraining. As few as
**25 fixation maps already beat both general models** (CC 0.819 warm-started) and **100
beat every model tested including SUM** (CC 0.862 vs 0.841 on the same held-out test
set); the warm-start advantage is confined to the low-data regime and vanishes by n≈400.

**In-domain baselines (fair comparison).** To separate the effect of in-domain data from
the representation, we also fine-tune DeepGaze IIE in-domain on ADD1000 under the identical
seed-0 5-fold protocol (readout adapted, backbone frozen, common 384 scoring). It rises
from **0.766 zero-shot to 0.850±0.004**: most of the zero-shot gap is closed by the data,
confirming benchmark-matched fixation data as the larger, model-agnostic lever. At equal
data and comparable scale, xSoyelNet's 3.7M readout still leads by **+0.045 CC** (0.895 vs
0.850). Two levers compound: in-domain data lifts any model, and a strong frozen general
representation adds a further margin on top. Script: `ft_deepgaze_iie_indomain.py`; result:
`results/deepgaze_indomain_add1000.json`. A near-duplicate scan
(`results/near_dup_analysis.json`) finds only 5/1000 similar image pairs (4 cross-fold),
too few to explain the gap.

**Module ablation (paper Table II).** Removing one readout component at a time under the
identical 5-fold protocol shows no single component is individually large (CC, full 0.895):
plain 2-conv readout instead of ASPP 0.892, no Gaussian finalizer 0.892, KL-only loss 0.890,
no center bias 0.895; every effect is larger on SIM/KLD than on CC. The result comes from
the frozen representation and in-domain data, each readout choice adding a small refinement.
Script: `ablation_add1000.py`; result: `results/ablation_add1000.json`.

**Cross-corpus control (paper Table I, designed-media row).** The same architecture trained
from scratch on the Imp1k designed-media importance dataset (Fosco 2020; ads, infographics,
posters, web/mobile UIs), at the identical 448-input protocol, then applied zero-shot to
ADD1000, reaches only **CC 0.622** (SIM 0.632, KLD 0.471), below both general zero-shot
models. Imp1k uses importance annotations (mouse interaction) rather than eye-tracking
fixations, so similar-domain data is no substitute for benchmark-matched data. The finalizer
bandwidth is selected on Imp1k, not on ADD1000 (0.635 if tuned directly on ADD1000).
Script: `crosscorpus_imp1k_add1000.py`; result: `results/crosscorpus_imp1k_add1000.json`.

![sample efficiency](figures/fig_sampeff.png)

## Fully reproducible without any private weights

The paper's warm-start checkpoint is not distributed (it derives from a commercial
product), but the **from-scratch condition needs no checkpoint at all** and matches or
exceeds the warm-started headline numbers (CC 0.898 at n=800). Everything in the main
results table is therefore reproducible from this repo plus two public downloads
(ADD1000 and the auto-downloaded DINOv2/DeepGaze weights).

## Reproduction

1. **Get ADD1000** (third-party advertising creative, not redistributed here):
   https://github.com/lscumt/ADD1000. Extract so you have the `ADD1000_*/` image folders
   and `GroundTruth1000_*/` density-map folders (the "ext" layout). For the baseline
   scripts, also make a "flat" convenience copy: `flat/images/*.jpg`, `flat/maps/*.png`
   (basenames pair images to maps).
2. `pip install -r requirements.txt`
3. Edit the `EDIT ME` path constants at the top of each script.

```bash
# Main result (Table I, from-scratch row -- no checkpoint needed): CC 0.898 +/- 0.003
python kfold_train_eval.py --folds 5 --epochs 60 --scratch

# Sample-efficiency curve (Fig. 2), from-scratch condition (CKPT=None default)
python sample_efficiency.py

# Baselines (Table I): DeepGaze IIE (CC 0.766) and DeepGazeMSDB (CC 0.777),
# both with a held-out 2-fold ad-domain center-bias prior
python eval_deepgaze_iie.py
python eval_deepgaze_msdb.py   # ~8 s/image on CPU, ~2.3 h; resumes from msdb_checkpoint.json

# SUM (WACV'25) baseline: CUDA required (mamba-ssm). eval_sum_add1000.py is the
# GPU entry point we used; it installs SUM from the authors' repo,
# loads their HuggingFace checkpoint, and sweeps all 4 domain tokens (best: CC 0.836).
# Per-image and summary outputs are pre-computed in results/sum_add1000_*.json.

# Figures (Fig. 1, the framework diagram, is drawn in the LaTeX source)
python make_qualitative_fig.py      # qualitative panels, full width (needs a trained checkpoint from kfold_train_eval.py)
python make_qualitative_fig_col.py  # Fig. 2 as printed: the same panels re-tiled at single-column width
python make_sampeff_fig.py          # sample-efficiency plot, full width, from results/*.json
python make_sampeff_fig_col.py      # Fig. 3 as printed: same data at single-column width
```

Pre-computed result JSONs for every table/figure entry are in `results/`.

## Repository layout

```
kfold_train_eval.py        5-fold CV train+eval (--scratch for the no-warm-start condition)
sample_efficiency.py       CC vs training-set size (25..800), fixed held-out test set
eval_deepgaze_iie.py       DeepGaze IIE zero-shot baseline, 2-fold held-out center-bias
eval_deepgaze_msdb.py      DeepGazeMSDB (ICCV'25) baseline, generalization mode, resumable
eval_sum_add1000.py        SUM (WACV'25) baseline on a CUDA GPU, all 4 domain tokens
model_aspp.py              xSoyelNet architecture: frozen DINOv2-base + ASPP readout
make_qualitative_fig.py    qualitative panels (full width)
make_qualitative_fig_col.py Fig. 2 as printed (single column)
make_sampeff_fig.py        sample-efficiency plot (full width)
make_sampeff_fig_col.py    Fig. 3 as printed (single column)
results/                   pre-computed JSONs backing every number in the paper
figures/                   pre-rendered paper figures
```

## Notes on protocol

- **Gaussian finalizer**: predictions are smoothed with sigma selected on training folds
  only (grid includes 0). Selected values are stable: sigma=12 for the untrained
  warm-started model, sigma=4 after training.
- **Metrics**: CC, SIM, KLD (distribution-based). NSS/AUC are not computable on ADD1000,
  which releases density maps only, no discrete fixation coordinates.
- **No leakage**: all reported numbers are on held-out folds/test sets; baseline
  center-bias priors are fit on the complementary fold; figure examples are drawn from
  the displayed checkpoint's held-out split.
- **Known upstream bug**: `DeepGazeMSDB.train()` does not `return self`, so a chained
  `.eval()` returns `None`. Call `model.eval()` as its own statement (our scripts do).

## License / data

- Code released under the MIT License (see `LICENSE`).
- **ADD1000 images and ground-truth maps are NOT redistributed**: obtain them from the
  dataset's own release page and observe its terms. This repo contains only code, derived
  aggregate results, and figure renderings.
- DINOv2 weights: Apache 2.0 (auto-downloaded). DeepGaze weights: see the
  matthias-k/DeepGaze repository for their terms; used here for benchmarking only.
