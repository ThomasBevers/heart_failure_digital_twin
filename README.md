# Heart Failure Digital Twin

A patient-specific, closed-loop cardiovascular simulator for heart failure research and education, built on top of the [EchoNet-Dynamic](https://echonet.github.io/dynamic/) dataset/models and the PhysioNet Zigong heart-failure cohort. Ships as a Streamlit app.

> **Research prototype only.** This model is not clinically validated and must not be used for diagnosis, treatment, or patient management.

## What it does

- **Digital twin simulation** — enter a patient's heart rate, blood pressure, and LVEF; the app calibrates a closed-loop 0D cardiovascular model (time-varying-elastance left ventricle, forward-only mitral/aortic valves, arterial + venous compliance) to match it, then lets you run "what-if" scenarios (contractility, preload, afterload, heart-rate changes) and see the resulting pressure/volume traces.
- **EchoNet-based EF prediction** — optionally upload an A4C echo video/DICOM and get an EF prediction from the official pretrained EchoNet-Dynamic checkpoint, instead of typing EF in by hand.
- **EDV/ESV estimation** — optional, trainable video-feature regression model for end-diastolic/end-systolic volume, as a research-grade proxy (not true LV segmentation).


## Project structure

```
.
├── app.py                       # Streamlit UI
├── backend/src              # Core package (self-contained, no external framework deps)
│   ├── patient.py                #   PatientProfile: validated inputs + EDV/ESV resolution
│   ├── physiology.py             #   Closed-loop 0D cardiovascular model
│   ├── twin.py                   #   Baseline calibration + what-if scenario orchestration
│   ├── imaging.py                #   DICOM/video inspection, conversion, preview (no inference)
│   ├── echonet_integration.py    #   EF + EDV/ESV inference against real checkpoints
│   
├── scripts/
│   ├── download_echonet_weights.py  # Fetch the official pretrained EF/segmentation checkpoints
│   ├── train_volume_estimator.py    # Train the optional EDV/ESV estimator (configurable split sizes)
│   ├── inspect_echo_study.py        # CLI: inspect one video's technical metadata
│   └── validate_echo_dataset.py     # CLI: validate a local EchoNet-Dynamic dataset copy
├── vendor/echonet_dynamic/      # Vendored upstream EchoNet-Dynamic repo (untouched)
├── tests/
└── reference/original_echonet_repo/  # Upstream paper-reproduction scripts, kept for citation only
```

## Quick start

```bash
git clone "<repository-link>"
cd <repository-folder-name>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt # or: pip install streamlit pandas numpy scikit-learn joblib opencv-python-headless pydicom torch torchvision

streamlit run app.py
```

Select **"Manual entry"** in the sidebar, fill in heart rate / blood pressure / LVEF, and click **"Create or update twin."** The simulation itself is pure math — this works immediately, with no data download and no training.

## Optional: EchoNet video inference

**EF prediction** needs only a one-time checkpoint download, no training:

```bash
python scripts/download_echonet_weights.py
```

This saves the official checkpoints to `outputs/echonet_weights/`. Once present, the app's "EchoNet (from uploaded video)" mode produces real EF predictions.

**EDV/ESV prediction** is a separate, *trainable* research estimator and needs the [EchoNet-Dynamic dataset](https://echonet.github.io/dynamic/) (not included in this repo — request access via the link above) placed at the project root:

```
EchoNet-Dynamic/
├── FileList.csv
├── VolumeTracings.csv
└── Videos/*.avi
```

Start small to sanity-check the pipeline before committing to a full run:

```bash
python scripts/train_volume_estimator.py --n-train 50 --n-val 20 --n-test 20
```

Then scale up (or drop the flags to use every patient):

```bash
python scripts/train_volume_estimator.py --n-train 2000 --n-val 500 --n-test 500
python scripts/train_volume_estimator.py
```

This writes `outputs/echonet_volume_estimator.joblib`, which the app picks up automatically. Both checkpoint and estimator paths can be overridden via `ECHONET_EF_CHECKPOINT` / `ECHONET_VOLUME_ESTIMATOR` environment variables if you'd rather point at your own.


## The physiology model

`ClosedLoopCardiovascularModel` (`src/heart_twin/physiology.py`) is a 0D lumped-parameter model: a time-varying-elastance left ventricle, idealized forward-only mitral and aortic valves, and arterial + venous compliance, integrated with forward-Euler. Blood volume is conserved across cycles (checked numerically in `tests/`). `twin.py` calibrates `Emax` by bisection so the baseline reproduces the patient's entered LVEF, then applies "what-if" percentage perturbations on top of that fitted baseline — it does not re-fit from new data.

## Testing

```bash
pytest
```

`conftest.py` puts both `src/` and `vendor/echonet_dynamic/` on `sys.path` automatically.

## Data & model provenance

- EchoNet-Dynamic dataset, EF checkpoint, and segmentation checkpoint: [Ouyang et al., *Nature* 2020](https://www.nature.com/articles/s41586-020-2145-8), [echonet.github.io/dynamic](https://echonet.github.io/dynamic/). Downloaded checkpoints are the authors' original releases, used unmodified.
- Zigong cohort: Zhang et al., PhysioNet, "Hospitalized patients with heart failure."
- The vendored `vendor/echonet_dynamic/` code is the upstream repo, included unmodified for citation and compatibility; project-specific logic lives entirely in `src/heart_twin/`.

## Limitations

- The EDV/ESV estimator is a video-feature regression proxy, not true LV-segmentation-based volumetry — it is not validated and should be treated as exploratory.
- EF inference uses fixed Kinetics normalization statistics rather than the paper's exact per-dataset training statistics, as a practical approximation.
- Nothing in this repository is a clinical decision tool. See the in-app disclaimer.
