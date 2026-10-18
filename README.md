# Predicting At-Risk Students in Online Learning

**Traditional machine learning vs deep learning on the Open University Learning Analytics Dataset (OULAD)**

Aderline Gashugi, African Leadership University, Machine Learning summative project

| | Link |
|---|---|
| Report | _add link_ |
| Google Drive (raw dataset + saved models) | _[Link](https://drive.google.com/drive/folders/1BdUGR3Fn_NW-Zt01Bxzt_0B8QNr_Vt0F?usp=sharing)_ |
| Video presentation | _add link_ |

## The problem

Many students who enrol in online courses never finish. Tutors can help, but only if they know early which students are struggling. This project predicts, **using only data from the first 28 days of a course**, whether a student will end the course with a Fail or Withdrawn result (at risk) or with a Pass or Distinction (not at risk).

## Dataset

[Open University Learning Analytics Dataset](https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset) (Kuzilek, Hlosta and Zdrahal, 2017), CC BY 4.0.

* 32,593 student-course records across 7 modules and 22 course presentations
* 10,655,280 rows of daily clicks in the virtual learning environment
* 173,912 assessment submissions

The raw data is **not** in this repository. The notebook downloads it from UCI the first time it runs. A zipped copy is also in the Google Drive folder.

After removing students who had already withdrawn before day 28, **27,535** records remain (44.1% at risk).

## Results

| Model | Experiment | Val ROC-AUC | Test ROC-AUC | Test recall at tuned threshold |
|---|---|---|---|---|
| Logistic Regression, registration data only (baseline) | ML-01 | 0.660 | 0.654 | |
| Hist Gradient Boosting, tuned (best traditional) | ML-08 | 0.783 | **0.792** | 0.807 |
| Two-input network, MLP + GRU on daily clicks (best deep) | DL-09 | **0.786** | 0.790 | 0.795 |

* Feature engineering added **+0.113** validation AUC to the same Logistic Regression (ML-01 to ML-03).
* The best traditional and best deep models are **statistically tied**: the 95% bootstrap interval for their test AUC difference is -0.003 to +0.007.
* Without regularisation the MLP overfits after 9 epochs (DL-01). With batch norm, dropout and early stopping it matches gradient boosting.
* Most missed students look engaged in month one and withdraw much later (median day 125), which a 28-day model cannot see.

All 19 experiments are in `results/experiments.csv` and in the notebook's results table.

## Repository layout

```
at_risk_students_oulad.ipynb   main notebook, runs top to bottom (about 4 minutes on a laptop CPU)
requirements.txt               dependencies
figures/                       every figure used in the report
results/experiments.csv        one row per experiment: settings, split, all metrics
logs/                          TensorBoard logs, one folder per experiment ID
```

Not committed (in Google Drive instead): `data/` (raw dataset) and `models/` (saved weights).

## How to run

**Google Colab:** open the notebook, then *Runtime > Run all*. Everything it needs is already installed on Colab, and it downloads the data itself.

**Locally:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook at_risk_students_oulad.ipynb
```

To use the Drive copy of the data instead of downloading it, unzip `OULAD_raw_dataset_UCI.zip` into `data/`.

**TensorBoard:**

```bash
tensorboard --logdir logs
```

The Scalars tab shows per-epoch loss and AUC for every Keras run, and the HParams tab compares the settings and final metrics of all 19 experiments.

## Reproducibility

* One seed (42) for Python, NumPy and TensorFlow, reset before every Keras model, plus deterministic TensorFlow ops
* The train/validation/test split is grouped by student ID, so no student appears in two sets
* All preprocessing is inside a scikit-learn pipeline fitted on the training set only
* Running the notebook twice from scratch gave identical numbers for all 19 experiments

## Saved models (Google Drive)

| File | What it is |
|---|---|
| `best_traditional_ML-08.joblib` | full scikit-learn pipeline (preprocessing + Hist Gradient Boosting) |
| `best_deep_DL-09.keras` | Keras two-input model (tabular + daily click sequence) |
| `deep_model_preprocessor.joblib` | fitted preprocessing for the Keras model's tabular input |
| `model_card.json` | cutoff day, feature list and the decision thresholds chosen on validation |

## Acknowledgements

The dataset is by the Open University (Kuzilek, J., Hlosta, M. and Zdrahal, Z., "Open University Learning Analytics dataset," *Scientific Data*, vol. 4, 170171, 2017). No pretrained models were used.
