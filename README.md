# Wait or Go? Bus travel time prediction: ML vs historical averages

CSC340 Artificial Intelligence, SIT KMUTT. Group project.

**Research question:** Can a machine-learning model predict bus travel time between stops more accurately than the historical-average methods a commuter uses today?

**Answer (test set, 9–21 Sep 2024):** Yes, but by less than our 15% goal. Tuned XGBoost cuts the per-segment error from 36.07 s to 32.09 s (**−11.0% MAE**) compared with the best non-AI baseline (segment × hour average). Over an 8-stop journey, Random Forest cuts the error from 205.5 s to 187.3 s (−8.9%). For On time / Late, a Random Forest classifier catches 68% of late journeys (F1 0.57), while the historical average catches none.

## Quick start: run the app from GitHub

The demo app ("Wait or Go?") needs no extra data. The trained model and lookup tables are in the repo.

```bash
git clone https://github.com/pinkerparker/csc340-wait-or-go.git
cd csc340-wait-or-go
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/app.py
```

Then open http://localhost:8501. Next time: `cd csc340-wait-or-go`, `source .venv/bin/activate`, `streamlit run app/app.py`.
On macOS, if XGBoost complains about `libomp`, run `brew install libomp`.

To re-run the notebook or retrain the models, first download the raw data (step 1 below).

## Folder layout

```
bus-travel-time/
├── data/
│   ├── segment_level_data.csv      # download from Zenodo (see below), NOT included
│   └── weather_astana_hourly.csv   # Open-Meteo, Astana, hourly, 29 Jul – 21 Sep 2024 (included)
├── src/
│   ├── features.py                 # feature lists + real-time traffic helper
│   ├── prepare.py                  # cleaning + features  -> data/*.parquet, data/lookup.joblib
│   └── train.py                    # baselines, models, tuning, classification -> models/, reports/
├── notebooks/bus_travel_time.ipynb # same steps, executed, with outputs
├── app/app.py                      # Streamlit demo ("Wait or Go?")
├── models/                         # saved models (*.joblib; xgboost_model.json is used by the app)
├── reports/                        # results tables (*.csv), charts (*.png), round1/ = first run
├── requirements.txt
└── README.md
```

## Run it step by step

Tested with Python 3.11–3.13.

**1. Get the data.** Download `segment_level_data.csv` from Zenodo, DOI [10.5281/zenodo.15769359](https://doi.org/10.5281/zenodo.15769359), and put it in `data/`.

**2. Install** (from inside the `bus-travel-time` folder):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On macOS, if XGBoost complains about `libomp`, run `brew install libomp`.

**3. Prepare the data** (~20 s):

```bash
python src/prepare.py
```

**4. Train and evaluate** (~14 min on a 2-core machine; the saved models in `models/` let you skip this):

```bash
python src/train.py
```

**5. Open the notebook** (re-runs prepare, loads the saved models, recomputes every metric, ~40 s):

```bash
jupyter notebook notebooks/bus_travel_time.ipynb
```

**6. Start the demo:**

```bash
streamlit run app/app.py
```

Then open http://localhost:8501. Choose route, direction, origin and destination stop, the time you reach the stop, your deadline, rain, and whether the previous bus was slow. The app shows the estimated wait, predicted travel time, arrival time, On time / Late and a recommendation. The second tab shows model vs baseline results.

## Method in short

| Step | What we did |
|---|---|
| Data | 785,976 bus × segment records, 3 routes, 19,769 trips, 29 Jul – 21 Sep 2024 (Astana, Kazakhstan) |
| Routes | The file has no `route_id`. Trips were clustered by visited stops (KMeans, k = 3 per direction) and paired across directions by shared buses → Routes A, B, C. The file does not say which is official route 10, 12 or 46. |
| Cleaning | Removed run time ≤ 0 s (925 rows) and > 1 h (105 rows), and segments seen < 50 times in train (5,011 rows). Capped each segment at its train 1st–99th percentile (15,537 values). See `reports/cleaning_log.csv`. |
| Features | Hour, day of week, weekend, KZ holiday, route, direction, segment, position, segment and segment × hour averages (train only), previous bus's run time on the segment (only if it had already arrived), headway at the stop, rain, temperature. Round 2 added route traffic (last 15 min) and this bus's delay so far. |
| Split | By time: 6 weeks train (29 Jul – 8 Sep), 2 weeks test (9 – 21 Sep). No random split. |
| Baselines (no AI) | Mean per segment; mean per segment × hour |
| Models | Linear Regression, Random Forest, XGBoost; tuned with RandomizedSearchCV + TimeSeriesSplit (3 folds) |
| Classification | 8-stop journey is Late if it takes > 1.2 × its usual (median) time. Logistic Regression vs Random Forest, both with class weights. All features frozen at the moment the commuter is at the origin stop. |

## Results (test set)

| Model | MAE (s) | RMSE (s) | MAPE % | R² | vs best baseline |
|---|---|---|---|---|---|
| Baseline: mean per segment | 41.01 | 124.64 | 76.6 | 0.462 | −13.7% |
| Baseline: mean per segment × hour | 36.07 | 103.43 | 65.5 | 0.630 | 0.0% |
| Linear Regression | 35.94 | 102.37 | 66.7 | 0.637 | +0.4% |
| Random Forest (tuned) | 33.50 | 96.52 | 61.8 | 0.677 | +7.1% |
| **XGBoost (tuned)** | **32.09** | 99.64 | 52.3 | 0.656 | **+11.0%** |

All numbers come from `reports/results_*.csv`, produced by `src/train.py`.

**The recommendation is not AI.** It is a fixed rule on top of the predicted arrival time: ≥ 5 min spare → wait for the bus; between 5 min spare and 3 min late → "bus is risky, consider a motorcycle taxi"; more than 3 min late → switch to Grab / motorcycle taxi.

## Data sources

* Mansurova, A., Mussina, A., Aubakirov, S., Nugumanova, A., & Yedilkhan, D. (2025). From raw GPS to GTFS: A real-world open dataset for bus travel time prediction. *Data, 10*(8), 119. https://doi.org/10.3390/data10080119 (dataset CC BY 4.0)
* Open-Meteo. (n.d.). *Historical weather API*. https://open-meteo.com/ (Astana, 51.17 N 71.45 E, hourly precipitation and temperature, local time UTC+5)
