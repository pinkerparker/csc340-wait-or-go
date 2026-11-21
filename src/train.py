"""
train.py - baselines vs ML models for bus segment travel time (regression) and
On time / Late for a multi-stop journey (classification).

Run after prepare.py:  python src/train.py
Outputs: models/*.joblib, reports/*.csv, reports/*.png
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, f1_score, recall_score, precision_score,
                             confusion_matrix)
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
REPORTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
SEED = 42
TUNE_ROWS = 120_000        # time-ordered random sample of train used for the search (compute)
NJOBS = -1

NAVY, ORANGE, GRAY, CREAM, INK, MUTED = "#14213D", "#B5541A", "#9AA3B2", "#F6F4EE", "#1F2430", "#5B6170"
plt.rcParams.update({"figure.facecolor": CREAM, "axes.facecolor": CREAM, "savefig.facecolor": CREAM,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#E2DED3", "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 11, "axes.titleweight": "bold",
                     "axes.titlesize": 13, "axes.titlecolor": INK})

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, FEATURES_R1   # round 2 = round 1 + route_traffic, bus_delay (+ flags)
ROUND1_XGB = {"subsample": 0.9, "objective": "reg:absoluteerror", "n_estimators": 900,
              "min_child_weight": 5, "max_depth": 6, "learning_rate": 0.05, "colsample_bytree": 0.7}
LR_FEATURES = [f for f in FEATURES if f != "seg_id"]      # an ID code is meaningless to a linear model
TARGET = "run_time"


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE_s": mean_absolute_error(y, p),
            "RMSE_s": float(np.sqrt(mean_squared_error(y, p))),
            "MAPE_%": float(np.mean(np.abs(y - p) / y) * 100),
            "R2": r2_score(y, p)}


def label_bars(ax, bars, fmt="{:.1f}"):
    for b in bars:
        ax.annotate(fmt.format(b.get_width()), (b.get_width(), b.get_y() + b.get_height() / 2),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=10, color=INK)


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA / "features.parquet")
    tr, te = df[df.is_train].reset_index(drop=True), df[~df.is_train].reset_index(drop=True)
    print(f"train {len(tr):,} rows ({tr.date.min():%d %b}-{tr.date.max():%d %b}), "
          f"test {len(te):,} rows ({te.date.min():%d %b}-{te.date.max():%d %b})")
    Xtr, ytr, Xte, yte = tr[FEATURES], tr[TARGET], te[FEATURES], te[TARGET]

    rows, preds = [], {}
    # ---------- baselines (no AI): what a commuter can do with historical averages ----------
    preds["Baseline: mean per segment"] = te.seg_mean.values
    preds["Baseline: mean per segment x hour"] = te.seg_hour_mean.values
    for k in preds:
        rows.append({"model": k, "type": "baseline", "tuned": "-", **metrics(yte, preds[k])})

    # ---------- tuning sample (kept in time order for TimeSeriesSplit) ----------
    samp = tr.sample(TUNE_ROWS, random_state=SEED).sort_values("start_time")
    tscv = TimeSeriesSplit(n_splits=3)

    # ---------- Linear Regression ----------
    lr = make_pipeline(StandardScaler(), LinearRegression()).fit(tr[LR_FEATURES], ytr)
    preds["Linear Regression"] = lr.predict(te[LR_FEATURES])
    rows.append({"model": "Linear Regression", "type": "ML", "tuned": "no tuning needed",
                 **metrics(yte, preds["Linear Regression"])})
    joblib.dump({"model": lr, "features": LR_FEATURES}, MODELS / "linear_regression.joblib")

    # ---------- Random Forest: default vs tuned ----------
    # each tree sees a 25% bootstrap sample (max_samples) to keep memory/time manageable on a laptop
    print("Random Forest (default) ...")
    rf0 = RandomForestRegressor(n_estimators=100, max_samples=0.25, n_jobs=NJOBS, random_state=SEED)
    rf0.fit(Xtr, ytr); p = rf0.predict(Xte)
    rows.append({"model": "Random Forest", "type": "ML", "tuned": "before (default)", **metrics(yte, p)})
    del rf0
    print("Random Forest (search) ...")
    rf_space = {"n_estimators": [100, 200], "max_depth": [12, 18, 24, None],
                "min_samples_leaf": [1, 5, 10, 20], "max_features": [0.33, 0.5, 0.8]}
    rs = RandomizedSearchCV(RandomForestRegressor(max_samples=0.5, n_jobs=NJOBS, random_state=SEED),
                            rf_space, n_iter=6, cv=tscv, scoring="neg_mean_absolute_error",
                            random_state=SEED, n_jobs=1, verbose=0)
    rs.fit(samp[FEATURES], samp[TARGET])
    rf_best = rs.best_params_
    pd.DataFrame(rs.cv_results_)[["params", "mean_test_score", "std_test_score"]] \
        .assign(MAE_cv=lambda d: -d.mean_test_score).sort_values("MAE_cv") \
        .to_csv(REPORTS / "tuning_rf.csv", index=False)
    print("  best", rf_best)
    rf = RandomForestRegressor(**rf_best, max_samples=0.25, n_jobs=NJOBS, random_state=SEED).fit(Xtr, ytr)
    preds["Random Forest"] = rf.predict(Xte)
    rows.append({"model": "Random Forest", "type": "ML", "tuned": "after (TimeSeriesSplit search)",
                 **metrics(yte, preds["Random Forest"])})
    joblib.dump({"model": rf, "features": FEATURES}, MODELS / "random_forest.joblib", compress=3)

    # ---------- XGBoost: default vs tuned ----------
    print("XGBoost (default) ...")
    xg0 = XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=NJOBS).fit(Xtr, ytr)
    rows.append({"model": "XGBoost", "type": "ML", "tuned": "before (default)",
                 **metrics(yte, xg0.predict(Xte))})
    print("XGBoost (search) ...")
    xg_space = {"n_estimators": [300, 600, 900], "learning_rate": [0.03, 0.05, 0.1],
                "max_depth": [6, 8, 10], "subsample": [0.7, 0.9], "colsample_bytree": [0.7, 0.9],
                "min_child_weight": [1, 5, 10], "objective": ["reg:squarederror", "reg:absoluteerror"]}
    rs = RandomizedSearchCV(XGBRegressor(tree_method="hist", random_state=SEED, n_jobs=NJOBS),
                            xg_space, n_iter=12, cv=tscv, scoring="neg_mean_absolute_error",
                            random_state=SEED, n_jobs=1)
    rs.fit(samp[FEATURES], samp[TARGET])
    xg_best = rs.best_params_
    pd.DataFrame(rs.cv_results_)[["params", "mean_test_score", "std_test_score"]] \
        .assign(MAE_cv=lambda d: -d.mean_test_score).sort_values("MAE_cv") \
        .to_csv(REPORTS / "tuning_xgb.csv", index=False)
    print("  best", xg_best)
    xg = XGBRegressor(**xg_best, tree_method="hist", random_state=SEED, n_jobs=NJOBS).fit(Xtr, ytr)
    preds["XGBoost"] = xg.predict(Xte)
    rows.append({"model": "XGBoost", "type": "ML", "tuned": "after (TimeSeriesSplit search)",
                 **metrics(yte, preds["XGBoost"])})
    joblib.dump({"model": xg, "features": FEATURES}, MODELS / "xgboost.joblib")
    # portable copy for the app: XGBoost 3 writes base_score as "[x]", which XGBoost 2.x cannot read
    xg.save_model(MODELS / "xgboost_model.json")
    js = json.load(open(MODELS / "xgboost_model.json"))
    lp = js["learner"]["learner_model_param"]; lp["base_score"] = str(lp["base_score"]).strip("[]")
    json.dump(js, open(MODELS / "xgboost_model.json", "w"))

    # round-1 reference: XGBoost with round-1 features and round-1 tuned parameters, same test set
    print("XGBoost round-1 features (reference) ...")
    xg1 = XGBRegressor(**ROUND1_XGB, tree_method="hist", random_state=SEED, n_jobs=NJOBS) \
        .fit(tr[FEATURES_R1], ytr)
    preds["XGBoost (round 1 features)"] = xg1.predict(te[FEATURES_R1])
    rows.append({"model": "XGBoost (round 1 features)", "type": "ML (round 1)",
                 "tuned": "round-1 tuned params", **metrics(yte, preds["XGBoost (round 1 features)"])})

    # ---------- results table ----------
    res = pd.DataFrame(rows)
    best_base = res[res.type == "baseline"].MAE_s.min()
    res["MAE_improvement_vs_best_baseline_%"] = (best_base - res.MAE_s) / best_base * 100
    # MAE on the raw (uncapped) test target too, for transparency
    raw = {k: mean_absolute_error(te.run_time_raw, v) for k, v in preds.items()}
    res["MAE_raw_uncapped_s"] = [raw.get(m) if t in ("-", "no tuning needed", "round-1 tuned params")
                                 or t.startswith("after") else np.nan for m, t in zip(res.model, res.tuned)]
    res = res.round(3)
    res.to_csv(REPORTS / "results_regression.csv", index=False)
    print(res.to_string(index=False))

    final = res[(res.type == "baseline") | (res.tuned.isin(["no tuning needed"])) |
                (res.tuned.str.startswith("after"))].copy()
    final.to_csv(REPORTS / "results_final_table.csv", index=False)
    best_ml = final[final.type == "ML"].sort_values("MAE_s").iloc[0].model
    print("best ML model:", best_ml)

    # ---------- journey-level (8 stops): ALL features frozen at the journey start ----------
    # predicted journey = sum of predicted segment run times + median dwell at intermediate stops
    js = pd.read_parquet(DATA / "journey_segments.parquet")
    js = js[~js.is_train].copy()
    lookup = joblib.load(DATA / "lookup.joblib")
    jp = {"Baseline: mean per segment": js.seg_mean.values,
          "Baseline: mean per segment x hour": js.seg_hour_mean.values,
          "Linear Regression": lr.predict(js[LR_FEATURES]),
          "Random Forest": rf.predict(js[FEATURES]),
          "XGBoost": xg.predict(js[FEATURES]),
          "XGBoost (round 1 features)": xg1.predict(js[FEATURES_R1])}
    js["dw_med"] = js.end_point.map(lookup["dwell_median"]).fillna(0)
    for k, v in jp.items():
        js[f"p_{k}"] = v
    g = js.groupby(["trip_id", "jid"])
    J = g.agg(actual_run=("run_time", "sum"), dwell=("dwell", "sum"), dw_med=("dw_med", "sum"),
              last_dwell=("dwell", "last"), last_dwmed=("dw_med", "last"),
              **{f"p_{k}": (f"p_{k}", "sum") for k in jp})
    J["actual"] = J.actual_run + J.dwell - J.last_dwell
    jrows = [{"model": k, **metrics(J.actual, J[f"p_{k}"] + J.dw_med - J.last_dwmed)} for k in jp]
    jres = pd.DataFrame(jrows)
    jbest = jres[jres.model.str.startswith("Baseline")].MAE_s.min()
    jres["MAE_improvement_vs_best_baseline_%"] = (jbest - jres.MAE_s) / jbest * 100
    jres["n_journeys"] = len(J)
    jres = jres.round(3); jres.to_csv(REPORTS / "results_journey_8stops.csv", index=False)
    print("\nJourney (8 stops, features as of journey start):\n", jres.to_string(index=False))

    # ---------- error analysis ----------
    best_b = "Baseline: mean per segment x hour"
    ea = te.assign(err_base=np.abs(yte - preds[best_b]), err_ml=np.abs(yte - preds[best_ml]))
    by_hour = ea.groupby("hour")[["err_base", "err_ml"]].mean().rename(
        columns={"err_base": f"MAE {best_b}", "err_ml": f"MAE {best_ml}"})
    by_hour["n"] = ea.groupby("hour").size()
    by_hour = by_hour[by_hour.n >= 500]
    by_route = ea.groupby("route")[["err_base", "err_ml"]].mean().rename(
        columns={"err_base": f"MAE {best_b}", "err_ml": f"MAE {best_ml}"})
    by_route["n"] = ea.groupby("route").size()
    by_rain = ea.groupby("is_rain")[["err_base", "err_ml"]].mean()
    by_hour.round(2).to_csv(REPORTS / "error_by_hour.csv")
    by_route.round(2).to_csv(REPORTS / "error_by_route.csv")
    by_rain.round(2).to_csv(REPORTS / "error_by_rain.csv")

    # ---------- feature importance ----------
    fi = pd.DataFrame({"feature": FEATURES, "random_forest": rf.feature_importances_,
                       "xgboost": xg.feature_importances_}).sort_values("xgboost", ascending=False)
    fi.round(4).to_csv(REPORTS / "feature_importance.csv", index=False)

    # ---------- classification: On time / Late for an 8-stop journey ----------
    j = pd.read_parquet(DATA / "journeys.parquet")
    j["base_hour_ratio"] = j.base_hour_sum / j.base_sum
    CF = ["route_code", "direction", "hour", "dow", "is_weekend", "is_holiday", "usual_time",
          "base_hour_ratio", "lag_ratio", "lag_missing", "headway_min", "rain_mm", "temp_c",
          "route_traffic", "traffic_missing", "bus_delay", "bus_delay_missing"]
    jtr, jte = j[j.is_train], j[~j.is_train]
    clf_rows, cms = [], {}

    def add(name, yhat):
        clf_rows.append({"model": name, "accuracy": accuracy_score(jte.late, yhat),
                         "F1_late": f1_score(jte.late, yhat, zero_division=0),
                         "recall_late": recall_score(jte.late, yhat, zero_division=0),
                         "precision_late": precision_score(jte.late, yhat, zero_division=0)})
        cms[name] = confusion_matrix(jte.late, yhat, labels=[0, 1])

    # non-AI: plan with the historical average -> the average is always within the budget
    add("Baseline: historical average (always On time)", np.zeros(len(jte), int))
    logit = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    logit.fit(jtr[CF], jtr.late); add("Logistic Regression (class weights)", logit.predict(jte[CF]))
    rfc = RandomForestClassifier(n_estimators=300, min_samples_leaf=10, class_weight="balanced",
                                 n_jobs=NJOBS, random_state=SEED).fit(jtr[CF], jtr.late)
    add("Random Forest Classifier (class weights)", rfc.predict(jte[CF]))
    clf = pd.DataFrame(clf_rows).round(3)
    clf.to_csv(REPORTS / "results_classification.csv", index=False)
    print("\nClassification (Late = journey > 1.2 x usual time):\n", clf.to_string(index=False))
    joblib.dump({"model": logit, "features": CF}, MODELS / "logistic_regression_late.joblib")
    joblib.dump({"model": rfc, "features": CF}, MODELS / "random_forest_late.joblib", compress=3)

    # =================== charts ===================
    # 1. model vs baseline MAE
    f = final.sort_values("MAE_s", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [GRAY if t == "baseline" else (ORANGE if m == best_ml else NAVY)
              for m, t in zip(f.model, f.type)]
    bars = ax.barh(f.model, f.MAE_s, color=colors, height=0.6)
    label_bars(ax, bars, "{:.1f} s")
    ax.set_xlabel("Mean absolute error per segment (seconds, lower is better)")
    ax.set_title("Test set (9-21 Sep 2024): ML models vs historical-average baselines", loc="left")
    ax.grid(axis="y", visible=False); ax.set_xlim(0, f.MAE_s.max() * 1.18)
    fig.tight_layout(); fig.savefig(REPORTS / "mae_model_vs_baseline.png", dpi=160); plt.close(fig)

    # 2. before vs after tuning
    tt = res[res.model.isin(["Random Forest", "XGBoost"])]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(2); w = 0.36
    for i, (lab, col) in enumerate([("before (default)", GRAY), ("after", NAVY)]):
        v = [tt[(tt.model == m) & tt.tuned.str.startswith(lab.split()[0])].MAE_s.iloc[0]
             for m in ["Random Forest", "XGBoost"]]
        b = ax.bar(x + (i - .5) * w, v, w * 0.92, color=col, label="Before tuning" if i == 0 else "After tuning")
        for bb in b:
            ax.annotate(f"{bb.get_height():.2f}", (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, ["Random Forest", "XGBoost"]); ax.set_ylabel("Test MAE (s)")
    ax.set_title("Hyper-parameter tuning (TimeSeriesSplit, 3 folds)", loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=2); ax.grid(axis="x", visible=False)
    ax.set_ylim(0, tt.MAE_s.max() * 1.35)
    fig.tight_layout(); fig.savefig(REPORTS / "tuning_before_after.png", dpi=160); plt.close(fig)

    # 3. feature importance (XGBoost)
    top = fi.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.barh(top.feature, top.xgboost, color=NAVY, height=0.6)
    label_bars(ax, bars, "{:.3f}")
    ax.set_xlabel("Importance (XGBoost gain share)"); ax.grid(axis="y", visible=False)
    ax.set_title("What drives the prediction? Top 12 features (XGBoost)", loc="left")
    ax.set_xlim(0, top.xgboost.max() * 1.18)
    fig.tight_layout(); fig.savefig(REPORTS / "feature_importance.png", dpi=160); plt.close(fig)

    # 4. error by hour
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(by_hour.index, by_hour.iloc[:, 0], color=GRAY, lw=2, marker="o", ms=5, label=best_b)
    ax.plot(by_hour.index, by_hour.iloc[:, 1], color=ORANGE, lw=2, marker="o", ms=5, label=best_ml)
    ax.set_xlabel("Hour of day"); ax.set_ylabel("Test MAE (s)"); ax.set_xticks(by_hour.index)
    ax.set_title("Error by hour of day", loc="left"); ax.legend(frameon=False)
    ax.set_ylim(0, by_hour.iloc[:, :2].values.max() * 1.15)
    fig.tight_layout(); fig.savefig(REPORTS / "error_by_hour.png", dpi=160); plt.close(fig)

    # 5. error by route
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    x = np.arange(len(by_route)); w = 0.36
    for i, (c, col, lab) in enumerate([(0, GRAY, best_b), (1, ORANGE, best_ml)]):
        b = ax.bar(x + (i - .5) * w, by_route.iloc[:, c], w * 0.92, color=col, label=lab)
        for bb in b:
            ax.annotate(f"{bb.get_height():.1f}", (bb.get_x() + bb.get_width() / 2, bb.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=10, color=INK)
    ax.set_xticks(x, [f"Route {r}" for r in by_route.index]); ax.set_ylabel("Test MAE (s)")
    ax.set_title("Error by route", loc="left"); ax.legend(frameon=False, fontsize=9)
    ax.grid(axis="x", visible=False); ax.set_ylim(0, by_route.iloc[:, :2].values.max() * 1.25)
    fig.tight_layout(); fig.savefig(REPORTS / "error_by_route.png", dpi=160); plt.close(fig)

    # 6. confusion matrices
    names = [n for n in cms if not n.startswith("Baseline")]
    fig, axes = plt.subplots(1, len(names), figsize=(9, 3.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("navy", [CREAM, NAVY])
    for ax, n in zip(axes, names):
        cm = cms[n]; ax.imshow(cm, cmap=cmap); ax.grid(False)
        for (r, c), v in np.ndenumerate(cm):
            ax.text(c, r, f"{v:,}", ha="center", va="center", fontsize=12,
                    color="white" if v > cm.max() / 2 else INK)
        ax.set_xticks([0, 1], ["On time", "Late"]); ax.set_yticks([0, 1], ["On time", "Late"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(n.split(" (")[0], fontsize=11)
    fig.suptitle("On time / Late (8-stop journey), test set", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(); fig.savefig(REPORTS / "confusion_matrices.png", dpi=160); plt.close(fig)

    # 7. EDA: travel time by hour (train)
    eda = tr.assign(ratio=tr.run_time / tr.seg_mean).groupby("hour").ratio.agg(["mean", "size"])
    eda = eda[eda["size"] >= 500]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.bar(eda.index, (eda["mean"] - 1) * 100, color=[ORANGE if v > 1 else NAVY for v in eda["mean"]],
           width=0.7)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(eda.index); ax.set_xlabel("Hour of day")
    ax.set_ylabel("% slower (+) / faster (-)\nthan segment average")
    ax.set_title("Same segment, different hour: travel time varies with time of day (train data)", loc="left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(REPORTS / "eda_hour_profile.png", dpi=160); plt.close(fig)

    summary = {"best_ml_model": best_ml, "best_baseline": final[final.type == "baseline"]
               .sort_values("MAE_s").iloc[0].model, "best_baseline_MAE_s": float(best_base),
               "best_ml_MAE_s": float(final[final.model == best_ml].MAE_s.iloc[0]),
               "improvement_%": float(final[final.model == best_ml]
                                      ["MAE_improvement_vs_best_baseline_%"].iloc[0]),
               "target_15pct_met": bool(final[final.model == best_ml]
                                        ["MAE_improvement_vs_best_baseline_%"].iloc[0] >= 15),
               "rf_best_params": {k: (None if v is None else v) for k, v in rf_best.items()},
               "xgb_best_params": xg_best,
               "xgb_round1_features_MAE_s": float(res[res.model == "XGBoost (round 1 features)"].MAE_s.iloc[0]),
               "journey_best_baseline_MAE_s": float(jbest),
               "journey_xgb_MAE_s": float(jres[jres.model == "XGBoost"].MAE_s.iloc[0]),
               "journey_rf_MAE_s": float(jres[jres.model == "Random Forest"].MAE_s.iloc[0]), "train_rows": len(tr), "test_rows": len(te),
               "runtime_min": round((time.time() - t0) / 60, 1)}
    json.dump(summary, open(REPORTS / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
