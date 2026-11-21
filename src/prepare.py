"""
prepare.py - clean the Astana bus GPS segment data and build features.

Input : data/segment_level_data.csv      (Zenodo 10.5281/zenodo.15769359)
        data/weather_astana_hourly.csv    (Open-Meteo archive, 51.17N 71.45E, local time UTC+5)
Output: data/features.parquet             (one row per bus x stop-to-stop segment)
        data/journeys.parquet             (multi-stop journeys for the On time / Late task)
        data/lookup.joblib                (stop sequences, medians, etc. used by the app)
        reports/cleaning_log.csv          (rows removed and why)

Notes on the real file (verified before writing this script):
  * There is NO route_id column. Routes are inferred: trips are clustered by the set of stops
    they visit (KMeans, k=3 per direction). Each direction-1 cluster pairs with exactly one
    direction-2 cluster through shared buses, giving 3 routes, labelled A, B, C.
  * `segment` is the position of the segment inside a trip, not a fixed stop pair, so a physical
    segment is identified by (start_point, end_point).
  * run_time = arrival_time - start_time; dwell_time = departure_time - arrival_time (checked: 100%).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import holidays
import joblib
from sklearn.cluster import KMeans
from features import route_traffic

ROOT = Path(__file__).resolve().parents[1]
DATA, REPORTS = ROOT / "data", ROOT / "reports"
REPORTS.mkdir(exist_ok=True)
SEED = 42
SPLIT_DATE = pd.Timestamp("2024-09-09")   # train: 29 Jul - 8 Sep (6 weeks); test: 9 - 21 Sep
MAX_RUN = 3600            # a single stop-to-stop run longer than 1 hour is a GPS/logging error
CAP_PCT = (0.01, 0.99)    # per-segment winsorising, percentiles computed on TRAIN only
MIN_SEG_ROWS = 50         # segments seen fewer times in train are rare detours
JOURNEY_LEN = 8           # stops per journey for the classification task
LATE_FACTOR = 1.2         # Late = journey takes > 120% of the usual (median) time


def log_step(log, name, before, after, why):
    log.append({"step": name, "rows_before": before, "rows_after": after,
                "rows_removed": before - after, "reason": why})
    print(f"  {name:<28} removed {before - after:>7,}  -> {after:,}  ({why})")


def infer_routes(df):
    """Cluster trips by visited stops (per direction) and pair directions via shared buses."""
    labels = {}
    for d in (1, 2):
        x = df[df.direction == d]
        M = pd.crosstab(x.trip_id, x.start_point).clip(upper=1)
        km = KMeans(3, n_init=5, random_state=0).fit(M.values)
        labels.update({t: f"{d}-{l}" for t, l in zip(M.index, km.labels_)})
    tr = df[["trip_id", "deviceid", "date"]].drop_duplicates("trip_id").copy()
    tr["cl"] = tr.trip_id.map(labels)
    pairs = {}
    for _, g in tr.groupby(["deviceid", "date"]):
        s = set(g.cl)
        for a in [c for c in s if c[0] == "1"]:
            for b in [c for c in s if c[0] == "2"]:
                pairs[(a, b)] = pairs.get((a, b), 0) + 1
    # greedy one-to-one matching by count
    match, used_a, used_b = {}, set(), set()
    for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1]):
        if a not in used_a and b not in used_b:
            match[a] = b; used_a.add(a); used_b.add(b)
    # name routes A, B, C by number of trips (largest first)
    sizes = {a: (tr.cl == a).sum() + (tr.cl == b).sum() for a, b in match.items()}
    name = {}
    for i, a in enumerate(sorted(sizes, key=lambda k: -sizes[k])):
        name[a] = name[match[a]] = "ABC"[i]
    print("  route pairing (dir1, dir2, shared bus-days):",
          [(a, match[a], pairs[(a, match[a])]) for a in match])
    return {t: name[c] for t, c in labels.items()}


def main():
    log = []
    print("Loading raw data ...")
    df = pd.read_csv(DATA / "segment_level_data.csv")
    print("  shape", df.shape)
    for c in ("start_time", "arrival_time", "departure_time"):
        df[c] = pd.to_datetime(df[c])
    df["date"] = pd.to_datetime(df["date"])

    print("Cleaning ...")
    n = len(df); df = df.drop_duplicates(subset=["trip_id", "start_point", "end_point", "start_time"])
    log_step(log, "duplicates", n, len(df), "same trip, stop pair and start time")
    n = len(df); df = df[df.run_time_in_seconds > 0]
    log_step(log, "run_time <= 0", n, len(df), "impossible travel time")
    n = len(df); df = df[df.run_time_in_seconds <= MAX_RUN]
    log_step(log, f"run_time > {MAX_RUN}s", n, len(df), "multi-hour gap = GPS/logging error")
    n = len(df); df = df[df.start_point != df.end_point]
    log_step(log, "start == end stop", n, len(df), "not a real segment")

    print("Inferring routes ...")
    df["route"] = df.trip_id.map(infer_routes(df))

    df["is_train"] = df.start_time < SPLIT_DATE
    seg_counts = df[df.is_train].groupby(["start_point", "end_point"]).size()
    keep = seg_counts[seg_counts >= MIN_SEG_ROWS].index
    n = len(df)
    df = df.set_index(["start_point", "end_point"]).loc[lambda x: x.index.isin(keep)].reset_index()
    log_step(log, f"rare segments (<{MIN_SEG_ROWS} in train)", n, len(df), "detours / rare stop pairs")

    # per-segment cap, percentiles from TRAIN only (applied to train and test)
    q = (df[df.is_train].groupby(["start_point", "end_point"]).run_time_in_seconds
         .quantile(list(CAP_PCT)).unstack())
    q.columns = ["lo", "hi"]
    df = df.merge(q, left_on=["start_point", "end_point"], right_index=True, how="left")
    capped = ((df.run_time_in_seconds < df.lo) | (df.run_time_in_seconds > df.hi)).sum()
    df["run_time_raw"] = df.run_time_in_seconds
    df["run_time"] = df.run_time_in_seconds.clip(df.lo, df.hi)
    log.append({"step": "cap per segment (1st-99th pct, train)", "rows_before": len(df),
                "rows_after": len(df), "rows_removed": 0,
                "reason": f"{capped:,} values capped, not removed"})
    print(f"  capped {capped:,} extreme values per segment (kept rows)")
    df["dwell"] = df.dwell_time_in_seconds.clip(upper=600)

    # ---------------- features ----------------
    print("Building features ...")
    df["seg_key"] = df.start_point.astype(str) + "-" + df.end_point.astype(str)
    df["seg_id"] = df.seg_key.astype("category").cat.codes
    df["hour"] = df.start_time.dt.hour
    df["minute_of_day"] = df.hour * 60 + df.start_time.dt.minute
    df["dow"] = df.start_time.dt.dayofweek
    df["is_weekend"] = (df.dow >= 5).astype(int)
    kz = holidays.country_holidays("KZ", years=2024)
    df["is_holiday"] = df.date.dt.date.map(lambda d: int(d in kz))
    df["route_code"] = df.route.map({"A": 0, "B": 1, "C": 2})
    df["position"] = df.segment          # position of the segment along the trip

    # historical averages (TRAIN only) = the commuter's "usual time"; also used as features
    tr = df[df.is_train]
    seg_mean = tr.groupby("seg_key").run_time.mean().rename("seg_mean")
    seg_hour_mean = tr.groupby(["seg_key", "hour"]).run_time.mean().rename("seg_hour_mean")
    df = df.join(seg_mean, on="seg_key").join(seg_hour_mean, on=["seg_key", "hour"])
    df["seg_hour_mean"] = df.seg_hour_mean.fillna(df.seg_mean)

    # LAG = run time of the most recent EARLIER bus on the same segment that had already
    # arrived before this bus started the segment (= current traffic, no future data)
    df = df.sort_values("start_time").reset_index(drop=True)
    prev = (df[["seg_key", "arrival_time", "run_time", "dwell"]]
            .rename(columns={"arrival_time": "prev_arrival", "run_time": "lag_run",
                             "dwell": "lag_dwell"})
            .sort_values("prev_arrival"))
    df = pd.merge_asof(df.sort_values("start_time"), prev, left_on="start_time",
                       right_on="prev_arrival", by="seg_key", allow_exact_matches=False,
                       direction="backward")
    df["lag_age_min"] = (df.start_time - df.prev_arrival).dt.total_seconds() / 60
    stale = df.lag_run.isna() | (df.lag_age_min > 60)          # older than 1 h = not "current"
    df["lag_missing"] = stale.astype(int)
    df.loc[stale, "lag_run"] = df.loc[stale, "seg_hour_mean"]
    df.loc[stale, "lag_dwell"] = np.nan
    df["lag_age_min"] = df.lag_age_min.where(~stale, 60).clip(upper=60)
    df["lag_ratio"] = df.lag_run / df.seg_mean                  # >1 = traffic slower than usual

    # HEADWAY at the stop = gap since the previous bus of the same route left this stop
    df = df.sort_values(["route", "direction", "start_point", "start_time"])
    gap = df.groupby(["route", "direction", "start_point"]).start_time.diff().dt.total_seconds() / 60
    df["headway_min"] = gap.clip(upper=60).fillna(60)

    # WEATHER (Open-Meteo hourly, local time)
    w = pd.read_csv(DATA / "weather_astana_hourly.csv", parse_dates=["time"])
    df["wtime"] = df.start_time.dt.floor("h")
    df = df.merge(w.rename(columns={"time": "wtime", "precipitation": "rain_mm",
                                    "temperature_2m": "temp_c"}), on="wtime", how="left")
    df["is_rain"] = (df.rain_mm > 0).astype(int)
    df["lag_dwell"] = df.lag_dwell.fillna(df.groupby("start_point").dwell.transform("median"))

    # ROUND 2 - real-time traffic features (only data observed BEFORE this bus starts the segment)
    # (a) route_traffic: average slowness (run / segment average) of all buses on the same route and
    #     direction that finished a segment in the last 15 minutes
    df["ratio"] = df.run_time / df.seg_mean
    rt, rm = route_traffic(df[["route", "direction", "arrival_time", "ratio"]],
                           df[["route", "direction"]].assign(qtime=df.start_time))
    df["route_traffic"] = np.clip(rt, 0.3, 3.0); df["traffic_missing"] = rm
    # (b) bus_delay: how slow THIS bus has been so far on this trip (earlier segments it already ran)
    df = df.sort_values(["trip_id", "position"])
    g = df.groupby("trip_id")
    run_before = g.run_time.cumsum() - df.run_time
    mean_before = g.seg_mean.cumsum() - df.seg_mean
    df["bus_delay_missing"] = (mean_before <= 0).astype(int)
    df["bus_delay"] = np.clip(np.where(mean_before > 0, run_before / mean_before.where(mean_before > 0, 1), 1.0), 0.3, 3.0)

    df = df.sort_values("start_time").reset_index(drop=True)
    cols = ["trip_id", "deviceid", "route", "route_code", "direction", "segment", "position",
            "start_point", "end_point", "seg_key", "seg_id", "start_time", "arrival_time",
            "date", "is_train", "run_time", "run_time_raw", "dwell", "hour", "minute_of_day",
            "dow", "is_weekend", "is_holiday", "seg_mean", "seg_hour_mean", "lag_run",
            "lag_dwell", "lag_age_min", "lag_missing", "lag_ratio", "headway_min", "rain_mm",
            "is_rain", "temp_c", "ratio", "route_traffic", "traffic_missing", "bus_delay",
            "bus_delay_missing"]
    df = df[cols]
    df.to_parquet(DATA / "features.parquet", index=False)
    print(f"  features: {df.shape}, train {df.is_train.sum():,}, test {(~df.is_train).sum():,}")
    print("  null counts:", df.isna().sum()[lambda s: s > 0].to_dict())

    # ---------------- journeys: everything frozen at the moment the commuter is at the origin ----------------
    print("Building journeys (features as of journey start) ...")
    df = df.sort_values(["trip_id", "position"]).reset_index(drop=True)
    df["k"] = df.groupby("trip_id").cumcount()
    df["jid"] = df.k // JOURNEY_LEN
    df["n_in_j"] = df.groupby(["trip_id", "jid"]).run_time.transform("size")
    js = df[df.n_in_j == JOURNEY_LEN].copy()
    first = js.groupby(["trip_id", "jid"]).transform("first")
    js["t0"] = first.start_time
    # time-of-day features use the journey start time (what is known at the stop)
    js["hour"] = js.t0.dt.hour
    js["minute_of_day"] = js.hour * 60 + js.t0.dt.minute
    for c in ["headway_min", "rain_mm", "is_rain", "temp_c", "route_traffic", "traffic_missing",
              "bus_delay", "bus_delay_missing"]:
        js[c] = first[c]                                   # values observed at the origin stop
    js = js.drop(columns=["seg_hour_mean"]).join(seg_hour_mean, on=["seg_key", "hour"])
    js["seg_hour_mean"] = js.seg_hour_mean.fillna(js.seg_mean)
    # previous-bus lag for EVERY segment of the journey, as of journey start time t0
    js = js.drop(columns=["lag_run", "lag_dwell", "lag_age_min", "lag_missing", "lag_ratio"])
    js = pd.merge_asof(js.sort_values("t0"), prev, left_on="t0", right_on="prev_arrival",
                       by="seg_key", allow_exact_matches=False, direction="backward")
    js["lag_age_min"] = (js.t0 - js.prev_arrival).dt.total_seconds() / 60
    stale = js.lag_run.isna() | (js.lag_age_min > 60)
    js["lag_missing"] = stale.astype(int)
    js.loc[stale, "lag_run"] = js.loc[stale, "seg_hour_mean"]
    js["lag_age_min"] = js.lag_age_min.where(~stale, 60).clip(upper=60)
    js["lag_ratio"] = js.lag_run / js.seg_mean
    js["lag_dwell"] = js.lag_dwell.where(~stale).fillna(js.end_point.map(
        df[df.is_train].groupby("end_point").dwell.median())).fillna(0)
    js = js.sort_values(["trip_id", "position"]).reset_index(drop=True)
    js.drop(columns=["prev_arrival", "k", "n_in_j"]).to_parquet(DATA / "journey_segments.parquet", index=False)

    g = js.groupby(["trip_id", "jid"])
    j = g.agg(route=("route", "first"), route_code=("route_code", "first"),
              direction=("direction", "first"), origin=("start_point", "first"),
              dest=("end_point", "last"), start_time=("t0", "first"), is_train=("is_train", "first"),
              hour=("hour", "first"), dow=("dow", "first"), is_weekend=("is_weekend", "first"),
              is_holiday=("is_holiday", "first"), run_sum=("run_time", "sum"),
              dwell_sum=("dwell", "sum"), base_sum=("seg_mean", "sum"),
              base_hour_sum=("seg_hour_mean", "sum"), lag_sum=("lag_run", "sum"),
              lag_missing=("lag_missing", "mean"), headway_min=("headway_min", "first"),
              rain_mm=("rain_mm", "first"), temp_c=("temp_c", "first"),
              route_traffic=("route_traffic", "first"), traffic_missing=("traffic_missing", "first"),
              bus_delay=("bus_delay", "first"), bus_delay_missing=("bus_delay_missing", "first"),
              last_dwell=("dwell", "last")).reset_index()
    j["lag_ratio"] = j.lag_sum / j.base_sum
    j["travel_time"] = j.run_sum + j.dwell_sum - j.last_dwell   # dwell at intermediate stops only
    j["od"] = j.origin.astype(str) + ">" + j.dest.astype(str)
    med = j[j.is_train].groupby("od").travel_time.median().rename("usual_time")
    j = j.join(med, on="od").dropna(subset=["usual_time"])
    j["deadline_s"] = LATE_FACTOR * j.usual_time
    j["late"] = (j.travel_time > j.deadline_s).astype(int)
    j.to_parquet(DATA / "journeys.parquet", index=False)
    print(f"  journeys: {len(j):,}, Late rate train {j[j.is_train].late.mean():.1%}, "
          f"test {j[~j.is_train].late.mean():.1%}")

    # ---------------- lookup tables for the app ----------------
    tr = df[df.is_train]
    seqs = {}
    for (r, d), x in tr.groupby(["route", "direction"]):
        top = x.groupby("trip_id").start_point.apply(tuple).value_counts().index[0]
        trip = x[x.trip_id.isin(x.groupby("trip_id").start_point.apply(tuple)
                                [lambda s: s == top].index[:1])].sort_values("position")
        seqs[(r, d)] = list(zip(trip.start_point, trip.end_point))
    lookup = {
        "sequences": seqs,
        "seg_mean": seg_mean.to_dict(),
        "seg_hour_mean": seg_hour_mean.to_dict(),
        "dwell_median": tr.groupby("start_point").dwell.median().to_dict(),
        "lag_dwell_median": tr.groupby("seg_key").dwell.median().to_dict(),
        "headway_median": tr.groupby(["route", "direction", "start_point", "hour"])
                            .headway_min.median().to_dict(),
        "headway_route_hour": tr.groupby(["route", "hour"]).headway_min.median().to_dict(),
        "seg_id": dict(zip(df.seg_key, df.seg_id)),
        "slow_ratio": float(tr.lag_ratio.quantile(0.95)),      # "Slow" = top 5% of observed
        "slow_traffic": float(tr.route_traffic.quantile(0.95)),
        "normal_traffic": float(tr.route_traffic.median()),
        "temp_median": float(tr.temp_c.median()),
        "position": tr.groupby(["route", "direction", "seg_key"]).position.median().to_dict(),
    }
    joblib.dump(lookup, DATA / "lookup.joblib")
    pd.DataFrame(log).to_csv(REPORTS / "cleaning_log.csv", index=False)
    print("Done.")


if __name__ == "__main__":
    main()
