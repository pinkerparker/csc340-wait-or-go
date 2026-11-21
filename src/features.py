"""Shared feature definitions and real-time traffic helpers (used by prepare, train and the app)."""
import numpy as np
import pandas as pd

TRAFFIC_WINDOW_MIN = 15

# Round 1 feature set
FEATURES_R1 = ["route_code", "direction", "seg_id", "position", "hour", "minute_of_day", "dow",
               "is_weekend", "is_holiday", "seg_mean", "seg_hour_mean", "lag_run", "lag_dwell",
               "lag_age_min", "lag_missing", "lag_ratio", "headway_min", "rain_mm", "is_rain",
               "temp_c"]
# Round 2 adds two real-time traffic features (both use only data observed before the query time)
FEATURES = FEATURES_R1 + ["route_traffic", "traffic_missing", "bus_delay", "bus_delay_missing"]


def route_traffic(hist, query):
    """Mean slowness ratio (run_time / segment average) of ALL buses on the same route+direction
    whose segment ended in the last 15 minutes before each query time.

    hist : DataFrame with route, direction, arrival_time, ratio
    query: DataFrame with route, direction, qtime
    returns (value, missing_flag) arrays aligned with query.index
    """
    val = np.ones(len(query)); miss = np.ones(len(query), dtype=int)
    qpos = {k: np.where((query.route.values == k[0]) & (query.direction.values == k[1]))[0]
            for k in query.groupby(["route", "direction"]).groups}
    for k, idx in qpos.items():
        h = hist[(hist.route == k[0]) & (hist.direction == k[1])].sort_values("arrival_time")
        if h.empty:
            continue
        a = h.arrival_time.values.astype("datetime64[ns]").astype(np.int64)
        c = np.concatenate([[0.0], np.cumsum(h.ratio.values)])
        q = query.qtime.values[idx].astype("datetime64[ns]").astype(np.int64)
        hi = np.searchsorted(a, q, side="left")                     # strictly before query time
        lo = np.searchsorted(a, q - TRAFFIC_WINDOW_MIN * 60 * 10**9, side="left")
        n = hi - lo
        ok = n > 0
        val[idx[ok]] = (c[hi[ok]] - c[lo[ok]]) / n[ok]
        miss[idx[ok]] = 0
    return val, miss
