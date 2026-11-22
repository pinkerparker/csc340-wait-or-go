"""
Wait or Go? - demo app for the CSC340 project (bus travel time prediction).
Run from the project folder:   streamlit run app/app.py   ->  http://localhost:8501

The travel-time prediction is the trained XGBoost model (ML).
The advice (wait / consider motorcycle taxi / switch to Grab) is a simple RULE on top of
the prediction, NOT machine learning.
"""
from datetime import datetime, date, time, timedelta
from pathlib import Path
import sys
import joblib
import pandas as pd
import streamlit as st
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import FEATURES  # noqa: E402

TIGHT_MIN = 5      # arrive >= 5 min before the deadline -> safe to wait for the bus
LATE_TOL_MIN = 3   # up to 3 min late -> "risky"; more than that -> switch
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

st.set_page_config(page_title="Wait or Go?", page_icon="🚌", layout="wide")

# ---------------------------------------------------------------- look & feel
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=League+Spartan:wght@600;800;900&family=Jost:wght@400;500;600;700&display=swap');
:root { --cream:#F6F0DF; --paper:#FBF7EA; --olive:#D2D383; --mustard:#F1CB7F; --pink:#D594AA;
        --ink:#1E1E1E; --road:#3F403B; --green:#4A6B4B; --sky:#BFD0D6; }
html, body, .stApp, [class*="css"] { font-family:'Jost','Futura','Avenir Next','Helvetica Neue',Arial,sans-serif; color:var(--ink); }
.stApp { background:var(--cream); }
.block-container { padding:1.75rem max(4vw,16px) 3rem !important; max-width:1240px !important; }
header[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif,'Futura','Avenir Next','Arial Black',sans-serif !important; color:var(--ink); }

.kick { font-family:'Jost','Futura','Avenir Next',sans-serif; font-size:.95rem; letter-spacing:.12em; text-transform:uppercase; margin:0; }
.hero { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:clamp(48px,8vw,92px); line-height:.95;
        color:var(--olive); -webkit-text-stroke:2px var(--ink); text-transform:uppercase; margin:.2rem 0 .8rem; }
.band { background:var(--olive); border:2px solid var(--ink); padding:.45rem 1rem; display:inline-block; }
.band span { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:1.45rem; color:var(--pink);
             -webkit-text-stroke:1px var(--ink); text-transform:uppercase; letter-spacing:.01em; }
.road { height:46px; background:var(--road); border-top:8px solid #333; margin:1.2rem 0 1.4rem; position:relative; }
.road::after { content:""; position:absolute; left:0; right:0; top:17px; height:5px;
               background:repeating-linear-gradient(90deg, var(--cream) 0 60px, transparent 60px 110px); }

.card { background:var(--mustard); border:2px solid var(--ink); padding:1.1rem 1.3rem 1.2rem; }
.card h3 { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.35rem; margin:0 0 .25rem !important; padding:0 !important; line-height:1.1; }
.card p { margin:0 !important; }
.card { padding:1rem 1.25rem !important; }
.step { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.05rem; text-transform:uppercase;
        letter-spacing:.04em; margin:.9rem 0 .1rem; }
.hint { font-size:.92rem; color:#4b4b4b; margin:0 0 .2rem; }

.answer { border:2px solid var(--ink); padding:1.3rem 1.5rem; }
.answer.ok { background:var(--olive); } .answer.warn { background:var(--mustard); } .answer.bad { background:var(--pink); }
.answer .big { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:2.9rem; line-height:1; text-transform:uppercase;
               color:var(--paper); -webkit-text-stroke:1.6px var(--ink); margin:.2rem 0 .5rem; }
.answer .do { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.6rem; margin:0; }
.answer .why { font-size:1.05rem; margin:.5rem 0 0; }

.tl { display:flex; gap:0; margin-top:1rem; border:2px solid var(--ink); background:var(--paper); }
.tl > div { flex:1; padding:.8rem .9rem; border-right:2px solid var(--ink); }
.tl > div:last-child { border-right:none; }
.tl .t { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.9rem; line-height:1; }
.tl .l { font-size:.85rem; text-transform:uppercase; letter-spacing:.06em; margin-bottom:.3rem; }
.tl .d { font-size:.9rem; color:#4b4b4b; }
.tl .class { background:var(--sky); }

.stat { background:var(--paper); border:2px solid var(--ink); padding:1rem 1.2rem; }
.stat .n { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:2.6rem; color:var(--olive);
           -webkit-text-stroke:1.4px var(--ink); line-height:1; }
.stat .s { font-size:1rem; margin-top:.35rem; }
.note { font-size:.9rem; color:#4b4b4b; }

/* widgets: square, black outline, paper background */
div[data-baseweb="select"] > div, div[data-baseweb="input"], div[data-baseweb="base-input"] {
  border-radius:0 !important; border:2px solid var(--ink) !important; background:var(--paper) !important; }
[data-testid="stWidgetLabel"] p { font-weight:600; font-size:1rem; }
button[kind="segmented_control"], button[kind="segmented_controlActive"], [data-testid="stBaseButton-segmented_control"],
[data-testid="stBaseButton-segmented_controlActive"] { border-radius:0 !important; border:2px solid var(--ink) !important; }
[data-testid="stBaseButton-segmented_controlActive"] { background:var(--ink) !important; color:var(--paper) !important; }
.stTabs [data-baseweb="tab"] { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.1rem; text-transform:uppercase; }
details { border:2px solid var(--ink) !important; border-radius:0 !important; background:var(--paper); }
h2.sec { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.7rem; margin:1.6rem 0 .7rem; }
.bars { display:grid; gap:8px; }
.bar { display:grid; grid-template-columns:minmax(0,14em) minmax(0,1fr) 4.5em; gap:10px; align-items:center; font-size:.95rem; }
.bar .tr { height:26px; background:var(--paper); border:2px solid var(--ink); }
.bar .fill { height:100%; background:var(--ink); }
.bar .fill.base { background:#A9A9A0; } .bar .fill.hot { background:var(--pink); }
.bar .v { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.scroll { overflow-x:auto; }
table.res { border-collapse:collapse; width:100%; font-size:.95rem; background:var(--paper); }
table.res th, table.res td { border:2px solid var(--ink); padding:7px 10px; text-align:left; }
table.res th { background:var(--olive); font-weight:600; }
table.res .n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
table.res tr.hl td { background:#F3DDB0; font-weight:600; }
/* match the web version: Jost for all text, icons keep their own font */
.stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div, .stApp td, .stApp th,
.stApp input, .stApp button, .stApp summary { font-family:'Jost','Futura','Avenir Next','Helvetica Neue',Arial,sans-serif; }
.stApp [data-testid="stIconMaterial"], .stApp .material-symbols-rounded, .stApp [class*="material-symbols"] {
  font-family:'Material Symbols Rounded' !important; }
.stApp .hero, .stApp .band span, .stApp .big, .stApp .do, .stApp .step, .stApp .card h3, .stApp .tl .t,
.stApp .stat .n, .stApp h2.sec { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; }
.stTabs [data-baseweb="tab-list"] { gap:0; border-bottom:2px solid var(--ink); align-items:flex-end; margin-bottom:8px; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stTabs [data-baseweb="tab"] { padding:14px 18px 12px; height:auto; min-height:44px; background:transparent; border:0;
  margin:0; line-height:1; }
.stTabs [data-baseweb="tab"] p { font-family:'League Spartan','Futura','Avenir Next',sans-serif !important; font-weight:800;
  font-size:17px; line-height:1; margin:0; text-transform:uppercase; color:#4f4f4f; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:var(--olive); border:2px solid var(--ink); border-bottom:0;
  margin-bottom:-2px; }
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color:var(--ink); }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load():
    lk = joblib.load(ROOT / "data" / "lookup.joblib")
    # portable JSON model: same predictions on XGBoost 2.x and 3.x (a pickle is version-specific)
    m = XGBRegressor()
    m.load_model(str(ROOT / "models" / "xgboost_model.json"))
    return lk, m, FEATURES


lk, model, feats = load()
assert feats == FEATURES


def predict_trip(route, direction, i0, i1, t_arrive, dow, rain, traffic_slow):
    """Wait + ML travel time for the segments i0..i1-1 of the route's usual stop sequence."""
    seq = lk["sequences"][(route, direction)][i0:i1]
    hour = t_arrive.hour
    hw = lk["headway_median"].get((route, direction, seq[0][0], hour),
                                  lk["headway_route_hour"].get((route, hour), 15.0))
    wait_min = hw / 2                               # random arrival at the stop -> half a headway
    t_board = t_arrive + timedelta(minutes=wait_min)
    hb = t_board.hour
    rows = []
    for k, (a, b) in enumerate(seq):
        key = f"{a}-{b}"
        sm = lk["seg_mean"][key]
        shm = lk["seg_hour_mean"].get((key, hb), sm)
        lag_ratio = lk["slow_ratio"] if traffic_slow else 1.0
        rows.append({
            "route_code": {"A": 0, "B": 1, "C": 2}[route], "direction": direction,
            "seg_id": lk["seg_id"][key],
            "position": lk["position"].get((route, direction, key), i0 + k + 1),
            "hour": hb, "minute_of_day": hb * 60 + t_board.minute, "dow": dow,
            "is_weekend": int(dow >= 5), "is_holiday": 0, "seg_mean": sm, "seg_hour_mean": shm,
            "lag_run": sm * lag_ratio, "lag_dwell": lk["lag_dwell_median"].get(key, 20),
            "lag_age_min": 10, "lag_missing": 0, "lag_ratio": lag_ratio, "headway_min": hw,
            "rain_mm": 1.0 if rain else 0.0, "is_rain": int(rain), "temp_c": lk["temp_median"],
            "route_traffic": lk["slow_traffic"] if traffic_slow else lk["normal_traffic"],
            "traffic_missing": 0, "bus_delay": 1.0, "bus_delay_missing": 0})
    X = pd.DataFrame(rows)[FEATURES]
    run = model.predict(X)
    dwell = sum(lk["dwell_median"].get(b, 20) for _, b in seq[:-1])   # stops in between
    travel_s = float(run.sum() + dwell)
    base_s = float(X.seg_hour_mean.sum() + dwell)
    return wait_min, hw, t_board, travel_s, base_s, X.assign(pred_s=run)


# ---------------------------------------------------------------- header
st.markdown("""
<div class="hero">Wait or Go?</div>
<div class="band"><span>Predicting bus travel time with machine learning</span></div>
<div class="road"></div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Plan my trip", "How good is the AI?"])

with tab1:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown('<div class="card"><h3>Your trip</h3>'
                    '<p class="hint">Answer 5 quick questions. The answer updates right away.</p></div>',
                    unsafe_allow_html=True)

        st.markdown('<p class="step">1 · Which bus?</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        route = c1.selectbox("Bus line", ["A", "B", "C"], format_func=lambda r: f"Line {r}")
        nst = lambda r, dd: len(lk["sequences"][(r, dd)]) + 1
        direction = c2.selectbox("Which way?", [1, 2],
                                 format_func=lambda dd: f"{'Way 1  →' if dd == 1 else 'Way 2  ←'}  ({nst(route, dd)} stops)")
        seq = lk["sequences"][(route, direction)]
        n_stops = len(seq) + 1
        with st.expander("What are lines A, B, C?"):
            st.markdown(
                "The data has **3 real city bus lines in Astana, Kazakhstan** (official numbers 10, 12 and 46). "
                "The file does not say which line is which, so we grouped the trips by **the stops each bus visits**. "
                "That gave 3 clear groups, which we call **Line A, B and C**.\n\n"
                f"- **Line A:** {nst('A', 1)} stops one way, {nst('A', 2)} the other way · the most trips\n"
                f"- **Line B:** {nst('B', 1)} / {nst('B', 2)} stops\n"
                f"- **Line C:** {nst('C', 1)} / {nst('C', 2)} stops\n\n"
                "Stops are numbered in riding order (Stop 1 = where the bus starts). The data has no stop names, "
                "only ID numbers.")

        st.markdown('<p class="step">2 · Where?</p>', unsafe_allow_html=True)
        o = st.selectbox("I get on at", range(n_stops - 1),
                         format_func=lambda i: f"Stop {i + 1} of {n_stops}   (ID {seq[i][0]})")
        stop_id = lambda i: seq[i][0] if i < len(seq) else seq[-1][1]
        dests = list(range(o + 1, n_stops))            # only stops after the one you get on at
        d = st.selectbox("I get off at", dests, index=min(7, len(dests) - 1),
                         format_func=lambda i: f"Stop {i + 1} of {n_stops}   (ID {stop_id(i)})")
        ride = d - o
        st.caption(f"That is {ride} stop{'s' if ride > 1 else ''} on the bus.")

        st.markdown('<p class="step">3 · When?</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        t_arr = c1.time_input("I reach the bus stop at", time(7, 30), step=300)
        deadline = c2.time_input("I must arrive by", time(8, 10), step=300)
        dow = st.segmented_control("Day", list(range(7)), default=0,
                                   format_func=lambda i: DAYS[i], key="day")
        dow = 0 if dow is None else dow

        c1, c2 = st.columns(2)
        c1.markdown('<p class="step">4 · Weather?</p>', unsafe_allow_html=True)
        rain = c1.segmented_control("Rain", ["☀️ Dry", "🌧️ Rain"], default="☀️ Dry",
                                    label_visibility="collapsed", key="rain") == "🌧️ Rain"
        c2.markdown('<p class="step">5 · Last bus?</p>', unsafe_allow_html=True)
        slow = c2.segmented_control("Traffic", ["🙂 Normal", "🐢 Slow"], default="🙂 Normal",
                                    label_visibility="collapsed", key="traffic") == "🐢 Slow"
        st.caption(f"Slow = the previous bus took about {lk['slow_ratio']:.1f}× its usual time "
                   "(the slowest 5% seen in the data).")

    # ---------------------------------------------------------------- answer
    t0 = datetime.combine(date(2024, 9, 9), t_arr)
    dl = datetime.combine(date(2024, 9, 9), deadline)
    wait, hw, t_board, travel, base, detail = predict_trip(route, direction, o, d, t0, dow, rain, slow)
    t_dest = t_board + timedelta(seconds=travel)
    margin = (dl - t_dest).total_seconds() / 60
    m_txt = f"{abs(margin):.0f} min"
    if margin >= TIGHT_MIN:
        cls, big, do = "ok", "Yes, you'll make it", "✅ Wait for the bus"
        why = f"You arrive about {m_txt} before your deadline."
    elif margin >= 0:
        cls, big, do = "warn", "It's tight", "⚠️ The bus is risky. Consider a motorcycle taxi."
        why = f"Only about {m_txt} to spare. One slow stretch and you're late."
    elif margin >= -LATE_TOL_MIN:
        cls, big, do = "warn", "Maybe a bit late", "⚠️ The bus is risky. Consider a motorcycle taxi."
        why = f"The bus gets you there about {m_txt} after your deadline."
    else:
        cls, big, do = "bad", "You'll be late", "🚕 Switch to Grab / motorcycle taxi now"
        why = f"The bus gets you there about {m_txt} after your deadline."

    with right:
        st.markdown(f"""
<div class="answer {cls}">
  <p class="kick">The answer</p>
  <div class="big">{big}</div>
  <p class="do">{do}</p>
  <p class="why">{why}</p>
</div>
<div class="tl">
  <div><div class="l">You reach the stop</div><div class="t">{t0:%H:%M}</div></div>
  <div><div class="l">Bus comes</div><div class="t">{t_board:%H:%M}</div>
       <div class="d">wait ≈ {wait:.0f} min</div></div>
  <div><div class="l">You arrive</div><div class="t">{t_dest:%H:%M}</div>
       <div class="d">ride ≈ {travel / 60:.0f} min · {ride} stop{'s' if ride > 1 else ''}</div></div>
  <div class="class"><div class="l">Must arrive by</div><div class="t">{dl:%H:%M}</div></div>
</div>
""", unsafe_allow_html=True)
        st.write("")
        with st.expander("How did we get this answer?"):
            st.markdown(
                f"""
1. **Wait for the bus:** at this stop and hour, buses usually come every **{hw:.0f} min**.
   If you arrive at a random moment, you wait about half of that: **{wait:.0f} min**.
2. **Ride time (AI):** the XGBoost model predicts each stop-to-stop part of the trip, using the
   hour, the day, the weather and how slow the last bus was. Total: **{travel / 60:.1f} min**.
   The old way (average for this hour) would say {base / 60:.1f} min.
3. **The advice is a simple rule, not AI:** {TIGHT_MIN}+ min spare → wait ·
   a little time or up to {LATE_TOL_MIN} min late → risky · more than {LATE_TOL_MIN} min late → switch.
""")
            show = detail[["seg_mean", "seg_hour_mean", "pred_s"]].copy()
            show.index = [f"Stop {o + k + 1} → {o + k + 2}" for k in range(len(show))]
            st.dataframe(show.round(0).astype(int).rename(columns={
                "seg_mean": "Usual (s)", "seg_hour_mean": "Usual at this hour (s)",
                "pred_s": "AI prediction (s)"}), width="stretch")
        st.markdown('<p class="note">Data: Astana bus GPS, Jul–Sep 2024 (Bangkok data is not public). '
                    'Stops are shown by number because the data has no stop names.</p>',
                    unsafe_allow_html=True)

# ---------------------------------------------------------------- tab 2
NICE = {"Baseline: mean per segment": "Average per segment",
        "Baseline: mean per segment x hour": "Average per segment × hour"}


def bars(rows, unit, vmax, dec=1):
    """rows: (label, value, kind) with kind in {'base', 'ai', 'hot'} -> HTML bar rows."""
    out = []
    for label, v, kind in rows:
        out.append(f'<div class="bar"><span>{label}</span><div class="tr"><div class="fill {kind}" '
                   f'style="width:{v / vmax * 100:.1f}%"></div></div><span class="v">{v:.{dec}f}{unit}</span></div>')
    return '<div class="bars">' + "".join(out) + "</div>"


def html_table(head, rows, hl=None):
    th = "".join(f'<th class="{"n" if i else ""}">{h}</th>' for i, h in enumerate(head))
    body = "".join(
        f'<tr class="{"hl" if r[0] == hl else ""}">' +
        "".join(f'<td class="{"n" if i else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="scroll"><table class="res"><tr>{th}</tr>{body}</table></div>'


with tab2:
    rep = ROOT / "reports"
    res = pd.read_csv(rep / "results_final_table.csv")
    reg = pd.read_csv(rep / "results_regression.csv")
    late = pd.read_csv(rep / "late_trips_caught.csv")
    jr = pd.read_csv(rep / "results_journey_8stops.csv")
    fi = pd.read_csv(rep / "feature_importance_top12.csv").head(8)
    xgb_before = reg[(reg.model == "XGBoost") & reg.tuned.str.startswith("before")].MAE_s.iloc[0]
    base_mae = res[res.type == "baseline"].MAE_s.min()
    best = res[res.type == "ML"].sort_values("MAE_s").iloc[0]
    rf_late = late[late.method == "Random Forest"].caught_pct.iloc[0]
    sign = lambda v: f"{v:+.1f}%" if abs(v) >= 0.05 else "0.0%"

    st.markdown("Tested on **2 weeks the AI never saw** (9–21 Sep 2024).")
    a, b, c = st.columns(3)
    a.markdown(f'<div class="stat"><div class="n">{base_mae:.0f} → {best.MAE_s:.0f} s</div>'
               f'<div class="s">average error per stop<br>(old way → AI)</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="stat"><div class="n">{best["MAE_improvement_vs_best_baseline_%"]:.0f}% better</div>'
               f'<div class="s">than the best old way<br>(our goal was 15%: not reached)</div></div>',
               unsafe_allow_html=True)
    c.markdown(f'<div class="stat"><div class="n">{rf_late:.0f}%</div>'
               f'<div class="s">of late trips warned<br>(the old way warns 0%)</div></div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    mrows = [(NICE.get(m, m), v, "base" if t == "baseline" else "ai") for m, v, t in zip(res.model, res.MAE_s, res.type)]
    mrows.insert(len(mrows) - 1, ("XGBoost, before tuning", xgb_before, "ai"))
    mrows[-1] = ("XGBoost, tuned", mrows[-1][1], "hot")
    left.markdown('<h2 class="sec">Average error per stop</h2>' +
                  bars(mrows, " s", max(v for _, v, _ in mrows)) +
                  '<p class="note">Seconds, lower is better. Grey = old way (averages), dark = AI.</p>',
                  unsafe_allow_html=True)
    lrows = [(m, v, "base" if i == 0 else ("hot" if m == "Random Forest" else "ai"))
             for i, (m, v) in enumerate(zip(late.method, late.caught_pct))]
    right.markdown('<h2 class="sec">Late trips warned</h2>' + bars(lrows, "%", 100, dec=0) +
                   '<p class="note">Late = an 8-stop trip takes more than 1.2 × its usual time.</p>',
                   unsafe_allow_html=True)

    st.markdown('<h2 class="sec">All results</h2>' + html_table(
        ["Method", "Avg error (s)", "RMSE (s)", "MAPE (%)", "R²", "vs best old way"],
        [[NICE.get(m, m + (" (tuned)" if m in ("Random Forest", "XGBoost") else "")), f"{mae:.2f}", f"{rm:.1f}",
          f"{mp:.1f}", f"{r2:.2f}", sign(imp)]
         for m, mae, rm, mp, r2, imp in zip(res.model, res.MAE_s, res.RMSE_s, res["MAPE_%"], res.R2,
                                             res["MAE_improvement_vs_best_baseline_%"])],
        hl="XGBoost (tuned)"), unsafe_allow_html=True)

    jr = jr[~jr.model.str.contains("round 1")]
    st.markdown('<h2 class="sec">A whole 8-stop trip</h2>'
                '<p class="note" style="margin:0 0 10px">Everything known while you stand at the first stop.</p>' +
                html_table(["Method", "Avg error (s)", "vs best old way"],
                           [[NICE.get(m, m), f"{v:.1f}", sign(i)]
                            for m, v, i in zip(jr.model, jr.MAE_s, jr["MAE_improvement_vs_best_baseline_%"])],
                           hl="Random Forest"), unsafe_allow_html=True)

    frows = [(l, v, "hot" if i == 0 else "ai") for i, (l, v) in enumerate(zip(fi.label, fi.importance_pct))]
    st.markdown('<h2 class="sec">What drives the prediction?</h2>' + bars(frows, "%", frows[0][1]) +
                '<p class="note">Share of XGBoost\'s importance. The hourly average already explains most of the '
                'time, so AI adds 11% on top.</p>', unsafe_allow_html=True)
