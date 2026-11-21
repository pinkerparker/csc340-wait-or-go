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
.block-container { padding-top:1.6rem; max-width:1250px; }
header[data-testid="stHeader"] { background:transparent; }
h1,h2,h3 { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif,'Futura','Avenir Next','Arial Black',sans-serif !important; color:var(--ink); }

.kick { font-family:'Jost','Futura','Avenir Next',sans-serif; font-size:.95rem; letter-spacing:.12em; text-transform:uppercase; margin:0; }
.hero { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:4.1rem; line-height:.95;
        color:var(--olive); -webkit-text-stroke:2px var(--ink); text-transform:uppercase; margin:.2rem 0 .8rem; }
.band { background:var(--olive); border:2px solid var(--ink); padding:.45rem 1rem; display:inline-block; }
.band span { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:900; font-size:1.45rem; color:var(--pink);
             -webkit-text-stroke:1px var(--ink); text-transform:uppercase; letter-spacing:.01em; }
.road { height:46px; background:var(--road); border-top:8px solid #333; margin:1.2rem 0 1.4rem; position:relative; }
.road::after { content:""; position:absolute; left:0; right:0; top:17px; height:5px;
               background:repeating-linear-gradient(90deg, var(--cream) 0 60px, transparent 60px 110px); }

.card { background:var(--mustard); border:2px solid var(--ink); padding:1.1rem 1.3rem 1.2rem; }
.card h3 { font-family:'League Spartan','Futura','Avenir Next','Arial Black',sans-serif; font-weight:800; font-size:1.35rem; margin:0 0 .2rem; }
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

tab1, tab2 = st.tabs(["🚌  Plan my trip", "📊  How good is the AI?"])

with tab1:
    left, right = st.columns([1, 1.25], gap="large")

    with left:
        st.markdown('<div class="card"><h3>Your trip</h3>'
                    '<p class="hint">Answer 5 quick questions. The answer updates right away.</p></div>',
                    unsafe_allow_html=True)

        st.markdown('<p class="step">1 · Which bus?</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        route = c1.selectbox("Bus line", ["A", "B", "C"], format_func=lambda r: f"Line {r}",
                             help="One of the 3 real bus lines in the data. See 'What are lines A, B, C?' below.")
        nst = lambda r, dd: len(lk["sequences"][(r, dd)]) + 1
        direction = c2.selectbox("Which way?", [1, 2],
                                 format_func=lambda dd: f"{'Way 1  →' if dd == 1 else 'Way 2  ←'}  ({nst(route, dd)} stops)",
                                 help="Each line runs both ways. Way 1 and Way 2 are the two directions in the data.")
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
        deadline = c2.time_input("I must arrive by", time(8, 10), step=300,
                                 help="Class, work, a meeting, a flight… any time you must be there.")
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
with tab2:
    rep = ROOT / "reports"
    res = pd.read_csv(rep / "results_final_table.csv")
    clf = pd.read_csv(rep / "results_classification.csv")
    base_mae = res[res.type == "baseline"].MAE_s.min()
    best = res[res.type == "ML"].sort_values("MAE_s").iloc[0]
    late = clf[clf.model.str.startswith("Random Forest")].recall_late.iloc[0]
    st.markdown("Tested on **2 weeks the AI never saw** (9–21 Sep 2024).")
    a, b, c = st.columns(3)
    a.markdown(f'<div class="stat"><div class="n">{base_mae:.0f} → {best.MAE_s:.0f} s</div>'
               f'<div class="s">average error per stop<br>(old way → AI)</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="stat"><div class="n">{best["MAE_improvement_vs_best_baseline_%"]:.0f}% better</div>'
               f'<div class="s">than the best old way<br>(our goal was 15%: not reached)</div></div>',
               unsafe_allow_html=True)
    c.markdown(f'<div class="stat"><div class="n">{late * 100:.0f}%</div>'
               f'<div class="s">of late trips warned<br>(the old way warns 0%)</div></div>', unsafe_allow_html=True)
    st.write("")
    nice = res[["model", "MAE_s", "RMSE_s", "MAPE_%", "R2", "MAE_improvement_vs_best_baseline_%"]].rename(columns={
        "model": "Method", "MAE_s": "Avg error (s)", "RMSE_s": "RMSE (s)", "MAPE_%": "MAPE (%)",
        "MAE_improvement_vs_best_baseline_%": "vs best old way (%)"})
    st.dataframe(nice.round(2), hide_index=True, width="stretch")
    j = rep / "results_journey_8stops.csv"
    if j.exists():
        st.markdown("**A whole 8-stop trip** (everything known while you stand at the first stop):")
        jt = pd.read_csv(j).drop(columns=["n_journeys"], errors="ignore").rename(columns={
            "model": "Method", "MAE_s": "Avg error (s)", "RMSE_s": "RMSE (s)", "MAPE_%": "MAPE (%)",
            "MAE_improvement_vs_best_baseline_%": "vs best old way (%)"})
        st.dataframe(jt.round(2), hide_index=True, width="stretch")
    x, y = st.columns(2)
    x.image(str(rep / "mae_model_vs_baseline.png"), width="stretch")
    y.image(str(rep / "feature_importance.png"), width="stretch")
