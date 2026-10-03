import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Multi-Factor Scanner", layout="wide")

BIN = "https://data-api.binance.vision"
OKX = "https://www.okx.com"
TIMEOUT = 10
BTC_LINKED = ["ETH", "SOL", "ADA", "XRP", "DOGE", "SUI", "LTC", "NEAR", "SNX"]
MIN_QUOTE_VOL = 5_000_000
STABLES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "USD1", "EUR", "AEUR"}
# Ye thresholds conventions hain, proven nahi. Backtest karke badlo.
TRADE_TH = 35
IMB_RATIO = 3.0
fmt = lambda x: f"{x:.6g}"


# ---------------- DATA ----------------
def _get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        return r.json(), "OK"
    except Exception as e:
        return None, f"ERR {type(e).__name__}"


@st.cache_data(ttl=60)
def tickers():
    return _get(f"{BIN}/api/v3/ticker/24hr")


@st.cache_data(ttl=120)
def klines(symbol, interval="1h", limit=300):
    d, s = _get(f"{BIN}/api/v3/klines",
                {"symbol": symbol, "interval": interval, "limit": limit})
    if d is None:
        return None, s
    df = pd.DataFrame(d, columns=["t", "o", "h", "l", "c", "v", "ct", "qv",
                                  "n", "tb", "tq", "ig"])
    for col in ["o", "h", "l", "c", "v", "tb"]:
        df[col] = df[col].astype(float)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    return df[["t", "o", "h", "l", "c", "v", "tb"]], s


@st.cache_data(ttl=60)
def agg_trades(symbol):
    d, s = _get(f"{BIN}/api/v3/aggTrades", {"symbol": symbol, "limit": 1000})
    if d is None or len(d) == 0:
        return None, s
    df = pd.DataFrame(d)
    df["p"] = df["p"].astype(float)
    df["q"] = df["q"].astype(float)
    df["T"] = pd.to_datetime(df["T"], unit="ms")
    return df[["p", "q", "m", "T"]], s


@st.cache_data(ttl=120)
def funding_oi(symbol):
    inst = symbol[:-4] + "-USDT-SWAP"
    f, fs = _get(f"{OKX}/api/v5/public/funding-rate", {"instId": inst})
    o, os_ = _get(f"{OKX}/api/v5/public/open-interest",
                  {"instType": "SWAP", "instId": inst})
    fr = oi = None
    try:
        fr = float(f["data"][0]["fundingRate"])
    except Exception:
        pass
    try:
        oi = float(o["data"][0]["oi"])
    except Exception:
        pass
    return {"funding": fr, "fs": fs, "oi": oi, "os": os_}


# ---------------- INDICATORS ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    pc = df["c"].shift()
    tr = pd.concat([df["h"] - df["l"], (df["h"] - pc).abs(),
                    (df["l"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def swings(df, k=3):
    hs, ls = [], []
    for i in range(k, len(df) - k):
        if df["h"].iloc[i] == df["h"].iloc[i - k:i + k + 1].max():
            hs.append((i, float(df["h"].iloc[i])))
        if df["l"].iloc[i] == df["l"].iloc[i - k:i + k + 1].min():
            ls.append((i, float(df["l"].iloc[i])))
    return hs, ls


def structure(hs, ls):
    if len(hs) < 2 or len(ls) < 2:
        return 0, "Unclear"
    hh = hs[-1][1] > hs[-2][1]
    hl = ls[-1][1] > ls[-2][1]
    if hh and hl:
        return 1, "Bullish (HH+HL)"
    if not hh and not hl:
        return -1, "Bearish (LH+LL)"
    return 0, "Range / mixed"


def vprofile(df, bins=40):
    lo, hi = df["l"].min(), df["h"].max()
    if hi <= lo:
        return None
    edges = np.linspace(lo, hi, bins + 1)
    tp = ((df["h"] + df["l"] + df["c"]) / 3).values
    idx = np.clip(np.searchsorted(edges, tp, side="right") - 1, 0, bins - 1)
    vol = np.zeros(bins)
    for i, v in zip(idx, df["v"].values):
        vol[i] += v
    poc = int(vol.argmax())
    total, acc, a, b = vol.sum(), vol[poc], poc, poc
    while acc < 0.7 * total and (a > 0 or b < bins - 1):
        up = vol[b + 1] if b < bins - 1 else -1
        dn = vol[a - 1] if a > 0 else -1
        if up >= dn:
            b += 1
            acc += up
        else:
            a -= 1
            acc += dn
    return {"poc": float((edges[poc] + edges[poc + 1]) / 2),
            "vah": float(edges[b + 1]), "val": float(edges[a])}


def imbalances(tr, bins=20):
    lo, hi = tr["p"].min(), tr["p"].max()
    if hi <= lo:
        return [], []
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.searchsorted(edges, tr["p"].values, side="right") - 1, 0, bins - 1)
    buy, sell = np.zeros(bins), np.zeros(bins)
    for i, q, m in zip(idx, tr["q"].values, tr["m"].values):
        if m:
            sell[i] += q
        else:
            buy[i] += q
    minv = tr["q"].sum() / bins * 0.5
    bz, sz = [], []
    for i in range(bins):
        mid = float((edges[i] + edges[i + 1]) / 2)
        if buy[i] >= IMB_RATIO * max(sell[i], 1e-12) and buy[i] >= minv:
            bz.append(mid)
        if sell[i] >= IMB_RATIO * max(buy[i], 1e-12) and sell[i] >= minv:
            sz.append(mid)
    return bz, sz


# ---------------- ANALYSIS ----------------
def make_plan(sig, price, a, hs, ls, vp):
    if sig not in ("BUY", "SELL"):
        return None
    d = 1 if sig == "BUY" else -1
    if d == 1:
        near = [p for _, p in ls if p < price and price - p <= 3 * a]
        sl = (max(near) - 0.2 * a) if near else price - 1.5 * a
        sl = min(sl, price - 1.0 * a)
    else:
        near = [p for _, p in hs if p > price and p - price <= 3 * a]
        sl = (min(near) + 0.2 * a) if near else price + 1.5 * a
        sl = max(sl, price + 1.0 * a)
    risk = abs(price - sl)
    raw = [p for _, p in (hs if d == 1 else ls)]
    if vp:
        raw += [vp["vah"], vp["poc"]] if d == 1 else [vp["val"], vp["poc"]]
    cands = sorted([p for p in raw if d * (p - price) >= risk], key=lambda p: d * p)
    tps = []
    for p in cands:
        if not tps or d * (p - tps[-1]) >= 0.5 * risk:
            tps.append(p)
        if len(tps) == 3:
            break
    for m in (1.5, 2.5, 4.0)[len(tps):]:
        nxt = price + d * risk * m
        if tps and d * (nxt - tps[-1]) <= 0:
            nxt = tps[-1] + d * 0.5 * risk
        tps.append(nxt)
    return {"entry": price, "sl": sl, "tps": tps, "rr1": abs(tps[0] - price) / risk}


def analyze(sym):
    k, ks = klines(sym, "1h", 300)
    if k is None or len(k) < 60:
        return {"err": f"Klines unavailable ({ks})"}
    c = k["c"]
    price = float(c.iloc[-1])
    e20, e50, e200 = (float(ema(c, n).iloc[-1]) for n in (20, 50, 200))
    r = float(rsi(c).iloc[-1])
    r = 50.0 if np.isnan(r) else r
    a = float(atr(k).iloc[-1])
    hs, ls = swings(k)
    sdir, stxt = structure(hs, ls)
    vp = vprofile(k.tail(168))
    F = []  # (factor, score, data_ok, note)

    # Trend
    if price > e20 > e50 > e200:
        F.append(("Trend (EMA)", 20, True, "Price>EMA20>50>200"))
    elif price < e20 < e50 < e200:
        F.append(("Trend (EMA)", -20, True, "Bearish EMA stack"))
    else:
        F.append(("Trend (EMA)", 8 if price > e50 else -8, True, "Mixed EMAs"))
    F.append(("Structure", sdir * 15, True, stxt))
    rs = -5 if r > 75 else 5 if r >= 50 else -5 if r > 25 else 5
    F.append(("RSI", rs, True, f"RSI {r:.1f}"))

    # Volume profile (7d, 1h candles)
    if vp:
        ps = 10 if price > vp["vah"] else 5 if price > vp["poc"] else -10 if price < vp["val"] else -5
        F.append(("Volume profile", ps, True,
                  f'POC {fmt(vp["poc"])} | VAH {fmt(vp["vah"])} | VAL {fmt(vp["val"])}'))
    else:
        F.append(("Volume profile", 0, False, "DATA UNAVAILABLE"))

    # Delta (24h, taker-buy volume from klines)
    l24 = k.tail(24)
    tot = l24["v"].sum()
    dpct = float((2 * l24["tb"] - l24["v"]).sum() / tot) if tot > 0 else 0.0
    F.append(("Delta (24h)", float(np.clip(dpct * 300, -15, 15)), True,
              f"Net taker delta {dpct * 100:+.1f}% of volume"))

    # Imbalances + Absorption (recent aggTrades)
    tr, ts = agg_trades(sym)
    bz, sz = [], []
    if tr is None:
        F.append(("Imbalances", 0, False, f"DATA UNAVAILABLE ({ts})"))
        F.append(("Absorption", 0, False, f"DATA UNAVAILABLE ({ts})"))
    else:
        mins = max((tr["T"].max() - tr["T"].min()).total_seconds() / 60, 1.0)
        bz, sz = imbalances(tr)
        F.append(("Imbalances", float(np.clip(3 * (len(bz) - len(sz)), -10, 10)), True,
                  f"Buy zones {len(bz)} | Sell zones {len(sz)} ({mins:.0f} min window)"))
        buy = tr.loc[~tr["m"], "q"].sum()
        sell = tr.loc[tr["m"], "q"].sum()
        wd = (buy - sell) / (buy + sell) if buy + sell > 0 else 0
        exp = a * (mins / 60) ** 0.5
        resp = (tr["p"].iloc[-1] - tr["p"].iloc[0]) / exp if exp > 0 else 0
        if wd >= 0.2 and resp <= 0.25:
            F.append(("Absorption", -10, True, "Buy aggression absorbed (bearish, heuristic)"))
        elif wd <= -0.2 and resp >= -0.25:
            F.append(("Absorption", 10, True, "Sell aggression absorbed (bullish, heuristic)"))
        else:
            F.append(("Absorption", 0, True, "None detected"))

    # Funding / OI (OKX)
    fo = funding_oi(sym)
    if fo["funding"] is None:
        F.append(("Funding", 0, False, f'DATA UNAVAILABLE ({fo["fs"]})'))
    else:
        fr = fo["funding"]
        fs_ = -5 if fr > 0.0005 else 5 if fr < -0.0002 else 0
        F.append(("Funding", fs_, True, f"{fr * 100:.4f}%"))
    if fo["oi"] is None:
        F.append(("Open interest", 0, False, f'DATA UNAVAILABLE ({fo["os"]})'))
    else:
        F.append(("Open interest", 0, True, f'OI {fo["oi"]:,.0f} contracts (change: N/A, history chahiye)'))
    F.append(("Gamma levels", 0, False, "DATA UNAVAILABLE (options data source nahi)"))

    total = sum(f[1] for f in F)
    sign = 1 if total > 0 else -1
    agree = sum(1 for f in F if f[1] != 0 and (f[1] > 0) == (sign > 0))
    if abs(total) < 15:
        sig = "NO TRADE"
    elif abs(total) >= TRADE_TH and agree >= 4:
        sig = "BUY" if total > 0 else "SELL"
    else:
        sig = "HOLD"
    lean = "Bullish lean" if total > 0 else "Bearish lean" if total < 0 else "Neutral"
    plan = make_plan(sig, price, a, hs, ls, vp)
    return {"k": k, "price": price, "F": F, "total": total, "agree": agree,
            "sig": sig, "lean": lean, "plan": plan, "vp": vp, "atr": a,
            "bz": bz, "sz": sz}


# ---------------- UI ----------------
def pick(s):
    if s:
        st.session_state["sel"] = s


def show(sym):
    st.divider()
    st.header(sym)
    with st.spinner("Analyzing..."):
        R = analyze(sym)
    if "err" in R:
        st.error(R["err"])
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Price", fmt(R["price"]))
    c2.metric("Signal", R["sig"])
    c3.metric("Score", f'{R["total"]:+.0f}', f'{R["agree"]} factors agree')
    st.caption(R["lean"])
    st.dataframe(pd.DataFrame(
        [(n, round(s, 1), "OK" if ok else "N/A", note) for n, s, ok, note in R["F"]],
        columns=["Factor", "Score", "Data", "Note"]), hide_index=True)

    vp = R["vp"]
    if vp:
        st.subheader("Next move scenarios")
        st.write(f'- **Bullish confirm:** 1h close {fmt(vp["vah"])} ke upar')
        st.write(f'- **Bearish confirm:** 1h close {fmt(vp["val"])} ke neeche')
        st.write(f'- **Beech mein ({fmt(vp["val"])} - {fmt(vp["vah"])}):** range, POC {fmt(vp["poc"])} magnet')
        st.caption("Ye scenarios hain, prediction nahi.")

    st.subheader("Entry / SL / TP")
    p = R["plan"]
    if p:
        st.write(f'**{R["sig"]}** | Entry `{fmt(p["entry"])}` | SL `{fmt(p["sl"])}`')
        st.write(" | ".join(f"TP{i + 1} `{fmt(t)}`" for i, t in enumerate(p["tps"])))
        st.caption(f'TP1 R:R = {p["rr1"]:.2f} | Sirf paper trade')
    else:
        st.warning("Plan nahi: kaafi factors agree nahi karte. Entry/SL/TP khaali.")

    k = R["k"].tail(120)
    fig = go.Figure(go.Candlestick(x=k["t"], open=k["o"], high=k["h"],
                                   low=k["l"], close=k["c"], name="Price"))
    if vp:
        for nm, v in (("POC", vp["poc"]), ("VAH", vp["vah"]), ("VAL", vp["val"])):
            fig.add_hline(y=v, line_dash="dot", annotation_text=nm)
    if p:
        fig.add_hline(y=p["sl"], line_color="red", annotation_text="SL")
        for i, t in enumerate(p["tps"]):
            fig.add_hline(y=t, line_color="green", annotation_text=f"TP{i + 1}")
    fig.update_layout(height=420, xaxis_rangeslider_visible=False,
                      margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig)


def linked_rows(by_sym):
    btc, _ = klines("BTCUSDT", "1h", 300)
    out = []
    if btc is None:
        return out
    br = btc.set_index("t")["c"].pct_change()
    for b in BTC_LINKED:
        s = b + "USDT"
        kk, _ = klines(s, "1h", 300)
        corr = None
        if kk is not None:
            corr = kk.set_index("t")["c"].pct_change().corr(br)
        chg = float(by_sym[s]["priceChangePercent"]) if s in by_sym else None
        out.append({"Pair": s, "Corr vs BTC (1h, 300c)": None if corr is None else round(corr, 2),
                    "24h %": chg})
    return out


st.title("Crypto Multi-Factor Scanner")
st.caption("Research / paper-trading only. Financial advice nahi.")
st.session_state.setdefault("sel", "BTCUSDT")

tk, tks = tickers()
fo_btc = funding_oi("BTCUSDT")
st.sidebar.subheader("Data status")
st.sidebar.write({"Binance mirror": tks, "OKX funding": fo_btc["fs"], "OKX OI": fo_btc["os"]})

st.text_input("Koi bhi pair likho (e.g. SOLUSDT) aur Enter", key="manual",
              on_change=lambda: pick(st.session_state.get("manual", "").strip().upper()))

if tk is None:
    st.error(f"Market data unavailable: {tks}. Sidebar status check karo.")
else:
    rows = [x for x in tk if x["symbol"].endswith("USDT")
            and x["symbol"][:-4] not in STABLES
            and float(x["quoteVolume"]) > MIN_QUOTE_VOL]
    rows.sort(key=lambda x: float(x["priceChangePercent"]), reverse=True)
    gain, lose = rows[:10], rows[-10:][::-1]
    by_sym = {x["symbol"]: x for x in tk}
    tg, tl, tb = st.tabs(["Top 10 Gainers", "Top 10 Losers", "BTC-linked"])
    for tab, lst, pre in ((tg, gain, "g_"), (tl, lose, "l_")):
        with tab:
            for x in lst:
                st.button(f'{x["symbol"]}  {float(x["priceChangePercent"]):+.2f}%  ${fmt(float(x["lastPrice"]))}',
                          key=pre + x["symbol"], on_click=pick, args=(x["symbol"],))
    with tb:
        lr = linked_rows(by_sym)
        if lr:
            st.dataframe(pd.DataFrame(lr), hide_index=True)
        cols = st.columns(3)
        for i, b in enumerate(BTC_LINKED + ["BTC"]):
            cols[i % 3].button(b + "USDT", key="b_" + b, on_click=pick, args=(b + "USDT",))

show(st.session_state["sel"])
