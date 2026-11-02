import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime
import pytz

st.set_page_config(page_title="F&O VWAP + RSI Screener", layout="wide")

# ------------------------------------------------------------------
# F&O STOCKS LIST (NSE) - ज़रूरत हो तो यहाँ add/remove करें
# ------------------------------------------------------------------
FNO_STOCKS = [
    "AARTIIND", "ABB", "ABBOTINDIA", "ABCAPITAL", "ABFRL", "ACC", "ADANIENT", "ADANIPORTS",
    "ALKEM", "AMBUJACEM", "ANGELONE", "APOLLOHOSP", "APOLLOTYRE", "ASHOKLEY", "ASIANPAINT",
    "ASTRAL", "ATGL", "AUBANK", "AUROPHARMA", "AXISBANK", "BAJAJ-AUTO", "BAJAJFINSV",
    "BAJFINANCE", "BALKRISIND", "BANDHANBNK", "BANKBARODA", "BANKINDIA", "BDL", "BEL",
    "BHARATFORG", "BHARTIARTL", "BHEL", "BIOCON", "BOSCHLTD", "BPCL", "BRITANNIA", "BSE",
    "BSOFT", "CAMS", "CANBK", "CDSL", "CESC", "CGPOWER", "CHAMBLFERT", "CHOLAFIN", "CIPLA",
    "COALINDIA", "COFORGE", "COLPAL", "CONCOR", "CROMPTON", "CUMMINSIND", "CYIENT", "DABUR",
    "DALBHARAT", "DEEPAKNTR", "DELHIVERY", "DIVISLAB", "DIXON", "DLF", "DMART", "DRREDDY",
    "EICHERMOT", "ESCORTS", "EXIDEIND", "FEDERALBNK", "GAIL", "GLENMARK", "GMRAIRPORT",
    "GODREJCP", "GODREJPROP", "GRANULES", "GRASIM", "HAL", "HAVELLS", "HCLTECH", "HDFCAMC",
    "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HFCL", "HINDALCO", "HINDCOPPER", "HINDPETRO",
    "HINDUNILVR", "HUDCO", "ICICIBANK", "ICICIGI", "ICICIPRULI", "IDEA", "IDFCFIRSTB", "IEX",
    "IGL", "IIFL", "INDHOTEL", "INDIANB", "INDIGO", "INDUSINDBK", "INDUSTOWER", "INFY",
    "INOXWIND", "IOC", "IRB", "IRCTC", "IREDA", "IRFC", "ITC", "JINDALSTEL", "JIOFIN",
    "JSWENERGY", "JSWSTEEL", "JUBLFOOD", "KALYANKJIL", "KAYNES", "KEI", "KFINTECH",
    "KOTAKBANK", "KPITTECH", "LAURUSLABS", "LICHSGFIN", "LICI", "LODHA", "LT", "LTF",
    "LTIM", "LUPIN", "M&M", "M&MFIN", "MANAPPURAM", "MANKIND", "MARICO", "MARUTI",
    "MAXHEALTH", "MAZDOCK", "MCX", "MFSL", "MGL", "MOTHERSON", "MPHASIS", "MUTHOOTFIN",
    "NATIONALUM", "NAUKRI", "NBCC", "NCC", "NESTLEIND", "NHPC", "NMDC", "NTPC", "NYKAA",
    "OBEROIRLTY", "OFSS", "OIL", "ONGC", "PAGEIND", "PATANJALI", "PAYTM", "PERSISTENT",
    "PETRONET", "PFC", "PGEL", "PHOENIXLTD", "PIDILITIND", "PIIND", "PNB", "PNBHOUSING",
    "POLICYBZR", "POLYCAB", "POONAWALLA", "POWERGRID", "PPLPHARMA", "PRESTIGE", "RBLBANK",
    "RECLTD", "RELIANCE", "RVNL", "SAIL", "SBICARD", "SBILIFE", "SBIN", "SHREECEM",
    "SHRIRAMFIN", "SIEMENS", "SOLARINDS", "SONACOMS", "SRF", "SUNPHARMA", "SUPREMEIND",
    "SUZLON", "SYNGENE", "TATACHEM", "TATACONSUM", "TATAELXSI", "TATAMOTORS", "TATAPOWER",
    "TATASTEEL", "TATATECH", "TCS", "TECHM", "TIINDIA", "TITAGARH", "TITAN", "TORNTPHARM",
    "TORNTPOWER", "TRENT", "TVSMOTOR", "ULTRACEMCO", "UNIONBANK", "UNITDSPR", "UNOMINDA",
    "UPL", "VBL", "VEDL", "VOLTAS", "WIPRO", "YESBANK", "ZYDUSLIFE",
]

IST = pytz.timezone("Asia/Kolkata")


# ------------------------------------------------------------------
# Indicators
# ------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-12)
    return 100 - (100 / (1 + rs))


def vwap(df: pd.DataFrame) -> pd.Series:
    """रोज़ का VWAP, हर दिन 9:15 से reset"""
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].fillna(0)
    idx = df.index.tz_convert(IST) if df.index.tz is not None else df.index
    day = pd.Series(idx.date, index=df.index)
    pv = (typical * vol).groupby(day).cumsum()
    cv = vol.groupby(day).cumsum()
    return pv / cv.replace(0, float("nan"))


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def download(symbols: tuple, interval: str):
    tickers = [s + ".NS" for s in symbols]
    return yf.download(
        tickers,
        period="5d",
        interval=interval,
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )


def get_ohlcv(data, symbol, min_len=20):
    t = symbol + ".NS"
    try:
        if t not in data.columns.get_level_values(0):
            return None
        df = data[t][["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
    except Exception:
        return None
    return df if len(df) >= min_len else None


def drop_incomplete(df, minutes):
    """अगर आख़िरी candle अभी चल रही है तो उसे हटाओ (बंद candle ही गिनो)"""
    if df is None or df.empty:
        return df
    last = df.index[-1]
    last = last.tz_convert("UTC") if last.tzinfo else last.tz_localize("UTC")
    if last + pd.Timedelta(minutes=minutes) > pd.Timestamp.now(tz="UTC"):
        return df.iloc[:-1]
    return df


def ist_day(ts):
    return ts.tz_convert(IST).date() if ts.tzinfo else ts.date()


def fmt_time(ts):
    try:
        return ts.tz_convert(IST).strftime("%H:%M")
    except Exception:
        return str(ts)


# ------------------------------------------------------------------
# STEP 1: BASE LIST (VWAP + 15-minute candle)
#   CE: Price > VWAP  और  आख़िरी बंद 15m candle ने, उसी दिन की
#       किसी भी पिछली 15m candle के HIGH को अभी-अभी ऊपर cross करके close किया
#   PE: Price < VWAP  और  आख़िरी बंद 15m candle ने, उसी दिन की
#       किसी भी पिछली 15m candle के LOW को अभी-अभी नीचे cross करके close किया
#
#   "अभी-अभी" = पिछली 15m candle का close level के इस तरफ था,
#                आख़िरी 15m candle का close level के उस पार है
# ------------------------------------------------------------------
def base_lists(symbols):
    d15_all = download(tuple(symbols), "15m")
    d2_all = download(tuple(symbols), "2m")
    ce, pe = {}, {}

    for sym in symbols:
        d2 = get_ohlcv(d2_all, sym)
        d15 = get_ohlcv(d15_all, sym, min_len=3)
        if d2 is None or d15 is None:
            continue

        # अभी का price और VWAP (2m data से)
        price = float(d2["Close"].iloc[-1])
        v = vwap(d2).iloc[-1]
        if pd.isna(v):
            continue
        v = float(v)

        d15 = drop_incomplete(d15, 15)
        if d15 is None or len(d15) < 2:
            continue

        c_ts = d15.index[-1]
        day_df = d15[[ist_day(t) == ist_day(c_ts) for t in d15.index]]
        if len(day_df) < 2:
            continue

        C = day_df.iloc[-1]            # आख़िरी बंद 15m candle
        P = day_df.iloc[-2]            # उससे पिछली candle
        earlier = day_df.iloc[:-1]     # आज की सभी पिछली 15m candles
        c_close = float(C["Close"])
        p_close = float(P["Close"])

        info = {
            "Price": round(price, 2),
            "VWAP": round(v, 2),
            "15m Close": round(c_close, 2),
            "15m Time": fmt_time(day_df.index[-1]),
        }

        # ---- CE: किसी भी पिछली 15m High को ऊपर cross ----
        if price > v:
            highs = earlier["High"]
            crossed = highs[(highs >= p_close) & (highs < c_close)]
            if not crossed.empty:
                ce[sym] = {**info, "Crossed 15m High": round(float(crossed.max()), 2)}

        # ---- PE: किसी भी पिछली 15m Low को नीचे cross ----
        if price < v:
            lows = earlier["Low"]
            crossed = lows[(lows <= p_close) & (lows > c_close)]
            if not crossed.empty:
                pe[sym] = {**info, "Crossed 15m Low": round(float(crossed.min()), 2)}

    return ce, pe


# ------------------------------------------------------------------
# STEP 2: उसी list में से RSI cross
#   CE: RSI 40/60 को नीचे से ऊपर cross
#   PE: RSI 60/40 को ऊपर से नीचे cross
# ------------------------------------------------------------------
def rsi_scan(interval, minutes, candidates, side, lookback):
    if not candidates:
        return pd.DataFrame()
    data = download(tuple(candidates.keys()), interval)
    rows = []

    for sym, info in candidates.items():
        df = get_ohlcv(data, sym)
        if df is None:
            continue
        df = drop_incomplete(df, minutes)
        if df is None or len(df) < 20:
            continue
        r = rsi(df["Close"]).dropna()
        if len(r) < lookback + 1:
            continue

        for i in range(1, lookback + 1):
            prev, cur = float(r.iloc[-i - 1]), float(r.iloc[-i])
            levels = []
            if side == "CE":
                if prev < 40 <= cur:
                    levels.append("40")
                if prev < 60 <= cur:
                    levels.append("60")
            else:
                if prev > 60 >= cur:
                    levels.append("60")
                if prev > 40 >= cur:
                    levels.append("40")
            if levels:
                rows.append({
                    "Stock": sym,
                    "RSI Level": " & ".join(levels),
                    "Prev RSI": round(prev, 1),
                    "Cross RSI": round(cur, 1),
                    "RSI (now)": round(float(r.iloc[-1]), 1),
                    "Cross candle": fmt_time(r.index[-i]),
                    **info,
                })
                break

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Cross candle", ascending=False)
    return out


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📈 F&O VWAP + 15m + RSI(14) Screener")
now = datetime.now(IST)
st.caption(f"अभी: {now.strftime('%d-%b-%Y %H:%M:%S')} IST  |  Data: Yahoo Finance (कुछ delay संभव)")

with st.sidebar:
    st.header("Settings")
    lookback = st.slider("RSI cross पिछली कितनी candles में देखें", 1, 5, 2)
    st.write(f"कुल F&O stocks: **{len(FNO_STOCKS)}**")
    if st.button("🔄 Refresh / Scan Now"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("Base list (VWAP + 15m) बन रही है..."):
    ce_base, pe_base = base_lists(FNO_STOCKS)

st.write(f"✅ Base list: CE **{len(ce_base)}** stocks | PE **{len(pe_base)}** stocks")

tab_ce, tab_pe = st.tabs(["🟢 CE Stocks", "🔴 PE Stocks"])

with tab_ce:
    st.caption("Price > VWAP + किसी 15m High के ऊपर अभी close → फिर RSI(14) 40/60 को नीचे से ऊपर cross")
    with st.spinner("CE scan..."):
        ce2 = rsi_scan("2m", 2, ce_base, "CE", lookback)
        ce5 = rsi_scan("5m", 5, ce_base, "CE", lookback)
    st.subheader(f"⏱ 2 Minute CE ({len(ce2)})")
    st.dataframe(ce2, use_container_width=True, hide_index=True)
    st.subheader(f"⏱ 5 Minute CE ({len(ce5)})")
    st.dataframe(ce5, use_container_width=True, hide_index=True)
    with st.expander(f"Base CE list ({len(ce_base)})"):
        st.dataframe(pd.DataFrame(ce_base).T, use_container_width=True)

with tab_pe:
    st.caption("Price < VWAP + किसी 15m Low के नीचे अभी close → फिर RSI(14) 60/40 को ऊपर से नीचे cross")
    with st.spinner("PE scan..."):
        pe2 = rsi_scan("2m", 2, pe_base, "PE", lookback)
        pe5 = rsi_scan("5m", 5, pe_base, "PE", lookback)
    st.subheader(f"⏱ 2 Minute PE ({len(pe2)})")
    st.dataframe(pe2, use_container_width=True, hide_index=True)
    st.subheader(f"⏱ 5 Minute PE ({len(pe5)})")
    st.dataframe(pe5, use_container_width=True, hide_index=True)
    with st.expander(f"Base PE list ({len(pe_base)})"):
        st.dataframe(pd.DataFrame(pe_base).T, use_container_width=True)

st.caption("⚠️ सिर्फ screening tool है, trading advice नहीं। Entry से पहले chart खुद confirm करें।")
