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


def prev_closed(df, ref_ts, minutes):
    """
    ref_ts = सबसे ताज़ा (अभी वाली) candle का समय।
    ref_ts जिस `minutes` वाली candle में आता है, वह "अभी वाली" candle है।
    उससे पहले की सारी candles "बंद" मानी जाती हैं -> आख़िरी = "पिछली candle"।
    (बाज़ार खुला हो या बंद, chart जैसा ही नतीजा)
    """
    cut = ref_ts.floor(f"{minutes}min")
    return df[df.index < cut]


def fmt_time(ts):
    try:
        return ts.tz_convert(IST).strftime("%H:%M")
    except Exception:
        return str(ts)


# ------------------------------------------------------------------
# STEP 1: CE STOCKS / PE STOCKS (base lists)
#   CE: अभी का price > पिछली 15m candle का close  AND  price > VWAP
#   PE: अभी का price < पिछली 15m candle का close  AND  price < VWAP
# ------------------------------------------------------------------
def base_lists(symbols):
    d15_all = download(tuple(symbols), "15m")
    d2_all = download(tuple(symbols), "2m")
    ce, pe, refs = {}, {}, {}

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

        ref_ts = d2.index[-1]                       # अभी वाली candle का समय
        prev15 = prev_closed(d15, ref_ts, 15)       # पिछली बंद 15m candles
        if prev15.empty:
            continue

        prev15_close = float(prev15["Close"].iloc[-1])

        info = {
            "Price": round(price, 2),
            "VWAP": round(v, 2),
            "Prev 15m Close": round(prev15_close, 2),
            "Prev 15m Time": fmt_time(prev15.index[-1]),
            "Price vs VWAP %": round((price - v) / v * 100, 2),
        }

        if price > prev15_close and price > v:
            ce[sym] = info
            refs[sym] = ref_ts
        elif price < prev15_close and price < v:
            pe[sym] = info
            refs[sym] = ref_ts

    return ce, pe, refs


# ------------------------------------------------------------------
# STEP 2: Base list में से RSI(14) cross
#   CE: RSI ने 40 या 60 को नीचे से ऊपर cross किया
#   PE: RSI ने 60 या 40 को ऊपर से नीचे cross किया
#   lookback = 1  ->  सिर्फ पिछली (बंद) candle
# ------------------------------------------------------------------
def rsi_scan(interval, minutes, candidates, refs, side, lookback):
    if not candidates:
        return pd.DataFrame()
    data = download(tuple(candidates.keys()), interval)
    rows = []

    for sym, info in candidates.items():
        df = get_ohlcv(data, sym)
        if df is None:
            continue
        df = prev_closed(df, refs[sym], minutes)
        if len(df) < 20:
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
                    "Cross candle": fmt_time(r.index[-i]),
                    **info,
                })
                break

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("Cross candle", ascending=False)
    return out


def base_table(d):
    df = pd.DataFrame(d).T
    if df.empty:
        return df
    df.insert(0, "Stock", df.index)
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📈 F&O VWAP + 15m + RSI(14) Screener")
now = datetime.now(IST)
st.caption(f"अभी: {now.strftime('%d-%b-%Y %H:%M:%S')} IST  |  Data: Yahoo Finance (कुछ delay संभव)")

with st.sidebar:
    st.header("Settings")
    lookback = st.slider(
        "RSI cross कितनी पिछली candles में देखें",
        1, 3, 1,
        help="1 = सिर्फ पिछली बंद candle (आपकी शर्त)। बढ़ाने पर थोड़ी पुरानी cross भी दिखेंगी।",
    )
    st.write(f"कुल F&O stocks: **{len(FNO_STOCKS)}**")
    if st.button("🔄 Refresh / Scan Now"):
        st.cache_data.clear()
        st.rerun()

with st.spinner("CE / PE Stocks list बन रही है..."):
    ce_base, pe_base, refs = base_lists(FNO_STOCKS)

tab_ce, tab_pe = st.tabs([f"🟢 CE ({len(ce_base)})", f"🔴 PE ({len(pe_base)})"])

with tab_ce:
    st.subheader(f"🟢 CE Stocks ({len(ce_base)})")
    st.caption("Price > पिछली 15m candle का close  और  Price > VWAP")
    st.dataframe(base_table(ce_base), use_container_width=True, hide_index=True)

    with st.spinner("CE RSI scan..."):
        ce2 = rsi_scan("2m", 2, ce_base, refs, "CE", lookback)
        ce5 = rsi_scan("5m", 5, ce_base, refs, "CE", lookback)

    st.subheader(f"⏱ 2 Minute CE ({len(ce2)})")
    st.caption("CE Stocks में से — 2m RSI(14) ने 40 या 60 को ऊपर की ओर cross किया")
    st.dataframe(ce2, use_container_width=True, hide_index=True)

    st.subheader(f"⏱ 5 Minute CE ({len(ce5)})")
    st.caption("CE Stocks में से — 5m RSI(14) ने 40 या 60 को ऊपर की ओर cross किया")
    st.dataframe(ce5, use_container_width=True, hide_index=True)

with tab_pe:
    st.subheader(f"🔴 PE Stocks ({len(pe_base)})")
    st.caption("Price < पिछली 15m candle का close  और  Price < VWAP")
    st.dataframe(base_table(pe_base), use_container_width=True, hide_index=True)

    with st.spinner("PE RSI scan..."):
        pe2 = rsi_scan("2m", 2, pe_base, refs, "PE", lookback)
        pe5 = rsi_scan("5m", 5, pe_base, refs, "PE", lookback)

    st.subheader(f"⏱ 2 Minute PE ({len(pe2)})")
    st.caption("PE Stocks में से — 2m RSI(14) ने 60 या 40 को नीचे की ओर cross किया")
    st.dataframe(pe2, use_container_width=True, hide_index=True)

    st.subheader(f"⏱ 5 Minute PE ({len(pe5)})")
    st.caption("PE Stocks में से — 5m RSI(14) ने 60 या 40 को नीचे की ओर cross किया")
    st.dataframe(pe5, use_container_width=True, hide_index=True)

st.caption("⚠️ सिर्फ screening tool है, trading advice नहीं। Entry से पहले chart खुद confirm करें।")
