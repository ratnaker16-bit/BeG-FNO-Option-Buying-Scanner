import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="BeG@ F&O Option Buying Scanner",
    page_icon="📊",
    layout="wide"
)

st.title("BeG@ F&O Option Buying Scanner V1.0")
st.caption("Full NSE F&O Stock Universe | CE / PE Setup Scanner")

# ============================================================
# FULL NSE F&O INDIVIDUAL STOCK UNIVERSE
# NSE official list
# ============================================================

FNO_STOCKS = [
    "AARTIIND",
    "ABB",
    "ABBOTINDIA",
    "ACC",
    "ADANIENT",
    "ADANIPORTS",
    "ABCAPITAL",
    "ABFRL",
    "ALKEM",
    "AMBUJACEM",
    "APOLLOHOSP",
    "APOLLOTYRE",
    "ASHOKLEY",
    "ASIANPAINT",
    "ASTRAL",
    "ATUL",
    "AUBANK",
    "AUROPHARMA",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BALKRISIND",
    "BALRAMCHIN",
    "BANDHANBNK",
    "BANKBARODA",
    "BATAINDIA",
    "BERGEPAINT",
    "BEL",
    "BHARATFORG",
    "BHEL",
    "BPCL",
    "BHARTIARTL",
    "BIOCON",
    "BSOFT",
    "BOSCHLTD",
    "BRITANNIA",
    "CANFINHOME",
    "CANBK",
    "CHAMBLFERT",
    "CHOLAFIN",
    "CIPLA",
    "CUB",
    "COALINDIA",
    "COFORGE",
    "COLPAL",
    "CONCOR",
    "COROMANDEL",
    "CROMPTON",
    "CUMMINSIND",
    "DABUR",
    "DALBHARAT",
    "DEEPAKNTR",
    "DELTACORP",
    "DIVISLAB",
    "DIXON",
    "DLF",
    "LALPATHLAB",
    "DRREDDY",
    "EICHERMOT",
    "ESCORTS",
    "EXIDEIND",
    "GAIL",
    "GLENMARK",
    "GMRINFRA",
    "GODREJCP",
    "GODREJPROP",
    "GRANULES",
    "GRASIM",
    "GUJGASLTD",
    "GNFC",
    "HAVELLS",
    "HCLTECH",
    "HDFCAMC",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HAL",
    "HINDCOPPER",
    "HINDPETRO",
    "HINDUNILVR",
    "HDFC",
    "ICICIBANK",
    "ICICIGI",
    "ICICIPRULI",
    "IDFCFIRSTB",
    "IDFC",
    "IBULHSGFIN",
    "INDIAMART",
    "IEX",
    "IOC",
    "IRCTC",
    "IGL",
    "INDUSTOWER",
    "INDUSINDBK",
    "NAUKRI",
    "INFY",
    "INTELLECT",
    "INDIGO",
    "IPCALAB",
    "ITC",
    "JINDALSTEL",
    "JKCEMENT",
    "JSWSTEEL",
    "JUBLFOOD",
    "KOTAKBANK",
    "L&TFH",
    "LTTS",
    "LTIM",
    "LT",
    "LAURUSLABS",
    "LICHSGFIN",
    "LUPIN",
    "MGL",
    "M&MFIN",
    "M&M",
    "MANAPPURAM",
    "MARICO",
    "MARUTI",
    "MFSL",
    "METROPOLIS",
    "MOTHERSON",
    "MPHASIS",
    "MRF",
    "MCX",
    "MUTHOOTFIN",
    "NATIONALUM",
    "NAVINFLUOR",
    "NESTLEIND",
    "NMDC",
    "NTPC",
    "OBEROIRLTY",
    "ONGC",
    "OFSS",
    "PAGEIND",
    "PERSISTENT",
    "PETRONET",
    "PIIND",
    "PIDILITIND",
    "PEL",
    "POLYCAB",
    "PFC",
    "POWERGRID",
    "PNB",
    "PVRINOX",
    "RAIN",
    "RBLBANK",
    "RECLTD",
    "RELIANCE",
    "SBICARD",
    "SBILIFE",
    "SHREECEM",
    "SHRIRAMFIN",
    "SIEMENS",
    "SRF",
    "SBIN",
    "SAIL",
    "SUNPHARMA",
    "SUNTV",
    "SYNGENE",
    "TATACHEM",
    "TATACOMM",
    "TCS",
    "TATACONSUM",
    "TATAMOTORS",
    "TATAPOWER",
    "TATASTEEL",
    "TECHM",
    "FEDERALBNK",
    "INDIACEM",
    "INDHOTEL",
    "RAMCOCEM",
    "TITAN",
    "TORNTPHARM",
    "TRENT",
    "TVSMOTOR",
    "ULTRACEMCO",
    "UBL",
    "MCDOWELL-N",
    "UPL",
    "VEDL",
    "IDEA",
    "VOLTAS",
    "WHIRLPOOL",
    "WIPRO",
    "ZEEL",
    "ZYDUSLIFE"
]

# ============================================================
# SETTINGS
# ============================================================

st.sidebar.header("Scanner Settings")

scan_tf = st.sidebar.selectbox(
    "Scan Timeframe",
    ["5m", "15m"],
    index=0
)

rsi_len = st.sidebar.number_input(
    "RSI Length",
    min_value=2,
    max_value=50,
    value=9
)

di_len = st.sidebar.number_input(
    "DI Length",
    min_value=2,
    max_value=50,
    value=14
)

adx_smoothing = st.sidebar.number_input(
    "ADX Smoothing",
    min_value=2,
    max_value=50,
    value=14
)

min_adx = st.sidebar.number_input(
    "Minimum ADX",
    min_value=1.0,
    max_value=100.0,
    value=18.0,
    step=0.5
)

volume_multiplier = st.sidebar.number_input(
    "Volume Multiplier",
    min_value=0.1,
    max_value=10.0,
    value=1.30,
    step=0.05
)

ce_rsi = st.sidebar.number_input(
    "BUY CE RSI ≥",
    min_value=1.0,
    max_value=100.0,
    value=55.0,
    step=0.5
)

pe_rsi = st.sidebar.number_input(
    "BUY PE RSI ≤",
    min_value=1.0,
    max_value=100.0,
    value=45.0,
    step=0.5
)

use_breakout = st.sidebar.checkbox(
    "Previous Candle Breakout",
    value=True
)

# ============================================================
# SESSION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

SESSION_START = time(9, 20)
SESSION_END = time(14, 45)

now_ist = datetime.now(IST)
current_time = now_ist.time()

market_open = (
    current_time >= SESSION_START
    and current_time <= SESSION_END
)

# ============================================================
# INDICATOR FUNCTIONS
# ============================================================

def rma(series, length):
    return series.ewm(
        alpha=1 / length,
        adjust=False,
        min_periods=length
    ).mean()


def calculate_rsi(close, length=9):

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = rma(gain, length)
    avg_loss = rma(loss, length)

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_dmi_adx(df, di_len=14, adx_len=14):

    high = df["High"]
    low = df["Low"]
    close = df["Close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()

    tr = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = rma(tr, di_len)

    plus_di = (
        100 *
        rma(pd.Series(plus_dm, index=df.index), di_len)
        / atr
    )

    minus_di = (
        100 *
        rma(pd.Series(minus_dm, index=df.index), di_len)
        / atr
    )

    di_sum = plus_di + minus_di

    dx = (
        100 *
        (plus_di - minus_di).abs()
        / di_sum.replace(0, np.nan)
    )

    adx = rma(dx, adx_len)

    return plus_di, minus_di, adx


def calculate_vwap(df):

    typical_price = (
        df["High"] +
        df["Low"] +
        df["Close"]
    ) / 3

    volume = df["Volume"]

    temp = pd.DataFrame({
        "TPV": typical_price * volume,
        "VOL": volume
    }, index=df.index)

    # Session-wise daily VWAP
    local_date = pd.Series(
        df.index.date,
        index=df.index
    )

    cumulative_tpv = temp["TPV"].groupby(local_date).cumsum()
    cumulative_vol = temp["VOL"].groupby(local_date).cumsum()

    vwap = cumulative_tpv / cumulative_vol.replace(0, np.nan)

    return vwap


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if df is None or df.empty:
        return None

    df = df.copy()

    # Flatten Yahoo multi-index
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:
        if col not in df.columns:
            return None

    df = df[required].copy()

    df.dropna(subset=["Close"], inplace=True)

    # EMA
    df["EMA5"] = df["Close"].ewm(
        span=5,
        adjust=False
    ).mean()

    df["EMA9"] = df["Close"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA15"] = df["Close"].ewm(
        span=15,
        adjust=False
    ).mean()

    # RSI
    df["RSI"] = calculate_rsi(
        df["Close"],
        rsi_len
    )

    # DMI / ADX
    (
        df["PLUS_DI"],
        df["MINUS_DI"],
        df["ADX"]
    ) = calculate_dmi_adx(
        df,
        di_len,
        adx_smoothing
    )

    # VWAP
    df["VWAP"] = calculate_vwap(df)

    # Volume SMA
    df["VOL_SMA20"] = df["Volume"].rolling(20).mean()

    df["VOL_RATIO"] = (
        df["Volume"] /
        df["VOL_SMA20"].replace(0, np.nan)
    )

    # ADX rising
    df["ADX_RISING"] = df["ADX"] > df["ADX"].shift(1)

    return df


# ============================================================
# SIGNAL LOGIC
# ============================================================

def scan_stock(symbol, raw_df):

    df = prepare_data(raw_df)

    if df is None or len(df) < 50:
        return None

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Latest candle time
    timestamp = df.index[-1]

    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(
            tzinfo=ZoneInfo("Asia/Kolkata")
        )

    timestamp_ist = timestamp.astimezone(IST)

    candle_time = timestamp_ist.time()

    # Session filter
    in_session = (
        candle_time >= SESSION_START
        and candle_time <= SESSION_END
    )

    if not in_session:
        return None

    # ========================================================
    # BUY CE
    # ========================================================

    bullish_trend = (
        last["EMA5"] >
        last["EMA9"] >
        last["EMA15"]
        and
        last["Close"] >
        last["VWAP"]
    )

    bullish_di = (
        last["PLUS_DI"] >
        last["MINUS_DI"]
    )

    bullish_breakout = (
        last["Close"] >
        prev["High"]
    )

    ce_breakout_ok = (
        bullish_breakout
        if use_breakout
        else True
    )

    buy_ce = (
        bullish_trend
        and last["RSI"] >= ce_rsi
        and bullish_di
        and last["ADX"] >= min_adx
        and last["ADX_RISING"]
        and last["VOL_RATIO"] >= volume_multiplier
        and ce_breakout_ok
    )

    # ========================================================
    # BUY PE
    # ========================================================

    bearish_trend = (
        last["EMA5"] <
        last["EMA9"] <
        last["EMA15"]
        and
        last["Close"] <
        last["VWAP"]
    )

    bearish_di = (
        last["MINUS_DI"] >
        last["PLUS_DI"]
    )

    bearish_breakout = (
        last["Close"] <
        prev["Low"]
    )

    pe_breakout_ok = (
        bearish_breakout
        if use_breakout
        else True
    )

    buy_pe = (
        bearish_trend
        and last["RSI"] <= pe_rsi
        and bearish_di
        and last["ADX"] >= min_adx
        and last["ADX_RISING"]
        and last["VOL_RATIO"] >= volume_multiplier
        and pe_breakout_ok
    )

    # ========================================================
    # RESULT
    # ========================================================

    if buy_ce:

        return {
            "STOCK": symbol,
            "OPTION": "BUY CE",
            "ENTRY": float(last["Close"]),
            "RSI": float(last["RSI"]),
            "ADX": float(last["ADX"]),
            "PLUS DI": float(last["PLUS_DI"]),
            "MINUS DI": float(last["MINUS_DI"]),
            "VOL RATIO": float(last["VOL_RATIO"]),
            "VWAP": float(last["VWAP"]),
            "TIME": timestamp_ist.strftime("%H:%M")
        }

    if buy_pe:

        return {
            "STOCK": symbol,
            "OPTION": "BUY PE",
            "ENTRY": float(last["Close"]),
            "RSI": float(last["RSI"]),
            "ADX": float(last["ADX"]),
            "PLUS DI": float(last["PLUS_DI"]),
            "MINUS DI": float(last["MINUS_DI"]),
            "VOL RATIO": float(last["VOL_RATIO"]),
            "VWAP": float(last["VWAP"]),
            "TIME": timestamp_ist.strftime("%H:%M")
        }

    return None


# ============================================================
# DOWNLOAD DATA
# ============================================================

@st.cache_data(
    ttl=30,
    show_spinner=False
)
def download_all_data(symbols, interval):

    tickers = [
        symbol + ".NS"
        for symbol in symbols
    ]

    try:

        data = yf.download(
            tickers=tickers,
            period="5d",
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False
        )

        return data

    except Exception:
        return pd.DataFrame()


# ============================================================
# SCANNER
# ============================================================

def run_scanner():

    results = []
    failed = []

    data = download_all_data(
        tuple(FNO_STOCKS),
        scan_tf
    )

    if data is None or data.empty:
        return results, len(FNO_STOCKS)

    # Yahoo returns MultiIndex:
    # ticker -> OHLCV

    for symbol in FNO_STOCKS:

        yahoo_symbol = symbol + ".NS"

        try:

            if isinstance(data.columns, pd.MultiIndex):

                if yahoo_symbol not in data.columns.get_level_values(0):
                    failed.append(symbol)
                    continue

                df = data[yahoo_symbol].copy()

            else:

                df = data.copy()

            if df.empty:
                failed.append(symbol)
                continue

            result = scan_stock(
                symbol,
                df
            )

            if result is not None:
                results.append(result)

        except Exception:
            failed.append(symbol)

    return results, len(failed)


# ============================================================
# MARKET STATUS
# ============================================================

if market_open:

    st.success(
        f"🟢 Market Scanner ACTIVE | "
        f"IST {now_ist.strftime('%H:%M:%S')}"
    )

else:

    st.warning(
        f"🔴 Scanner session: 09:20–14:45 | "
        f"Current IST {now_ist.strftime('%H:%M:%S')}"
    )


# ============================================================
# SCAN BUTTON
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "F&O Universe",
        len(FNO_STOCKS)
    )

with col2:
    st.metric(
        "Scan TF",
        scan_tf
    )

with col3:
    st.metric(
        "Trading Session",
        "09:20–14:45"
    )


if st.button(
    "🔄 SCAN FULL F&O",
    type="primary",
    use_container_width=True
):

    with st.spinner(
        f"Scanning {len(FNO_STOCKS)} F&O stocks..."
    ):

        results, failed_count = run_scanner()

    # ========================================================
    # RESULTS
    # ========================================================

    if results:

        result_df = pd.DataFrame(results)

        # CE first, PE second
        result_df["SORT"] = result_df["OPTION"].map({
            "BUY CE": 1,
            "BUY PE": 2
        })

        result_df = (
            result_df
            .sort_values(
                ["SORT", "ADX"],
                ascending=[True, False]
            )
            .drop(columns=["SORT"])
        )

        st.success(
            f"🎯 {len(result_df)} Active Setup Found"
        )

        # Main dashboard
        st.subheader("📊 Active F&O Setups")

        display_df = result_df[
            [
                "STOCK",
                "OPTION",
                "ENTRY",
                "RSI",
                "ADX",
                "PLUS DI",
                "MINUS DI",
                "VOL RATIO",
                "VWAP",
                "TIME"
            ]
        ].copy()

        display_df["ENTRY"] = display_df["ENTRY"].round(2)
        display_df["RSI"] = display_df["RSI"].round(1)
        display_df["ADX"] = display_df["ADX"].round(1)
        display_df["PLUS DI"] = display_df["PLUS DI"].round(1)
        display_df["MINUS DI"] = display_df["MINUS DI"].round(1)
        display_df["VOL RATIO"] = display_df["VOL RATIO"].round(2)
        display_df["VWAP"] = display_df["VWAP"].round(2)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # CE / PE SEPARATE
        # ====================================================

        st.subheader("🟢 BUY CE")

        ce_df = display_df[
            display_df["OPTION"] == "BUY CE"
        ]

        if not ce_df.empty:
            st.dataframe(
                ce_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No BUY CE setup")

        st.subheader("🔴 BUY PE")

        pe_df = display_df[
            display_df["OPTION"] == "BUY PE"
        ]

        if not pe_df.empty:
            st.dataframe(
                pe_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No BUY PE setup")

    else:

        st.info(
            "No active CE/PE setup found in the current scan."
        )

    st.caption(
        f"Stocks scanned: {len(FNO_STOCKS)} | "
        f"Data unavailable/skipped: {failed_count}"
    )


# ============================================================
# LOGIC DISPLAY
# ============================================================

with st.expander("📌 Scanner Logic"):

    st.markdown("""
### BUY CE

- EMA5 > EMA9 > EMA15
- Close > VWAP
- RSI ≥ 55
- +DI > -DI
- ADX ≥ 18
- ADX rising
- Volume ≥ 1.30 × Volume SMA20
- Close > Previous Candle High
- Time: 09:20–14:45

### BUY PE

- EMA5 < EMA9 < EMA15
- Close < VWAP
- RSI ≤ 45
- -DI > +DI
- ADX ≥ 18
- ADX rising
- Volume ≥ 1.30 × Volume SMA20
- Close < Previous Candle Low
- Time: 09:20–14:45
""")

st.caption(
    "Data source: Yahoo Finance. Intraday data may be delayed and should "
    "not be treated as exchange-grade real-time market data."
)
