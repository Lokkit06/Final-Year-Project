from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

DATA_DIR = Path(__file__).resolve().parent
DEFAULT_SPLIT_RATIO = 0.7

# Page config
st.set_page_config(page_title="Backtesting Dashboard", layout="wide")

# Styles
# The light background for the 'kpi' class is already defined here: background:#f7f9fc
st.markdown(
    """
<style>
.kpi {background:#f7f9fc;border-radius:12px;padding:16px;border:1px solid #e6ebf2}
.kpi .label{font-size:.9rem;color:#6b7280;margin-bottom:.25rem}
.kpi .value{font-size:1.6rem;font-weight:700}
.pos{color:#16a34a} .neg{color:#dc2626}
div[data-testid="stSidebar"] {min-width:300px; max-width:400px;}
</style>    
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(timeframe_code: str = "M5") -> pd.DataFrame:
    filename_map = {
        "M5": "XAUUSD.sml_M5_3Y.csv",
        "M15": "XAUUSD.sml_M15_3Y.csv",
        "M30": "XAUUSD.sml_M30_3Y.csv",
        "1H": "XAUUSD.sml_H1_3Y.csv",
    }
    filename = filename_map.get(timeframe_code)
    if not filename:
        st.error(f"Unsupported timeframe: {timeframe_code}")
        return pd.DataFrame()

    file_path = DATA_DIR / filename
    if not file_path.exists():
        st.error(f"Data file '{filename}' not found in {DATA_DIR}. Please add it and reload.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace("<", "").str.replace(">", "").str.strip()
        df["DATETIME"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], format="%Y.%m.%d %H:%M:%S")
        df.set_index("DATETIME", inplace=True)
        df.drop(columns=["DATE", "TIME"], inplace=True)
        return df
    except Exception as exc:
        st.error(f"Failed to load data for {timeframe_code}: {exc}")
        return pd.DataFrame()


def detect_structure(df: pd.DataFrame, swing: int = 3) -> pd.Series:
    if df.empty or any(col not in df for col in ("HIGH", "LOW")):
        return pd.Series(dtype=int)

    highs = df["HIGH"].to_numpy()
    lows = df["LOW"].to_numpy()
    length = len(df)
    structure = np.zeros(length, dtype=int)
    last_high_price = 0.0
    last_low_price = 0.0
    last_structure = 0

    for i in range(swing, length - swing):
        swing_high = all(highs[i] > highs[i - x] and highs[i] > highs[i + x] for x in range(1, swing + 1))
        swing_low = all(lows[i] < lows[i - x] and lows[i] < lows[i + x] for x in range(1, swing + 1))

        if swing_high:
            if highs[i] > last_high_price and last_high_price != 0:
                structure[i] = 1
            last_high_price = highs[i]

        if swing_low:
            if lows[i] < last_low_price and last_low_price != 0:
                structure[i] = -1
            last_low_price = lows[i]

        if structure[i] == 1 and last_structure == -1:
            structure[i] = 2
        elif structure[i] == -1 and last_structure == 1:
            structure[i] = -2

        if structure[i] != 0:
            last_structure = 1 if structure[i] > 0 else -1

    return pd.Series(structure, index=df.index, dtype=int)


def sma_strategy(df: pd.DataFrame, fast: int = 16, slow: int = 64) -> pd.DataFrame:
    out = df.copy()
    out[f"SMA_{fast}"] = out["CLOSE"].rolling(fast).mean()
    out[f"SMA_{slow}"] = out["CLOSE"].rolling(slow).mean()

    out["DIFF"] = out[f"SMA_{fast}"] - out[f"SMA_{slow}"]
    out["DIFF_PREV"] = out["DIFF"].shift(1)

    out["Signal"] = 0
    out.loc[(out["DIFF"] >= 0) & (out["DIFF_PREV"] < 0), "Signal"] = 1
    out.loc[(out["DIFF"] <= 0) & (out["DIFF_PREV"] > 0), "Signal"] = -1

    out["STRUCTURE"] = detect_structure(out)
    out.loc[~out["STRUCTURE"].isin([1, 2, -1, -2]), "Signal"] = 0

    current_position = 0
    position_series = []
    for idx in out.index:
        signal_val = out.at[idx, "Signal"]
        if signal_val != 0:
            current_position = signal_val
        position_series.append(current_position)

    out["Position"] = pd.Series(position_series, index=out.index, dtype=int)
    return out.dropna()


def macd_strategy(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    out["EMA_FAST"] = out["CLOSE"].ewm(span=fast, adjust=False).mean()
    out["EMA_SLOW"] = out["CLOSE"].ewm(span=slow, adjust=False).mean()
    out["MACD"] = out["EMA_FAST"] - out["EMA_SLOW"]
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    out["Signal"] = 0
    out.loc[
        (out["MACD"] > out["MACD_Signal"]) & (out["MACD"].shift(1) <= out["MACD_Signal"].shift(1)),
        "Signal",
    ] = 1
    out.loc[
        (out["MACD"] < out["MACD_Signal"]) & (out["MACD"].shift(1) >= out["MACD_Signal"].shift(1)),
        "Signal",
    ] = -1

    out["STRUCTURE"] = detect_structure(out)
    out.loc[~out["STRUCTURE"].isin([1, 2, -1, -2]), "Signal"] = 0

    current_position = 0
    position_series = []
    for idx in out.index:
        signal_val = out.at[idx, "Signal"]
        if signal_val != 0:
            current_position = signal_val
        position_series.append(current_position)

    out["Position"] = pd.Series(position_series, index=out.index, dtype=int)
    return out.dropna()


def rsi_strategy(df: pd.DataFrame, period: int = 14, oversold: int = 30, overbought: int = 70) -> pd.DataFrame:
    out = df.copy()
    delta = out["CLOSE"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    out["RSI"] = 100 - (100 / (1 + rs))

    out["Signal"] = 0
    out.loc[(out["RSI"] > oversold) & (out["RSI"].shift(1) <= oversold), "Signal"] = 1
    out.loc[(out["RSI"] < overbought) & (out["RSI"].shift(1) >= overbought), "Signal"] = -1

    out = out.dropna()
    out["STRUCTURE"] = detect_structure(out)
    out.loc[~out["STRUCTURE"].isin([1, 2, -1, -2]), "Signal"] = 0
    out = out.dropna()

    current_position = 0
    position_series = []
    for idx in out.index:
        signal_val = out.at[idx, "Signal"]
        if signal_val != 0:
            current_position = signal_val
        position_series.append(current_position)

    out["Position"] = pd.Series(position_series, index=out.index, dtype=int)
    return out


def bb_strategy(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    std_label = str(std).rstrip("0").rstrip(".") if isinstance(std, float) else str(std)
    bb_key = f"BB_{period}_{std_label}"
    mid_col = f"MID_{bb_key}"
    upper_col = f"UPPER_{bb_key}"
    lower_col = f"LOWER_{bb_key}"
    signal_col = f"IS_TRADE_{bb_key}"

    out["STRUCTURE"] = detect_structure(out)
    out[mid_col] = out["CLOSE"].rolling(window=period).mean()
    rolling_std = out["CLOSE"].rolling(window=period).std()
    out[upper_col] = out[mid_col] + std * rolling_std
    out[lower_col] = out[mid_col] - std * rolling_std

    out["BB_Middle"] = out[mid_col]
    out["BB_Upper"] = out[upper_col]
    out["BB_Lower"] = out[lower_col]

    out[signal_col] = 0
    out.loc[out["CLOSE"] <= out[lower_col], signal_col] = 1
    out.loc[out["CLOSE"] >= out[upper_col], signal_col] = -1
    out.loc[~out["STRUCTURE"].isin([1, 2, -1, -2]), signal_col] = 0

    out["Signal"] = out[signal_col]
    out.loc[out["Signal"] > 0, "Signal"] = 1
    out.loc[out["Signal"] < 0, "Signal"] = -1
    out = out.dropna()

    current_position = 0
    position_series = []
    for idx in out.index:
        signal_val = out.at[idx, "Signal"]
        if signal_val != 0:
            current_position = signal_val
        position_series.append(current_position)

    out["Position"] = pd.Series(position_series, index=out.index, dtype=int)
    return out


def run_strategy_with_split(strategy_fn, df: pd.DataFrame, split_ratio: float = DEFAULT_SPLIT_RATIO, *args, **kwargs) -> pd.DataFrame:
    if df.empty:
        return df
    split_idx = int(len(df) * split_ratio)
    chunks = [df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()]
    frames: list[pd.DataFrame] = []
    for chunk in chunks:
        if chunk.empty:
            continue
        result = strategy_fn(chunk, *args, **kwargs)
        if not result.empty:
            frames.append(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


def calc_metrics(
    df: pd.DataFrame, initial: float = 1000, debug: bool = False, debug_panel=None
) -> Tuple[Dict[str, float], pd.DataFrame]:
    def _empty_metrics() -> Dict[str, float]:
        return {
            "Total Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate (%)": 0.0,
            "Total PnL ($)": 0.0,
            "Total Win ($)": 0.0,
            "Total Loss ($)": 0.0,
            "Total Spread Cost ($)": 0.0,
            "Profit Factor": 0.0,
            "Expectancy ($)": 0.0,
            "Max Drawdown (%)": 0.0,
            "Sharpe Ratio": 0.0,
            "Portfolio": pd.Series(dtype=float),
            "Trade_Returns": pd.Series(dtype=float),
        }

    if df.empty or "Signal" not in df:
        return _empty_metrics(), pd.DataFrame()

    pip_size = 0.01
    take_profit = 1000
    stop_loss = -500
    max_loss_per_trade = initial * 0.02

    entries_idx = df.index[df["Signal"] != 0]
    if len(entries_idx) == 0:
        return _empty_metrics(), pd.DataFrame()

    df_trades = df.loc[entries_idx].copy()
    df_trades["NEXT_CLOSE"] = df_trades["CLOSE"].shift(-1)
    df_trades["DELTA_PIPS"] = ((df_trades["NEXT_CLOSE"] - df_trades["CLOSE"]) / pip_size) * df_trades["Signal"]
    
    df_trades["GAIN_RAW"] = df_trades["DELTA_PIPS"] * pip_size * df_trades["Signal"]
    if "SPREAD" in df_trades.columns:
        df_trades["SPREAD_COST"] = df_trades["SPREAD"] * pip_size
    else:
        df_trades["SPREAD_COST"] = 0
    df_trades["GAIN"] = df_trades["GAIN_RAW"] - df_trades["SPREAD_COST"]
    df_trades["GAIN"] = df_trades["GAIN"].clip(lower=stop_loss * pip_size, upper=take_profit * pip_size)
    df_trades["GAIN"] = df_trades["GAIN"].clip(lower=-max_loss_per_trade)
    df_trades = df_trades.dropna()

    if len(df_trades) == 0:
        return _empty_metrics(), df_trades

    total_trades = len(df_trades)
    win_trades = (df_trades["GAIN"] > 0).sum()
    loss_trades = (df_trades["GAIN"] < 0).sum()
    win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0.0

    total_profit = df_trades["GAIN"].sum()
    total_win = df_trades.loc[df_trades["GAIN"] > 0, "GAIN"].sum()
    total_loss = df_trades.loc[df_trades["GAIN"] < 0, "GAIN"].sum()
    total_spread_cost = df_trades["SPREAD_COST"].sum()

    avg_win = df_trades.loc[df_trades["GAIN"] > 0, "GAIN"].mean() if win_trades > 0 else 0.0
    avg_loss = df_trades.loc[df_trades["GAIN"] < 0, "GAIN"].mean() if loss_trades > 0 else 0.0
    loss_rate = 100 - win_rate
    expectancy = (win_rate / 100) * avg_win - (loss_rate / 100) * abs(avg_loss)

    gross_profit = total_win
    gross_loss = abs(total_loss)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    df_trades["CUM_GAIN"] = df_trades["GAIN"].cumsum()
    equity = initial + df_trades["CUM_GAIN"]
    rolling_max = equity.cummax()
    max_drawdown = ((equity - rolling_max) / initial * 100).min()

    sharpe_ratio = (
        df_trades["GAIN"].mean() / df_trades["GAIN"].std() * np.sqrt(252)
        if df_trades["GAIN"].std() != 0
        else np.nan
    )
    sharpe_ratio = 0.0 if np.isnan(sharpe_ratio) else float(sharpe_ratio)

    metrics = {
        "Total Trades": int(total_trades),
        "Wins": int(win_trades),
        "Losses": int(loss_trades),
        "Win Rate (%)": win_rate,
        "Total PnL ($)": float(total_profit),
        "Total Win ($)": float(total_win),
        "Total Loss ($)": float(total_loss),
        "Total Spread Cost ($)": float(total_spread_cost),
        "Profit Factor": float(profit_factor) if profit_factor != np.inf else np.inf,
        "Expectancy ($)": float(expectancy),
        "Max Drawdown (%)": float(max_drawdown),
        "Sharpe Ratio": sharpe_ratio,
        "Portfolio": equity,
        "Trade_Returns": df_trades["GAIN"] * 100,
    }
    return metrics, df_trades.reset_index()


def kpi(label: str, value: str, positive: bool | None = None):
    color = "" if positive is None else ("pos" if positive else "neg")
    st.markdown(
        f"<div class='kpi'><div class='label'>{label}</div><div class='value {color}'>{value}</div></div>",
        unsafe_allow_html=True,
    )


def price_chart(df: pd.DataFrame, strategy: str, timeframe_code: str = "M5"):
    timeframe_display_map = {
        "M5": "5 Minutes",
        "M15": "15 Minutes",
        "M30": "30 Minutes",
        "1H": "1 Hour",
    }
    timeframe_display = timeframe_display_map.get(timeframe_code, timeframe_code)

    rows = 2
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.15,
        subplot_titles=[f"Price ({timeframe_display})", "Volume"],
    )
    # --- START COLOR MODIFICATION FOR CANDLESTICK CHART ---
    # Use neutral blue tones for candles so Buy/Sell signals (green/red) stand out clearly
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["OPEN"],
            high=df["HIGH"],
            low=df["LOW"],
            close=df["CLOSE"],
            name=f"XAUUSD ({timeframe_display})",
            increasing_line_color="#2563eb",   # blue
            decreasing_line_color="#1d4ed8",   # darker blue
            increasing_fillcolor="#93c5fd",    # light blue fill
            decreasing_fillcolor="#60a5fa",    # medium blue fill
        ),
        row=1,
        col=1,
    )
    # --- END COLOR MODIFICATION FOR CANDLESTICK CHART ---
    
    if strategy.startswith("SMA"):
        ma_cols = [c for c in df.columns if c.startswith("SMA_") or c.startswith("MA_")]
        seen = set()
        # Use different, distinct colors for MAs
        ma_colors = ["#f97316", "#2563eb", "#8b5cf6"] 
        for i, col in enumerate(ma_cols):
            if col in seen:
                continue
            seen.add(col)
            # Assign distinct colors to MAs
            color = ma_colors[i % len(ma_colors)] 
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(width=2, color=color)), row=1, col=1)
            
    if strategy.startswith("MACD") and {"MACD", "MACD_Signal"} <= set(df.columns):
        # Use contrasting colors for MACD and Signal lines
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#2563eb")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#f97316")), row=2, col=1)
        
    rsi_shown = False
    if strategy.startswith("RSI") and "RSI" in df:
        # --- START COLOR MODIFICATION FOR RSI CHART ---
        fig.add_trace(
            go.Scatter(x=df.index, y=df["RSI"], name="RSI", mode="lines", line=dict(color="#2563eb", width=2)), # More visible blue
            row=2,
            col=1,
        )
        # --- END COLOR MODIFICATION FOR RSI CHART ---
        
        overbought = 70
        oversold = 30
        try:
            parts = [p.strip() for p in strategy.replace("RSI", "").split("/") if p.strip()]
            if len(parts) == 3:
                oversold = float(parts[1])
                overbought = float(parts[2])
        except Exception:
            pass
        for level, label in [(overbought, f"Overbought ({overbought:g})"), (oversold, f"Oversold ({oversold:g})")]:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=[level] * len(df),
                    name=label,
                    mode="lines",
                    line=dict(color="#a1a1aa", width=1), # Light grey lines for levels
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        fig.update_yaxes(range=[0, 100], row=2, col=1, title="")
        rsi_shown = True
        
    if strategy.startswith("BB"):
        # Use distinct colors for Bollinger Bands
        bb_colors = {"BB_Upper": "#FFC300", "BB_Middle": "#FFFFFF", "BB_Lower": "#10B981"}
        for col in ["BB_Upper", "BB_Middle", "BB_Lower"]:
            if col in df:
                fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col, line=dict(color=bb_colors.get(col))), row=1, col=1)
                
    if "Signal" in df:
        buys = df[df["Signal"] > 0]
        sells = df[df["Signal"] < 0]
        if not buys.empty:
            # --- START COLOR MODIFICATION FOR SIGNALS ---
            fig.add_trace(
                go.Scatter(
                    x=buys.index,
                    y=buys["CLOSE"],
                    mode="markers",
                    marker=dict(symbol="triangle-up", color="#16a34a", size=10), # Bright Green for Buy
                    name="Buy Signal",
                ),
                row=1,
                col=1,
            )
            if not sells.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sells.index,
                        y=sells["CLOSE"],
                        mode="markers",
                        marker=dict(symbol="triangle-down", color="#dc2626", size=10), # Bright Red for Sell
                        name="Sell Signal",
                    ),
                    row=1,
                    col=1,
                )
            # --- END COLOR MODIFICATION FOR SIGNALS ---
            
    if not rsi_shown and "VOL" in df:
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["VOL"],
                name="Volume",
                marker=dict(color="#93c5fd", line=dict(width=0)),  # no outline line
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(
        showgrid=True,
        zeroline=True,
        
        showticklabels=True,
        ticks="outside",
        tickfont=dict(color="#6b7280"),
        showline=False,
        row=2,
        col=1,
        title="",
    )

    # Set plot background to white/light and ensure the theme is light
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        template="plotly_white",
        font=dict(color="#FFFFFF"),
        title=dict(text="", font=dict(color="#FFFFFF")),
    )

    st.plotly_chart(fig, use_container_width=True)


def comparison_section(df: pd.DataFrame, initial_capital: float = 1000, timeframe_code: str = "M5"):
    timeframe_display_map = {
        "M5": "5 Minutes",
        "M15": "15 Minutes",
        "M30": "30 Minutes",
        "1H": "1 Hour",
    }
    timeframe_display = timeframe_display_map.get(timeframe_code, timeframe_code)
    st.subheader(f"Strategy Comparison ({timeframe_display})")
    strategies = {
        "SMA 8 / 32": run_strategy_with_split(sma_strategy, df, DEFAULT_SPLIT_RATIO, 8, 32),
        "SMA 16 / 64": run_strategy_with_split(sma_strategy, df, DEFAULT_SPLIT_RATIO, 16, 64),
        "SMA 20 / 80": run_strategy_with_split(sma_strategy, df, DEFAULT_SPLIT_RATIO, 20, 80),
        "MACD 12 / 26 / 9": run_strategy_with_split(macd_strategy, df, DEFAULT_SPLIT_RATIO, 12, 26, 9),
        "MACD 10 / 30 / 8": run_strategy_with_split(macd_strategy, df, DEFAULT_SPLIT_RATIO, 10, 30, 8),
        "MACD 20 / 50 / 10": run_strategy_with_split(macd_strategy, df, DEFAULT_SPLIT_RATIO, 20, 50, 10),
        "RSI 14 / 30 / 70": run_strategy_with_split(rsi_strategy, df, DEFAULT_SPLIT_RATIO, 14, 30, 70),
        "RSI 9 / 20 / 80": run_strategy_with_split(rsi_strategy, df, DEFAULT_SPLIT_RATIO, 9, 20, 80),
        "RSI 21 / 40 / 60": run_strategy_with_split(rsi_strategy, df, DEFAULT_SPLIT_RATIO, 21, 40, 60),
        "BB 20 / 2": run_strategy_with_split(bb_strategy, df, DEFAULT_SPLIT_RATIO, 20, 2.0),
        "BB 10 / 2": run_strategy_with_split(bb_strategy, df, DEFAULT_SPLIT_RATIO, 10, 2.0),
        "BB 20 / 1.5": run_strategy_with_split(bb_strategy, df, DEFAULT_SPLIT_RATIO, 20, 1.5),
    }
    comps = {}
    for name, sdf in strategies.items():
        m, _ = calc_metrics(sdf, initial_capital)
        if name.startswith("SMA"):
            params = name.replace("SMA ", "").replace(" / ", "/")
        elif name.startswith("MACD"):
            params = name.replace("MACD ", "").replace(" / ", "/")
        elif name.startswith("RSI"):
            params = name.replace("RSI ", "").replace(" / ", "/")
        elif name.startswith("BB"):
            params = name.replace("BB ", "").replace(" / ", "/")
        else:
            params = ""

        comps[name] = {
            "Parameters": params,
            "Total Trades": m["Total Trades"],
            "Total PnL ($)": round(m["Total PnL ($)"], 2),
            "Win Rate (%)": round(m["Win Rate (%)"], 2),
            "Profit Factor": round(m["Profit Factor"], 2) if m["Profit Factor"] != np.inf else np.inf,
            "Expectancy ($)": round(m["Expectancy ($)"], 2),
            "Sharpe": round(m["Sharpe Ratio"], 2),
            "Max Drawdown (%)": round(m["Max Drawdown (%)"], 2),
            "Wins": m.get("Wins", 0),
            "Losses": m.get("Losses", 0),
        }
    cmp_df = pd.DataFrame(comps).T
    cmp_df = cmp_df[
        [
            "Total Trades",
            "Wins",
            "Losses",
            "Total PnL ($)",
            "Win Rate (%)",
            "Profit Factor",
            "Expectancy ($)",
            "Max Drawdown (%)",
            "Sharpe",
        ]
    ].round(2)

    b1, b2 = st.columns(2)
    with b1:
        # --- START COLOR MODIFICATION FOR PNL CHART ---
        fig_pnl = px.bar(
            cmp_df.reset_index(),
            x="index",
            y="Total PnL ($)",
            color="Total PnL ($)",
            color_continuous_scale="Plasma", # Changed scale for better contrast
            title="Total PnL ($) by Strategy",
        )
        # --- END COLOR MODIFICATION FOR PNL CHART ---
        fig_pnl.update_layout(height=350, xaxis_title="Strategy", yaxis_title="PnL ($)", template="plotly_white")
        st.plotly_chart(fig_pnl, use_container_width=True)
    with b2:
        # --- START COLOR MODIFICATION FOR DRAWDOWN CHART ---
        fig_dd = px.bar(
            cmp_df.reset_index(),
            x="index",
            y="Max Drawdown (%)",
            color="Max Drawdown (%)",
            color_continuous_scale="Viridis", # Changed scale for better contrast
            title="Max Drawdown (%) by Strategy",
        )
        # --- END COLOR MODIFICATION FOR DRAWDOWN CHART ---
        fig_dd.update_layout(height=350, xaxis_title="Strategy", yaxis_title="Drawdown (%)", template="plotly_white")
        st.plotly_chart(fig_dd, use_container_width=True)

    b3, b4 = st.columns(2)
    with b3:
        # --- START COLOR MODIFICATION FOR SHARPE CHART ---
        fig_sharpe = px.bar(
            cmp_df.reset_index(),
            x="index",
            y="Sharpe",
            color="Sharpe",
            color_continuous_scale="Magma",  # Changed scale for better contrast
            title="Sharpe Ratio by Strategy",
        )
        # --- END COLOR MODIFICATION FOR SHARPE CHART ---
        fig_sharpe.update_layout(
            height=350,
            xaxis_title="Strategy",
            yaxis_title="Sharpe Ratio",
            template="plotly_white",
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)
    with b4:
        # --- START COLOR MODIFICATION FOR WIN RATE CHART ---
        fig_wr = px.bar(
            cmp_df.reset_index(),
            x="index",
            y="Win Rate (%)",
            color="Win Rate (%)",
            color_continuous_scale="Cividis",  # Changed scale for better contrast
            title="Win Rate (%) by Strategy",
        )
        # --- END COLOR MODIFICATION FOR WIN RATE CHART ---
        fig_wr.update_layout(
            height=350,
            xaxis_title="Strategy",
            yaxis_title="Win Rate (%)",
            template="plotly_white",
        )
        st.plotly_chart(fig_wr, use_container_width=True)

    # Expectancy chart
    fig_exp = px.bar(
        cmp_df.reset_index(),
        x="index",
        y="Expectancy ($)",
        color="Expectancy ($)",
        color_continuous_scale="Blues",
        title="Expectancy ($) by Strategy",
    )
    fig_exp.update_layout(
        height=350,
        xaxis_title="Strategy",
        yaxis_title="Expectancy ($)",
        template="plotly_white",
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.markdown("#### Win/Loss Distribution by Strategy")
    # --- START COLOR MODIFICATION FOR PIE CHARTS ---
    pie_colors = ["#16a34a", "#dc2626"]  # Wins (Green), Losses (Red)

    # Arrange pies so each strategy family (SMA / MACD / RSI / BB) is on its own row
    strategy_families = [
        ("SMA", "SMA Strategies"),
        ("MACD", "MACD Strategies"),
        ("RSI", "RSI Strategies"),
        ("BB", "Bollinger Band Strategies"),
    ]

    for row_idx, (prefix, family_label) in enumerate(strategy_families):
        family_strats = [idx for idx in cmp_df.index if str(idx).startswith(prefix)]
        if not family_strats:
            continue

        # Add extra vertical spacing before all rows except the first
        if row_idx > 0:
            st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

        st.markdown(f"**{family_label}**")
        row_cols = st.columns(len(family_strats))

        for col, strat_name in zip(row_cols, family_strats):
            with col:
                wins = cmp_df.loc[strat_name, "Wins"]
                losses = cmp_df.loc[strat_name, "Losses"]
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=["Wins", "Losses"],
                            values=[wins, losses],
                            hole=0.35,
                            marker=dict(colors=pie_colors),
                        )
                    ]
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent")
                fig_pie.update_layout(
                    height=260,
                    title=str(strat_name),
                    template="plotly_white",
                    margin=dict(l=0, r=60, t=40, b=0),  # right space for side legend
                    legend=dict(
                        orientation="v",
                        x=0.9,  # closer to the pie, still on the right
                        xanchor="left",
                        y=0.5,
                        yanchor="middle",
                        font=dict(size=11),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    ),
                )
                st.plotly_chart(fig_pie, use_container_width=True)
    # --- END COLOR MODIFICATION FOR PIE CHARTS ---

    # Add extra vertical spacing below the strategy pies before the performance table
    st.markdown("<div style='margin-top:40px;'></div>", unsafe_allow_html=True)

    if "Sharpe" in cmp_df.columns and not cmp_df["Sharpe"].isna().all():
        best_idx = cmp_df["Sharpe"].idxmax()
        cmp_df["Best Strategy"] = ""
        cmp_df.loc[best_idx, "Best Strategy"] = "🌟"

    # Rename Sharpe column for display
    display_df = cmp_df.rename(columns={"Sharpe": "Sharpe Ratio"})
    st.dataframe(display_df.round(2), use_container_width=True)


def trade_log(trades: pd.DataFrame, strategy_name: str):
    st.subheader(f"{strategy_name} Trade Log")
    if trades.empty:
        st.info("No trades for this configuration.")
        return
    entries = trades.copy()
    if "DATETIME" in entries.columns:
        entries = entries.rename(columns={"DATETIME": "Entry Time"})
    elif "index" in entries.columns:
        entries = entries.rename(columns={"index": "Entry Time"})

    entries["Direction"] = entries["Signal"]
    entries["Signal"] = entries["Direction"].apply(lambda x: "BUY" if x > 0 else "SELL")

    entries["Entry Price"] = entries["CLOSE"].round(2)
    entries["P/L ($)"] = entries["GAIN"].round(2)
    entries["Exit Price"] = (
        entries["Entry Price"] + entries["P/L ($)"] / entries["Direction"]
    ).round(2)
    entries["P/L (%)"] = (entries["P/L ($)"] / entries["Entry Price"] * 100).round(2)

    show_cols = [
        col
        for col in ["Entry Time", "Signal", "Entry Price", "Exit Price", "P/L ($)"]
        if col in entries.columns
    ]
    show = entries[show_cols]
    st.dataframe(show, use_container_width=True, height=400)
    st.download_button("Download Trades CSV", show.to_csv(index=False), file_name="trades.csv", mime="text/csv")


def main():
    st.sidebar.header("Settings")

    timeframe_options = ["5 Minutes", "15 Minutes", "30 Minutes", "1 Hour"]
    timeframe_display = st.sidebar.selectbox("Timeframe", timeframe_options, index=0)
    timeframe_map = {
        "5 Minutes": "M5",
        "15 Minutes": "M15",
        "30 Minutes": "M30",
        "1 Hour": "1H",
    }
    timeframe_code = timeframe_map.get(timeframe_display, "M5")

    df = load_data(timeframe_code)
    if df.empty:
        st.warning(f"No data available for {timeframe_display} timeframe. Please check if the CSV file exists.")
        return

    st.title(f"Backtesting Dashboard - {timeframe_display}")

    st.sidebar.header("Strategy Filter")
    strategy_options = {
        "SMA Crossover": "sma",
        "Moving Average Convergence Divergence": "macd",
        "Relative Strength Index": "rsi",
        "Bollinger Band": "bb",
    }
    strategy_label = st.sidebar.selectbox("Strategy", list(strategy_options.keys()), index=0)
    strategy_key = strategy_options[strategy_label]
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    start_date = st.sidebar.date_input("Start", value=min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End", value=max_date, min_value=min_date, max_value=max_date)
    st.sidebar.metric("Initial Capital ($)", "1000")

    st.sidebar.header("Performance")
    max_default = len(df)
    max_bars = int(
        st.sidebar.number_input(
            "Max bars to process",
            min_value=1,
            max_value=len(df),
            value=1000,
            step=1,  # allow any integer value 
            help="Adjust only if you want to limit rows for speed.",
        )
    )
    initial_capital = 1000

    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    data = df.loc[start_dt:end_dt]
    total_selected = len(data)
    if max_bars < total_selected:
        data = data.iloc[-max_bars:]
        st.caption(f"Showing the most recent {max_bars:,} bars out of {total_selected:,} selected to keep the app responsive.")

    if strategy_key == "sma":
        sma_variations = {
            "SMA 16 / 64": (16, 64),
            "SMA 8 / 32": (8, 32),
            "SMA 20 / 80": (20, 80),
        }
        selected = st.sidebar.selectbox("SMA Variation", list(sma_variations.keys()), index=0)
        fast, slow = sma_variations[selected]
        sdata = run_strategy_with_split(sma_strategy, data, DEFAULT_SPLIT_RATIO, fast, slow)
        strat_name = selected
    elif strategy_key == "macd":
        macd_variations = {
            "MACD 12 / 26 / 9": (12, 26, 9),
            "MACD 10 / 30 / 8": (10, 30, 8),
            "MACD 20 / 50 / 10": (20, 50, 10),
        }
        selected = st.sidebar.selectbox("MACD Variation", list(macd_variations.keys()), index=0)
        fast, slow, sig = macd_variations[selected]
        sdata = run_strategy_with_split(macd_strategy, data, DEFAULT_SPLIT_RATIO, fast, slow, sig)
        strat_name = selected
    elif strategy_key == "rsi":
        rsi_variations = {
            "RSI 14 / 30 / 70": (14, 30, 70),
            "RSI 9 / 20 / 80": (9, 20, 80),
            "RSI 21 / 40 / 60": (21, 40, 60),
        }
        selected = st.sidebar.selectbox("RSI Variation", list(rsi_variations.keys()), index=0)
        period, oversold, overbought = rsi_variations[selected]
        sdata = run_strategy_with_split(rsi_strategy, data, DEFAULT_SPLIT_RATIO, period, oversold, overbought)
        strat_name = selected
    else:
        bb_variations = {
            "BB 20 / 2": (20, 2.0),
            "BB 10 / 2": (10, 2.0),
            "BB 20 / 1.5": (20, 1.5),
        }
        selected = st.sidebar.selectbox("BB Variation", list(bb_variations.keys()), index=0)
        period, std = bb_variations[selected]
        sdata = run_strategy_with_split(bb_strategy, data, DEFAULT_SPLIT_RATIO, period, std)
        strat_name = selected

    m, trades = calc_metrics(sdata, initial_capital)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        kpi("Total Trade", f"{m['Total Trades']}", positive=True)
    with c2:
        kpi("Win Rate", f"{m['Win Rate (%)']:.2f}%", positive=m["Win Rate (%)"] >= 50)
    with c3:
        kpi("Profit Factor", f"{m['Profit Factor']:.2f}", positive=m["Profit Factor"] >= 1.75)
    with c4:
        kpi("Expectancy", f"${m['Expectancy ($)']:.2f}", positive=m["Expectancy ($)"] >= 0)
    with c5:
        # Max Drawdown is negative or zero. A less negative number is 'better'. Comparing to -20% as a threshold.
        kpi("Max Drawdown", f"{m['Max Drawdown (%)']:.2f}%", positive=m["Max Drawdown (%)"] > -20) 
    with c6:
        kpi("Sharpe Ratio", f"{m['Sharpe Ratio']:.2f}", positive=m["Sharpe Ratio"] >= 1)

    price_chart(sdata, strat_name, timeframe_code)
    comparison_section(data, initial_capital, timeframe_code)
    st.markdown("---")
    trade_log(trades, strat_name)


if __name__ == "__main__":
    main()