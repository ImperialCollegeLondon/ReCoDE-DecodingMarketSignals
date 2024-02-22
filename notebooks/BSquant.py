# pyproject.toml (package name, package version, dependency; can upload library onto PyPI)
# make library installable locally.
from typing import List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.express as px


def cs_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """

    :param df:
    :return:
    """

    import talib

    candle_names = talib.get_function_groups()["Pattern Recognition"]

    list_of_results_df = []

    for candle in candle_names:
        tds = df.copy()
        tds["cs_pattern"] = getattr(talib, candle)(
            df["open"], df["high"], df["low"], df["close"]
        )
        tds["candle"] = candle
        list_of_results_df.append(tds.set_index(["candle", "date"]))

    result_df = pd.concat(list_of_results_df)
    cs_signals_df = result_df[result_df["cs_pattern"] != 0]

    return cs_signals_df


def cs_performance(cs_signals_df: pd.DataFrame) -> pd.DataFrame:
    """


    :param cs_signals_df:
    :return:
    """

    cs_signals_df = cs_signals_df.copy()

    # Calculate TP, FP, TN, FN for each CS pattern
    cs_signals_df["TP"] = (
        (
            np.sign(cs_signals_df["cs_pattern"])
            == np.sign(cs_signals_df["sign_next_day_return"])
        )
        & (cs_signals_df["cs_pattern"] != 0)
    ).astype(int)

    cs_signals_df["FP"] = (
        (
            np.sign(cs_signals_df["cs_pattern"])
            != np.sign(cs_signals_df["sign_next_day_return"])
        )
        & (cs_signals_df["cs_pattern"] != 0)
    ).astype(int)

    # Group by the 'signal' level of the index (which represents the candlestick pattern) and sum TP and FP for each pattern
    performance_metrics = cs_signals_df.groupby("candle").agg(
        {
            "TP": "sum",
            "FP": "sum",
        }
    )
    performance_metrics["total_instances"] = (
        performance_metrics["TP"] + performance_metrics["FP"]
    )

    # Calculate the performance metrics for each candlestick pattern
    performance_metrics["precision"] = performance_metrics["TP"] / (
        performance_metrics["TP"] + performance_metrics["FP"]
    )

    # Replace any NaN values with 0
    performance_metrics.fillna(0, inplace=True)

    # Calculate the Wilson score interval for each row

    z = norm.ppf(0.975)

    for index, row in performance_metrics.iterrows():
        p_hat = row["precision"]  # The observed proportion (precision)
        n = row["total_instances"]  # The total instances (TP + FP)
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = (
            z * np.sqrt((p_hat * (1 - p_hat) / n) + z**2 / (4 * n**2))
        ) / denominator
        performance_metrics.at[index, "center"] = center
        performance_metrics.at[index, "margin"] = margin

    performance_metrics["ci_upper"] = (
        performance_metrics["center"] + performance_metrics["margin"]
    )
    performance_metrics["ci_lower"] = (
        performance_metrics["center"] - performance_metrics["margin"]
    )
    performance_metrics["TP_wilson"] = performance_metrics["center"]

    # Sort the performance DataFrame by 'TP' in descending order
    performance_metrics = performance_metrics.sort_values(
        by=["total_instances", "TP"], ascending=False
    )

    return performance_metrics


def plot_cs_performance(
    df: pd.DataFrame,
    criterion: str = "total_instances",
    plot_performance: bool = False,
    title_suffix: str = "",
) -> None:
    fig = px.bar(
        df,
        x=df.index,
        y=criterion,
        barmode="group",
    )

    fig.update_layout(
        font={"size": 18},
        legend={
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "right",
            "x": 0.99,
            "font": {"size": 20},
        },
        title=f"{criterion} for {len(df)} CS patterns<br>{title_suffix}",
        xaxis_title="CS Pattern",
        yaxis_title=criterion,
        autosize=False,
        width=800,
        height=800,
    )

    if plot_performance:
        fig.add_hrect(y0=0.0, y1=0.5, line_width=0, fillcolor="red", opacity=0.4)
        fig.add_hrect(y0=0.5, y1=0.55, line_width=0, fillcolor="yellow", opacity=0.4)
        fig.add_hrect(y0=0.55, y1=1.0, line_width=0, fillcolor="green", opacity=0.4)

    fig.show()

    return None


# def plot_cs_performance(
#     df: pd.DataFrame, criterion: str = "total_instances", plot_performance: bool = False
# ) -> None:
#     fig = go.Figure()
#
#     # Add the main bar chart
#     fig.add_trace(
#         go.Bar(
#             x=df.index,
#             y=df[criterion],
#             name=criterion,
#             error_y={
#                 "type": "data",
#                 "symmetric": False,
#                 "array": df["ci_upper"] - df[criterion],
#                 "arrayminus": df[criterion] - df["ci_lower"],
#             },
#         )
#     )
#
#     fig.update_layout(
#         font={"size": 18},
#         legend={
#             "yanchor": "top",
#             "y": 0.99,
#             "xanchor": "right",
#             "x": 0.99,
#             "font": {"size": 20},
#         },
#         title=f"{criterion} for {len(df)} CS patterns across the data set",
#         xaxis_title="CS Pattern",
#         yaxis_title=criterion,
#         autosize=False,
#         width=2000,
#         height=1000,
#     )
#
#     # Add performance zones if plot_performance is True
#     if plot_performance:
#         fig.add_hrect(y0=0.0, y1=0.5, line_width=0, fillcolor="red", opacity=0.2)
#         fig.add_hrect(y0=0.5, y1=0.55, line_width=0, fillcolor="yellow", opacity=0.2)
#         fig.add_hrect(y0=0.55, y1=1.0, line_width=0, fillcolor="green", opacity=0.2)
#
#     fig.show()
#
#     return None
#
#
# plot_cs_performance(
#     df=performance_metrics, criterion="TP_wilson", plot_performance=True
# )


######## pipeline pattern


def load_and_process_data(
    file_path: str,
    usecols: List[str],
    compression: Optional[str] = "gzip",
    ticker: Optional[str] = None,
    selected_start_date: Optional[pd.Timestamp] = None,
    selected_end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Loads data from a compressed CSV, processes it, and returns a cleaned DataFrame.

    Parameters:
    - file_path: Path to the compressed CSV file.
    - usecols: List of columns to read from the CSV.
    - compression: Optional; Compression type of the CSV file. Defaults to 'gzip'.
    - selected_start_date: Optional; the start date for filtering the data.
    - selected_end_date: Optional; the end date for filtering the data.

    Returns:
    - A pandas DataFrame with processed data.
    """

    rename_map = {
        "Ticker": "ticker",
        "DlyCalDt": "date",
        "DlyPrc": "prc",
        "DlyOpen": "open",
        "DlyHigh": "high",
        "DlyLow": "low",
        "DlyClose": "close",
        "DlyVol": "vol",
        "DlyPrcVol": "price_vol",
    }

    # Load and initially process the data
    df = pd.read_csv(file_path, usecols=usecols, compression=compression)

    filtered_rename_map = {k: v for k, v in rename_map.items() if k in usecols}
    df.rename(columns=filtered_rename_map, inplace=True)

    if ticker is not None:
        df = df.copy().query(" ticker == @ticker ")

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    # Apply date filtering only if both start and end dates are provided
    if selected_start_date is not None and selected_end_date is not None:
        df = df.query("@selected_start_date <= date <= @selected_end_date")

    # Continue processing the data
    df = (
        df.dropna()  # Consider if you want to fillna instead of dropping
        #           .rename(columns=filtered_rename_map)  # Assuming 'filtered_rename_map' is defined elsewhere
        .assign(
            vol=lambda x: x["vol"].astype(int),
            intraday_return=lambda x: (x["close"] - x["open"]) / x["open"],
            sign_intraday_return=lambda x: np.sign(x["intraday_return"]).astype(int),
        )
        .assign(
            log_return=lambda x: np.log(x["close"] / x["close"].shift(1)),
            next_intraday_return=lambda x: x["intraday_return"].shift(-1),
            sign_next_day_return=lambda x: x["sign_intraday_return"]
            .shift(-1)
            .astype("Int64"),
        )
        .dropna()
        .reset_index(drop=True)
    )

    return df
