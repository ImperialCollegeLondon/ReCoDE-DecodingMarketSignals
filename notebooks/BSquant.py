from typing import Optional
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import plotly.express as px


def load_data(
    file_path: Path,
    compression: Optional[str] = "gzip",
    ticker: Optional[str] = None,
    selected_start_date: Optional[pd.Timestamp] = None,
    selected_end_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Loads data from a compressed CSV file, filters it based on optional criteria,
    and returns a cleaned pandas DataFrame.

    The function reads a CSV file, optionally compressed, renames columns for consistency,
    filters the data by a specified ticker and date range, and returns the processed DataFrame.

    Args:
        file_path (Path): Path to the compressed CSV file.
        compression (Optional[str], optional): Compression type of the CSV file. Defaults to 'gzip'.
        ticker (Optional[str], optional): Specific ticker to filter on. If None, no ticker filtering is applied.
        selected_start_date (Optional[pd.Timestamp], optional): The start date for filtering the data.
            If None, no start date filtering is applied.
        selected_end_date (Optional[pd.Timestamp], optional): The end date for filtering the data.
            If None, no end date filtering is applied.

    Returns:
        pd.DataFrame: A DataFrame containing the processed and optionally filtered data.

    Examples:
        >>> load_data(Path('/path/to/data.csv'), ticker='AAPL', selected_start_date=pd.Timestamp('2020-01-01'),
        ... selected_end_date=pd.Timestamp('2020-12-31'))
        # This will return a DataFrame with data for AAPL between Jan 1, 2020, and Dec 31, 2020.
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
    df = pd.read_csv(file_path, usecols=rename_map.keys(), compression=compression)
    df.rename(columns=rename_map, inplace=True)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    if ticker is not None:
        df = df.copy().query("ticker == @ticker").reset_index(drop=True)

    # Apply date filtering
    if selected_start_date is not None:
        df = df[df["date"] >= selected_start_date]
    if selected_end_date is not None:
        df = df[df["date"] <= selected_end_date]

    return df.reset_index(drop=True)


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the given DataFrame by cleaning data, calculating intraday returns,
    and preparing for subsequent analysis.

    This function performs several steps:
    - Drops any rows with missing values.
    - Converts the volume column to integers.
    - Calculates the intraday return as (close - open) / open.
    - Determines the sign of the intraday return (positive, negative, or zero) as an integer.
    - Shifts the intraday returns and their signs to prepare for next-day return analysis.

    Parameters:
        df (pd.DataFrame): The DataFrame to process. Expects columns 'vol', 'open',
                           and 'close' to be present.

    Returns:
        pd.DataFrame: A DataFrame with the original data plus additional columns for
                      'intraday_return', 'sign_intraday_return', 'next_intraday_return',
                      and 'sign_next_day_return'. Rows with missing values after processing
                      are dropped to ensure completeness of the dataset.

    Note:
        The 'next_intraday_return' and 'sign_next_day_return' columns are intended for
        use in predictive models where the goal is to forecast the next day's market movement.
    """
    df = (
        df.dropna()
        .assign(
            vol=lambda x: x["vol"].astype(int),
            intraday_return=lambda x: (x["close"] - x["open"]) / x["open"],
            sign_intraday_return=lambda x: np.sign(x["intraday_return"]).astype(int),
        )
        .assign(
            # log_return=lambda x: np.log(x["close"] / x["close"].shift(1)),
            next_intraday_return=lambda x: x["intraday_return"].shift(-1),
            sign_next_day_return=lambda x: x["sign_intraday_return"]
            .shift(-1)
            .astype("Int64"),
        )
        .dropna()
        .reset_index(drop=True)
    )
    return df


def cs_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies candlestick patterns in financial data using the `talib` library.

    This function iterates over a list of candlestick pattern recognition functions provided by `talib` and applies each to the input DataFrame containing Open, High, Low, and Close prices (OHLC). It generates a DataFrame for each pattern indicating where the pattern occurs. The results are then consolidated into a single DataFrame, filtering out instances where no pattern was recognized.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing the financial data with columns for 'open', 'high', 'low', and 'close' prices. The DataFrame must also have a 'date' column used for indexing the results.

    Returns:
    - pd.DataFrame: A DataFrame indexed by 'candle' pattern names and 'date', containing rows where any candlestick pattern was recognized. The DataFrame includes the original OHLC data, a 'cs_pattern' column indicating the pattern strength or direction, and the 'candle' name for each recognized pattern.

    Note:
    - The 'cs_pattern' column values are provided by `talib` and represent the strength or direction of the recognized pattern, with non-zero values indicating the occurrence of a pattern.
    - The function requires `talib` to be installed and accessible in the environment.
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
    Analyses the performance of identified candlestick (CS) patterns by calculating
    precision and confidence intervals.

    This function takes a DataFrame containing candlestick pattern signals and next-day returns.
    It computes True Positives (TP) and False Positives (FP) for each pattern based on the alignment
    between the pattern signal and the actual next-day return. It then aggregates these values to
    calculate the precision (TP / (TP + FP)) for each candlestick pattern. Additionally, it computes
    the Wilson score confidence interval for the precision of each pattern.

    Parameters:
    - cs_signals_df (pd.DataFrame): A DataFrame indexed by candlestick pattern ('candle') and date,
      containing the signals (cs_pattern) and the next day's return direction
      (sign_next_day_return).

    Returns:
    - pd.DataFrame: A DataFrame indexed by candlestick pattern, containing performance metrics
      including the number of true positives (TP), false positives (FP), total instances of the
      pattern, precision, Wilson score confidence interval (ci_upper, ci_lower), and the
      center of the Wilson score interval (TP_wilson). The DataFrame is sorted by total instances
      and TP in descending order.

    The DataFrame columns include:
    - TP: Total true positives
    - FP: Total false positives
    - total_instances: Total instances of each candlestick pattern
    - precision: Precision of the pattern signal (TP / (TP + FP))
    - center: Center of the Wilson score interval
    - margin: Margin of the Wilson score interval
    - ci_upper: Upper bound of the Wilson score confidence interval
    - ci_lower: Lower bound of the Wilson score confidence interval
    - TP_wilson: Center of the Wilson score interval, representing adjusted true positives

    Note:
    - The function uses the norm.ppf function from scipy.stats to calculate the Wilson score interval,
      assuming a confidence level of 95%.
    """

    cs_signals_df_copy = cs_signals_df.copy()

    # Calculate TP and FP for each CS pattern
    cs_signals_df_copy["TP"] = (
        (
            np.sign(cs_signals_df_copy["cs_pattern"])
            == np.sign(cs_signals_df_copy["sign_next_day_return"])
        )
        & (cs_signals_df_copy["cs_pattern"] != 0)
    ).astype(int)

    cs_signals_df_copy["FP"] = (
        (
            np.sign(cs_signals_df_copy["cs_pattern"])
            != np.sign(cs_signals_df_copy["sign_next_day_return"])
        )
        & (cs_signals_df_copy["cs_pattern"] != 0)
    ).astype(int)

    # Group by the 'signal' level of the index (which represents the candlestick pattern) and sum TP and FP for each pattern
    performance_metrics = cs_signals_df_copy.groupby("candle").agg(
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
    """
    Generates a bar plot to visualize the performance of candlestick patterns based on a specified criterion.

    This function creates a bar chart for candlestick pattern performance metrics, with optional error bars
    representing 95% confidence intervals and performance zones. The performance zones are visual representations
    that categorize the patterns into red, yellow, and green zones based on their performance.

    Parameters:
    - df (pd.DataFrame): DataFrame containing candlestick pattern performance metrics, indexed by pattern names.
    - criterion (str): The performance metric to be visualized on the y-axis of the bar plot. Default is "total_instances".
    - plot_performance (bool): If True, adds error bars for 95% confidence intervals and performance zones to the plot.
                               Performance zones are marked in red (0.0 to 0.5), yellow (0.5 to 0.55), and green (0.55 to 1.0).
    - title_suffix (str): Additional text to append to the plot's title. Default is an empty string.

    Returns:
    - None: The function directly displays the plot and does not return any value.

    The function utilizes Plotly Express for plotting and updates the layout and traces with Plotly Graph Objects
    for customization, including error bars and shaded performance zones.
    """

    # Create the bar plot
    fig = px.bar(
        df,
        x=df.index,
        y=criterion,
        title=f"{criterion} for {len(df)} CS patterns<br>{title_suffix}",
        labels={"x": "CS Pattern", "y": criterion},
        width=800,
        height=800,
    )

    # Update layout
    fig.update_layout(
        font={"size": 18},
        autosize=False,
    )

    # Conditionally add 95% confidence intervals and performance zones
    if plot_performance:
        # Calculate the error relative to TP_wilson for upper and lower bounds
        error_y_upper = df["ci_upper"] - df[criterion]  # Distance to upper CI bound
        error_y_lower = df[criterion] - df["ci_lower"]  # Distance to lower CI bound

        # Adjust the bar plot to include error bars
        fig.update_traces(
            error_y={
                "type": "data",
                "symmetric": False,
                "array": error_y_upper,
                "arrayminus": error_y_lower,
            }
        )
        fig.add_hrect(y0=0.0, y1=0.5, line_width=0, fillcolor="red", opacity=0.4)
        fig.add_hrect(y0=0.5, y1=0.55, line_width=0, fillcolor="yellow", opacity=0.4)
        fig.add_hrect(y0=0.55, y1=1.0, line_width=0, fillcolor="green", opacity=0.4)

    fig.show()

    return None


def compute_trading_strategy_performance(
    df: pd.DataFrame, verbose: bool = False
) -> tuple:
    """
    Computes and returns the performance metrics of a trading strategy based on intraday returns,
    optionally printing the results.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'intraday_return' for the strategy's intraday returns.
    - verbose (bool): If True, prints the calculated metrics. Default is False.

    Returns:
    - tuple: Contains the annualised return, annualised standard deviation of returns, and the Sharpe ratio.
    """
    mean_intraday_return = df["intraday_return"].mean()
    std_intraday_return = df["intraday_return"].std()
    sharpe_ratio = (260 * mean_intraday_return) / (np.sqrt(260) * std_intraday_return)

    if verbose:
        print(f"Annualised strategy return [%]: {260 * mean_intraday_return:.4f}")
        print(
            f"Annualised strategy standard deviation of returns [%]: {np.sqrt(260) * std_intraday_return:.4f}"
        )
        print(f"Sharpe ratio of strategy: {sharpe_ratio:.4f}")

    return (
        260 * mean_intraday_return,
        np.sqrt(260) * std_intraday_return,
        sharpe_ratio,
    )
