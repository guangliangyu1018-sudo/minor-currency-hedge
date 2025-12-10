# hedge_utils.py
import pandas as pd
import numpy as np
from typing import Optional
from pandas.tseries.offsets import MonthEnd
from urllib.request import urlopen
import csv


# --------------------------------------------------------------
# 1. DATA LOADERS
#    - US 2Y/5Y: FRED (daily -> monthly average)
#    - AUD 2Y/5Y: RBA F2.1 (monthly, manual CSV parsing)
# --------------------------------------------------------------

def _to_monthly_average(df: pd.DataFrame, date_col: str, value_col: str) -> pd.DataFrame:
    """
    Daily / irregular series -> monthly average.
    Group explicitly by (year, month) to avoid freq quirks.
    Returns: Date (month-end), Yield
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month

    grouped = df.groupby(["Year", "Month"], as_index=False)[value_col].mean()

    grouped["Date"] = pd.to_datetime(
        dict(year=grouped["Year"], month=grouped["Month"], day=1)
    ) + MonthEnd(0)

    grouped = grouped[["Date", value_col]].rename(columns={value_col: "Yield"})
    return grouped


def get_us_yield_monthly(tenor: str) -> pd.DataFrame:
    """
    US Treasury 2Y / 5Y from FRED CSV → monthly average.
      - 2Y -> DGS2
      - 5Y -> DGS5

    Returns DataFrame(Date, Yield) where Yield is in percent.
    """
    tenor = tenor.upper()
    series_map = {"2Y": "DGS2", "5Y": "DGS5"}
    if tenor not in series_map:
        raise ValueError("Unsupported tenor for US: use '2Y' or '5Y'.")

    series_id = series_map[tenor]
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

    df = pd.read_csv(url)

    if df.shape[1] < 2:
        raise ValueError(
            f"Unexpected FRED CSV shape for {series_id}. Columns: {list(df.columns)}"
        )

    date_col = df.columns[0]   # 'observation_date'
    value_col = df.columns[1]  # 'DGS2' / 'DGS5'

    return _to_monthly_average(df, date_col, value_col)


def get_aud_yield_monthly(tenor: str) -> pd.DataFrame:
    """
    Australian Government 2Y / 5Y from RBA F2.1 CSV → monthly series.

    We avoid pandas.read_csv on the RBA URL to bypass C-engine tokenizing issues.

    RBA f2.1-data.csv structure (relevant part):

      - Column 0: dates (e.g. '31-May-13', '30-Jun-13', ...)
      - Column 1: 2-year yield   (FCMYGBAG2)
      - Column 3: 5-year yield   (FCMYGBAG5)

    Parsing logic:

      - Download CSV text via urllib
      - Use Python's csv.reader to iterate lines
      - Collect (date_string, value_string) pairs from column 0 and the tenor column
      - Use pandas.to_datetime(..., errors='coerce', dayfirst=True) for dates
      - Use pandas.to_numeric(..., errors='coerce') for yields
      - Keep only rows with valid Date & Yield
      - Force Date to month-end; deduplicate by last value per month

    Returns DataFrame(Date, Yield) with one row per month, Yield in percent.
    """
    tenor = tenor.upper()
    col_map = {"2Y": 1, "5Y": 3}  # B=2Y, D=5Y in RBA table
    if tenor not in col_map:
        raise ValueError("Unsupported tenor for AUD: use '2Y' or '5Y'.")

    value_idx = col_map[tenor]
    url = "https://www.rba.gov.au/statistics/tables/csv/f2.1-data.csv"

    # Fetch raw CSV text
    resp = urlopen(url)
    text = resp.read().decode("latin1")

    date_strs = []
    value_strs = []

    for row in csv.reader(text.splitlines()):
        if not row:
            continue
        # Some header lines may have fewer columns; skip those
        if len(row) <= value_idx:
            continue

        date_str = row[0].strip()
        val_str = row[value_idx].strip()

        # Collect everything; we'll let pandas decide what's a real date/number
        date_strs.append(date_str)
        value_strs.append(val_str)

    df = pd.DataFrame({"DateStr": date_strs, "YieldStr": value_strs})

    # Flexible parsing: non-date rows become NaT
    df["Date"] = pd.to_datetime(df["DateStr"], errors="coerce", dayfirst=True)
    df["Yield"] = pd.to_numeric(df["YieldStr"], errors="coerce")

    # Keep only valid date & yield
    df = df.dropna(subset=["Date", "Yield"]).sort_values("Date")

    if df.empty:
        raise ValueError("No valid AUD yield rows parsed from RBA F2.1 CSV (after cleaning).")

    # Force month-end index
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp("M")

    # If duplicates for same month, take the last
    df = (
        df.groupby("Date", as_index=False)["Yield"]
        .last()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return df[["Date", "Yield"]]


# --------------------------------------------------------------
# 2. MATH: exponential weights, decayed beta hedge, scenarios
# --------------------------------------------------------------

def exponential_weights(n: int, half_life_periods: Optional[float]) -> np.ndarray:
    """
    Exponential decay weights for time series of length n.
    Each index is one period (month).

    half_life_periods = #months after which weight = 50% of most recent.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if half_life_periods is None or half_life_periods <= 0:
        return np.ones(n) / float(n)

    lam = np.log(2.0) / float(half_life_periods)
    idx = np.arange(n)
    # 0 ... n-1, with n-1 = most recent
    w = np.exp(-lam * (n - 1 - idx))
    w /= w.sum()
    return w


def compute_cross_ccy_hedge_from_series(
    base_yield: pd.Series,
    hedge_yield: pd.Series,
    half_life_periods: float = 12.0,  # months
    dv01_base_per_1m: float = 1.0,
    dv01_hedge_per_1m: float = 1.0,
    base_notional_mn: float = 1.0,
):
    """
    Compute cross-currency hedge ratio using monthly yield series.

    Steps:
      - align on dates (inner join)
      - compute monthly Δyields (in percentage points, since yields are in %)
      - compute time-decayed covariance / variances
      - corr = cov / (σ_base * σ_hedge)
      - beta (Δy_base on Δy_hedge) = cov / var_hedge = corr * σ_base / σ_hedge
      - hedge ratio (notional per 1MM base) = beta * (DV01_base / DV01_hedge)
      - hedge notional = base_notional * hedge_ratio
    """
    df_yields = pd.concat(
        [base_yield.rename("base_yield"), hedge_yield.rename("hedge_yield")],
        axis=1,
        join="inner",
    ).dropna()

    if df_yields.empty or len(df_yields) < 5:
        raise ValueError("Not enough overlapping yield data between base and hedge.")

    df_yields = df_yields.sort_index()

    # Monthly changes (percentage points)
    df_changes = df_yields.diff().dropna()
    base_dx = df_changes["base_yield"].values.astype(float)
    hedge_dy = df_changes["hedge_yield"].values.astype(float)

    n = len(df_changes)
    w = exponential_weights(n, half_life_periods)

    # Weighted means
    mx = np.sum(w * base_dx)
    my = np.sum(w * hedge_dy)

    # Weighted variances and covariance
    var_x = np.sum(w * (base_dx - mx) ** 2)
    var_y = np.sum(w * (hedge_dy - my) ** 2)
    cov_xy = np.sum(w * (base_dx - mx) * (hedge_dy - my))

    if var_x <= 0 or var_y <= 0:
        corr = float("nan")
        std_x = float("nan")
        std_y = float("nan")
        beta = float("nan")
    else:
        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)
        corr = cov_xy / (std_x * std_y)
        # Regression of Δy_base on Δy_hedge
        beta = cov_xy / var_y  # = corr * std_x / std_y

    # DV01 scaling
    if dv01_hedge_per_1m == 0:
        dv01_ratio = float("nan")
        hedge_ratio = float("nan")
        hedge_notional_mn = float("nan")
    else:
        dv01_ratio = dv01_base_per_1m / dv01_hedge_per_1m
        # Hedge ratio per 1MM base: beta * DV01_base / DV01_hedge
        hedge_ratio = beta * dv01_ratio
        hedge_notional_mn = base_notional_mn * hedge_ratio

    return {
        "corr": corr,
        "std_base": std_x,
        "std_hedge": std_y,
        "beta": beta,
        "dv01_ratio": dv01_ratio,
        "hedge_ratio": hedge_ratio,
        "hedge_notional_mn": hedge_notional_mn,
        "yields": df_yields,
        "changes": df_changes,
    }


def build_corr_scenario_table(
    base_corr: float,
    hedge_ratio_per_corr_unit: float,
    base_notional_mn: float,
    step: float = 0.05,
    n_steps_each_side: int = 3,
) -> pd.DataFrame:
    """
    Scenario table: shock correlation and recompute hedge ratio & notional.

    hedge_ratio_per_corr_unit = hedge_ratio / base_corr
    (i.e. hedge ratio you would get if corr = 1,
     assuming vol ratio and DV01 ratio are constant).

    For each scenario corr_scen:
      hedge_ratio_scen = corr_scen * hedge_ratio_per_corr_unit
    """
    rows = []
    base_corr_rounded = round(base_corr, 4)

    for k in range(-n_steps_each_side, n_steps_each_side + 1):
        scen_corr = base_corr + k * step
        scen_corr = max(min(scen_corr, 1.0), -1.0)
        scen_corr_round = round(scen_corr, 4)

        hedge_ratio = scen_corr * hedge_ratio_per_corr_unit
        hedge_notional_mn = base_notional_mn * hedge_ratio

        is_base = (scen_corr_round == base_corr_rounded)

        rows.append(
            {
                "Corr": scen_corr_round,
                "Hedge ratio (notional per 1MM base)": round(hedge_ratio, 4),
                "Hedge notional (MM)": round(hedge_notional_mn, 4),
                "Base": "Yes" if is_base else "",
            }
        )

    return pd.DataFrame(rows)
