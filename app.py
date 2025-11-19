# app.py
import streamlit as st
import pandas as pd
import numpy as np

from hedge_utils import (
    get_us_yield_monthly,
    get_aud_yield_monthly,
    compute_cross_ccy_hedge_from_series,
    build_corr_scenario_table,
)

st.set_page_config(page_title="US–AUD Fixed Income Hedge", layout="wide")

st.title("Cross-Currency Fixed Income Hedge – US vs AUD (2Y / 5Y)")

st.markdown(
    """
This app uses **free official sources**:

- **US Treasury** yields from FRED (2Y / 5Y, daily → monthly average)  
- **Australian Government** yields from RBA table F2.1 (2Y / 5Y, monthly)

It then:

- aligns both series over a **date range you choose**
- computes **monthly changes in yield (Δyield)**
- applies **time-decayed regression / correlation**
- converts to a **beta-style hedge ratio** with DV01 scaling
- shows a **correlation scenario table** and **charts**
"""
)

# --------------------------------------------------------------
# Sidebar controls
# --------------------------------------------------------------
st.sidebar.header("Model parameters")

base_ccy = st.sidebar.selectbox(
    "Base currency (exposure you hedge / replicate)",
    ["AUD", "USD"],
    index=0,
)

tenor = st.sidebar.selectbox(
    "Curve tenor",
    ["2Y", "5Y"],
    index=0,
)

# Load full US & AUD series for this tenor
try:
    df_us_all = get_us_yield_monthly(tenor)
    df_aud_all = get_aud_yield_monthly(tenor)
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Overlapping date range (actual data only, no interpolation)
overlap_start = max(df_us_all["Date"].min(), df_aud_all["Date"].min())
overlap_end = min(df_us_all["Date"].max(), df_aud_all["Date"].max())

default_start = max(overlap_start, overlap_end - pd.DateOffset(years=5))

start_date = st.sidebar.date_input(
    "Start date",
    value=default_start.date(),
    min_value=overlap_start.date(),
    max_value=overlap_end.date(),
)

end_date = st.sidebar.date_input(
    "End date",
    value=overlap_end.date(),
    min_value=overlap_start.date(),
    max_value=overlap_end.date(),
)

if start_date > end_date:
    st.sidebar.error("Start date must be before end date.")

half_life_months = st.sidebar.slider(
    "Correlation half-life (months)",
    min_value=3,
    max_value=60,
    value=12,
    step=3,
)

st.sidebar.caption(
    "Half-life = number of months after which an observation's weight\n"
    "falls to 50% of the most recent monthly move in the correlation."
)

base_notional_mn = st.sidebar.number_input(
    "Base IP notional (MM)",
    min_value=0.1,
    max_value=10000.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("### DV01 per 1MM notional")

dv01_base_per_1m = st.sidebar.number_input(
    "Base DV01 (per 1MM)",
    min_value=0.0001,
    max_value=100000.0,
    value=1.0,
    step=0.1,
)

dv01_hedge_per_1m = st.sidebar.number_input(
    "Hedge DV01 (per 1MM)",
    min_value=0.0001,
    max_value=100000.0,
    value=1.0,
    step=0.1,
)

st.sidebar.markdown("### Correlation scenario")

corr_step = st.sidebar.slider(
    "Correlation shock step",
    min_value=0.01,
    max_value=0.5,
    value=0.05,
    step=0.01,
)

corr_n_steps = st.sidebar.slider(
    "Steps on each side of base corr",
    min_value=1,
    max_value=5,
    value=3,
)

run_btn = st.sidebar.button("Run analysis")

# --------------------------------------------------------------
# Main logic
# --------------------------------------------------------------
if not run_btn:
    st.info("Set parameters in the sidebar, then click **Run analysis**.")
else:
    if start_date > end_date:
        st.error("Start date must be before end date.")
    else:
        with st.spinner("Aligning data and computing hedge ratio..."):
            start_ts = pd.to_datetime(start_date)
            end_ts = pd.to_datetime(end_date)

            df_us = df_us_all[
                (df_us_all["Date"] >= start_ts) & (df_us_all["Date"] <= end_ts)
            ].copy()
            df_aud = df_aud_all[
                (df_aud_all["Date"] >= start_ts) & (df_aud_all["Date"] <= end_ts)
            ].copy()

            if df_us.empty or df_aud.empty:
                st.error("No data in selected date range for this tenor.")
                st.stop()

            us_series = df_us.set_index("Date")["Yield"]
            aud_series = df_aud.set_index("Date")["Yield"]

            if base_ccy == "AUD":
                base_series = aud_series
                hedge_series = us_series
            else:
                base_series = us_series
                hedge_series = aud_series

            hedge_ccy = "USD" if base_ccy == "AUD" else "AUD"

            try:
                res = compute_cross_ccy_hedge_from_series(
                    base_yield=base_series,
                    hedge_yield=hedge_series,
                    half_life_periods=half_life_months,
                    dv01_base_per_1m=dv01_base_per_1m,
                    dv01_hedge_per_1m=dv01_hedge_per_1m,
                    base_notional_mn=base_notional_mn,
                )
            except Exception as e:
                st.error(f"Error in data / calculation: {e}")
            else:
                corr = res["corr"]
                dv01_ratio = res["dv01_ratio"]
                hedge_ratio = res["hedge_ratio"]
                hedge_notional_mn = res["hedge_notional_mn"]
                beta = res["beta"]
                std_base = res["std_base"]
                std_hedge = res["std_hedge"]
                df_yields = res["yields"]
                df_changes = res["changes"]

                # hedge_ratio_per_corr_unit = (σ_base / σ_hedge) * DV01_ratio
                if std_hedge is None or np.isnan(std_hedge) or std_hedge <= 0:
                    hedge_ratio_per_corr_unit = float("nan")
                else:
                    vol_ratio = std_base / std_hedge
                    hedge_ratio_per_corr_unit = vol_ratio * dv01_ratio

                st.subheader("Results summary")

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Decayed corr (Δmonthly yield)", f"{corr:.4f}")
                col2.metric("Beta (Δbase vs Δhedge)", f"{beta:.4f}")
                col3.metric(
                    "Hedge ratio (notional per 1MM base)",
                    f"{hedge_ratio:.4f}",
                )
                col4.metric(
                    f"Hedge notional in {hedge_ccy} (MM)",
                    f"{hedge_notional_mn:.4f}",
                    help=(
                        f"If base position is {base_notional_mn:.2f}MM {base_ccy} {tenor}, "
                        f"you need {hedge_notional_mn:.4f}MM {hedge_ccy} {tenor} "
                        "to hedge under this regression & DV01 assumption."
                    ),
                )

                st.markdown(
                    f"""
**Base leg**: {base_ccy} {tenor}  
**Hedge leg**: {hedge_ccy} {tenor}  

Overlapping window after data alignment:  
**{df_yields.index.min().date()} → {df_yields.index.max().date()}**  

Correlation & beta are computed on **monthly Δyield (in percentage points)**  
with half-life **{half_life_months} months**.
"""
                )

                # -------------------------
                # Download aligned data
                # -------------------------
                st.subheader("Aligned monthly yields (intersection of months)")

                df_yields_out = df_yields.copy().reset_index().rename(
                    columns={"index": "Date"}
                )

                csv_data = df_yields_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download aligned monthly yields (CSV)",
                    data=csv_data,
                    file_name=f"US_AUD_{tenor}_{base_ccy}_aligned_{df_yields.index.min().date()}_{df_yields.index.max().date()}.csv",
                    mime="text/csv",
                )

                st.dataframe(df_yields_out, use_container_width=True)

                # -------------------------
                # Scenario table
                # -------------------------
                st.subheader("Correlation scenario table (hedge adjustment)")

                if np.isnan(corr) or np.isnan(hedge_ratio_per_corr_unit):
                    st.warning(
                        "Cannot build correlation scenarios because correlation or vol ratio is undefined."
                    )
                else:
                    scen_df = build_corr_scenario_table(
                        base_corr=corr,
                        hedge_ratio_per_corr_unit=hedge_ratio_per_corr_unit,
                        base_notional_mn=base_notional_mn,
                        step=corr_step,
                        n_steps_each_side=corr_n_steps,
                    )
                    cols = ["Base"] + [c for c in scen_df.columns if c != "Base"]
                    st.dataframe(scen_df[cols], use_container_width=True)

                # -------------------------
                # Charts – with nicer labels
                # -------------------------
                base_label = f"Base yield ({tenor} {base_ccy})"
                hedge_label = f"Hedge yield ({tenor} {hedge_ccy})"
                base_delta_label = f"Δ Base yield ({tenor} {base_ccy})"
                hedge_delta_label = f"Δ Hedge yield ({tenor} {hedge_ccy})"

                st.subheader("Monthly yield history (aligned)")
                df_yields_plot = df_yields.rename(
                    columns={
                        "base_yield": base_label,
                        "hedge_yield": hedge_label,
                    }
                )
                st.line_chart(df_yields_plot, use_container_width=True)

                st.subheader("Monthly Δyield (changes)")
                df_changes_plot = df_changes.rename(
                    columns={
                        "base_yield": base_delta_label,
                        "hedge_yield": hedge_delta_label,
                    }
                )
                st.line_chart(df_changes_plot, use_container_width=True)

                # -------------------------
                # Latest month snapshot
                # -------------------------
                st.subheader("Latest month snapshot")

                latest_yield_date = df_yields.index.max()
                latest_yields = df_yields.loc[latest_yield_date]

                col_a, col_b, col_c = st.columns(3)
                col_a.write(f"**Latest month:** {latest_yield_date.date()}")

                if not df_changes.empty:
                    latest_change_date = df_changes.index.max()
                    latest_changes = df_changes.loc[latest_change_date]

                    col_b.metric(
                        base_label,
                        f"{latest_yields['base_yield']:.3f}%",
                        help="Base yield in the latest available month.",
                    )
                    col_c.metric(
                        hedge_label,
                        f"{latest_yields['hedge_yield']:.3f}%",
                        help="Hedge yield in the latest available month.",
                    )

                    st.write(
                        f"Latest monthly Δyields (from {latest_change_date.date()} vs previous month):"
                    )
                    c1, c2 = st.columns(2)

                    # Δyields are in percentage points; 1 bp = 0.01 in these units
                    base_bp = latest_changes["base_yield"] * 100.0
                    hedge_bp = latest_changes["hedge_yield"] * 100.0

                    c1.metric(
                        base_delta_label,
                        f"{base_bp:.1f} bp",
                    )
                    c2.metric(
                        hedge_delta_label,
                        f"{hedge_bp:.1f} bp",
                    )
                else:
                    col_b.metric(
                        base_label,
                        f"{latest_yields['base_yield']:.3f}%",
                    )
                    col_c.metric(
                        hedge_label,
                        f"{latest_yields['hedge_yield']:.3f}%",
                    )
                    st.caption("Not enough data to compute monthly Δyields.")

                with st.expander("Raw Δyield data"):
                    df_changes_out = df_changes.copy().reset_index().rename(
                        columns={"index": "Date"}
                    )
                    st.dataframe(df_changes_out, use_container_width=True)

                st.caption(
                    "AUD yields: RBA F2.1 (column B=2Y, D=5Y, one row per month). "
                    "US yields: FRED DGS2/DGS5 (daily → monthly average). "
                    "Aligned data uses only the months where both series have observations."
                )
