import streamlit as st
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# import functions from your engine file
from risk_engine import (
    fetch_prices,
    compute_log_returns,
    annualize_mu_cov,
    historical_portfolio_losses,
    simulate_correlated_gbm_losses,
    stress_modify_params,
    var_cvar,
    var_forecast_backtest_historical,
    kupiec_pof_test,
)

st.set_page_config(page_title="Monte Carlo Portfolio Risk Engine", layout="wide")

st.title("Monte Carlo Portfolio Risk Engine")
st.caption("VaR / CVaR • Correlated GBM • Student-t fat tails • Stress testing • Backtesting (Kupiec)")

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Portfolio Inputs")

default_tickers = "SPY,TLT,GLD,QQQ"
tickers_str = st.sidebar.text_input("Tickers (comma-separated)", default_tickers)
tickers = [t.strip().upper() for t in tickers_str.split(",") if t.strip()]

st.sidebar.subheader("Weights")
st.sidebar.caption("Must match tickers. We'll normalize automatically.")

weights_dict = {}
for t in tickers:
    weights_dict[t] = st.sidebar.number_input(f"Weight: {t}", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", "2019-01-01")

portfolio_value = st.sidebar.number_input("Portfolio Value ($)", min_value=1_000.0, value=1_000_000.0, step=50_000.0)

alpha = st.sidebar.selectbox("VaR confidence (alpha)", [0.95, 0.99], index=0)

horizon_days = st.sidebar.selectbox("Horizon (trading days)", [1, 10, 21], index=1)

n_sims = st.sidebar.selectbox("Monte Carlo sims", [50_000, 100_000, 200_000], index=2)

dist_choice = st.sidebar.selectbox("Innovations", ["normal", "t"], index=0)
t_df = st.sidebar.slider("Student-t df (if t)", min_value=3, max_value=30, value=5, step=1)

run = st.sidebar.button("Run Risk Engine")

# -----------------------------
# Helpers
# -----------------------------
def normalize_weights(w: dict) -> dict:
    s = sum(w.values())
    if s <= 0:
        raise ValueError("Weights sum to 0. Set at least one weight > 0.")
    return {k: v / s for k, v in w.items()}

def plot_hist(losses: np.ndarray, title: str, alpha: float):
    v, cv = var_cvar(losses, alpha)
    fig, ax = plt.subplots()
    ax.hist(losses, bins=120)
    ax.axvline(v, linestyle="--")
    ax.axvline(cv, linestyle="--")
    ax.set_title(f"{title}\nVaR({int(alpha*100)}%)={v:,.0f} | CVaR={cv:,.0f}")
    ax.set_xlabel("Portfolio Loss")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def plot_backtest(bt: pd.DataFrame, title: str):
    fig, ax = plt.subplots()
    ax.plot(bt.index, bt["realized_loss"].values, label="Realized Loss")
    ax.plot(bt.index, bt["VaR"].values, linestyle="--", label="VaR Forecast")
    breaches = bt[bt["breach"]]
    ax.scatter(breaches.index, breaches["realized_loss"].values, label="Breaches")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Loss ($)")
    ax.legend()
    fig.tight_layout()
    return fig

# -----------------------------
# Run engine
# -----------------------------
if run:
    try:
        weights = normalize_weights(weights_dict)
        w_vec = np.array([weights[t] for t in tickers], dtype=float)

        with st.spinner("Fetching price data..."):
            prices = fetch_prices(tickers, start=start_date)

        # use only tickers that successfully fetched
        tickers_used = list(prices.columns)
        if len(tickers_used) == 0:
            st.error("No tickers returned any data. Try different tickers or start date.")
            st.stop()

        # rebuild weights for only returned tickers
        weights_used = {t: weights.get(t, 0.0) for t in tickers_used}
        weights_used = normalize_weights(weights_used)
        w_vec = np.array([weights_used[t] for t in tickers_used], dtype=float)

        log_rets = compute_log_returns(prices)
        mu_annual, cov_annual = annualize_mu_cov(log_rets)

        # Historical
        hist_losses = historical_portfolio_losses(
            log_rets, w_vec, horizon_days=horizon_days, portfolio_value=portfolio_value
        )
        hv, hcv = var_cvar(hist_losses, alpha)

        # Monte Carlo (dist toggle)
        mc_losses = simulate_correlated_gbm_losses(
            mu_annual, cov_annual, w_vec,
            horizon_days=horizon_days,
            n_sims=int(n_sims),
            portfolio_value=portfolio_value,
            dist=dist_choice,
            t_df=int(t_df),
            seed=123
        )
        mv, mcv = var_cvar(mc_losses, alpha)

        # Stress
        mu_s, cov_s = stress_modify_params(mu_annual, cov_annual)
        stress_losses = simulate_correlated_gbm_losses(
            mu_s, cov_s, w_vec,
            horizon_days=horizon_days,
            n_sims=150_000,
            portfolio_value=portfolio_value,
            dist="normal",
            seed=7
        )
        sv, scv = var_cvar(stress_losses, alpha)

        # Backtest + Kupiec
        bt_hist = var_forecast_backtest_historical(
            log_rets, w_vec, alpha=alpha, horizon_days=horizon_days, window=252, portfolio_value=portfolio_value
        )
        n = len(bt_hist)
        x = int(bt_hist["breach"].sum())
        lr, pval = kupiec_pof_test(n, x, alpha)

        # -----------------------------
        # Display
        # -----------------------------
        st.subheader("Inputs Used")
        st.write("Tickers used:", tickers_used)
        st.write("Weights used:", weights_used)

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Historical VaR ({int(alpha*100)}%)", f"${hv:,.0f}")
        c1.metric("Historical CVaR", f"${hcv:,.0f}")

        c2.metric(f"MC ({dist_choice}) VaR ({int(alpha*100)}%)", f"${mv:,.0f}")
        c2.metric(f"MC ({dist_choice}) CVaR", f"${mcv:,.0f}")

        c3.metric(f"Stress VaR ({int(alpha*100)}%)", f"${sv:,.0f}")
        c3.metric("Stress CVaR", f"${scv:,.0f}")

        st.subheader("VaR Backtest (Historical) + Kupiec Test")
        st.write(f"Forecasts: **{n}** | Breaches: **{x}** | Breach rate: **{x/n:.3%}** | Expected: **{1-alpha:.3%}**")
        st.write(f"Kupiec POF LR: **{lr:.3f}** | p-value: **{pval:.4f}**")

        st.divider()

        left, right = st.columns(2)
        with left:
            st.pyplot(plot_hist(hist_losses, "Historical Loss Distribution", alpha))
            st.pyplot(plot_hist(mc_losses, f"Monte Carlo ({dist_choice}) Loss Distribution", alpha))
        with right:
            st.pyplot(plot_hist(stress_losses, "Stress Scenario Loss Distribution", alpha))
            st.pyplot(plot_backtest(bt_hist, "Backtest: Historical VaR vs Realized Loss"))

        st.success("Done.")

    except Exception as e:
        st.error(f"Error: {e}")
