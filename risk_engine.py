import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import yfinance as yf
from pandas_datareader import data as pdr

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from scipy.stats import chi2
from scipy.stats import t as student_t


# ----------------------------
# Config / Types
# ----------------------------

@dataclass
class RiskResults:
    var: float
    cvar: float
    loss_distribution: np.ndarray


# ----------------------------
# Data
# ----------------------------

def fetch_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    end: Optional[str] = None
) -> pd.DataFrame:
    """
    Try Yahoo (yfinance) first; if it fails, fallback to Stooq via pandas-datareader.
    Returns adjusted close prices (auto_adjust=True).
    """
    # --- Attempt Yahoo via yfinance ---
    try:
        df = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )["Close"]

        if isinstance(df, pd.Series):
            df = df.to_frame()

        df = df.dropna(how="all").dropna(axis=1)

        if not df.empty and df.shape[1] > 0:
            return df
    except Exception as e:
        print(f"Yahoo download failed, falling back to Stooq. Reason: {e}")

    # --- Fallback: Stooq (often reliable) ---
    frames = []
    for t in tickers:
        sym = t.lower()
        try:
            s = pdr.DataReader(sym, "stooq", start, end)
            s = s.sort_index()
            frames.append(s["Close"].rename(t))
        except Exception as e:
            print(f"Stooq failed for {t}: {e}")

    if not frames:
        raise RuntimeError("No price data returned from Yahoo or Stooq. Check internet or tickers.")

    out = pd.concat(frames, axis=1).dropna(how="all").dropna(axis=1)
    if out.empty:
        raise RuntimeError("Price data fetched but empty after cleaning.")
    return out


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def annualize_mu_cov(log_rets: pd.DataFrame, trading_days: int = 252) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate annualized drift (mu) and covariance (Sigma) from daily log returns.
    """
    mu_daily = log_rets.mean().values
    cov_daily = log_rets.cov().values
    mu_annual = mu_daily * trading_days
    cov_annual = cov_daily * trading_days
    return mu_annual, cov_annual


# ----------------------------
# Risk metrics
# ----------------------------

def var_cvar(losses: np.ndarray, alpha: float = 0.95) -> Tuple[float, float]:
    if losses is None or len(losses) == 0:
        raise ValueError("Loss distribution is empty. Upstream data/simulation produced no losses.")
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return float(var), float(cvar)


def kupiec_pof_test(n: int, x: int, alpha: float) -> tuple[float, float]:
    """
    Kupiec Proportion of Failures (POF) test for VaR backtesting.
    n = number of forecasts
    x = number of VaR breaches
    alpha = VaR confidence level (e.g., 0.95 => expected breach prob p=0.05)
    Returns: (LR_stat, p_value)
    """
    p = 1.0 - alpha  # expected breach probability
    if n <= 0:
        raise ValueError("n must be > 0")
    if x < 0 or x > n:
        raise ValueError("x must be in [0, n]")

    eps = 1e-12
    phat = np.clip(x / n, eps, 1 - eps)
    p = np.clip(p, eps, 1 - eps)

    lr = -2.0 * (
        (n - x) * np.log((1 - p) / (1 - phat))
        + x * np.log(p / phat)
    )
    p_value = 1.0 - chi2.cdf(lr, df=1)
    return float(lr), float(p_value)


# ----------------------------
# Backtesting (Historical VaR)
# ----------------------------

def var_forecast_backtest_historical(
    log_rets: pd.DataFrame,
    weights: np.ndarray,
    alpha: float = 0.95,
    horizon_days: int = 10,
    window: int = 252,
    portfolio_value: float = 1_000_000,
) -> pd.DataFrame:
    """
    Rolling Historical VaR backtest.

    For each date t, use previous `window` days to estimate VaR of `horizon_days`,
    then compare to realized horizon PnL from t..t+horizon_days.
    Returns DataFrame with VaR, realized loss, breach.
    """
    if len(log_rets) < window + horizon_days + 5:
        raise ValueError("Not enough data for the requested window/horizon.")

    r = log_rets.values
    dates = log_rets.index

    rows = []
    for t in range(window, len(log_rets) - horizon_days):
        est_slice = r[t - window:t, :]
        est_df = pd.DataFrame(est_slice, columns=log_rets.columns)

        # distribution of horizon log-returns within the window
        est_h = est_df.rolling(horizon_days).sum().dropna().values
        port_log = est_h @ weights
        port_simple = np.expm1(port_log)
        est_losses = -portfolio_value * port_simple

        var_t = np.quantile(est_losses, alpha)

        realized_log = r[t:t + horizon_days, :]
        realized_port_log = realized_log.sum(axis=0) @ weights
        realized_loss = -portfolio_value * np.expm1(realized_port_log)

        rows.append((dates[t], var_t, realized_loss, realized_loss > var_t))

    out = pd.DataFrame(rows, columns=["date", "VaR", "realized_loss", "breach"]).set_index("date")
    return out


# ----------------------------
# Historical simulation
# ----------------------------

def historical_portfolio_losses(
    log_rets: pd.DataFrame,
    weights: np.ndarray,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000
) -> np.ndarray:
    """
    Uses realized returns to generate portfolio PnL distribution.
    For horizon_days>1, uses rolling sums of log returns.
    """
    if horizon_days == 1:
        r = log_rets.values
    else:
        r = log_rets.rolling(horizon_days).sum().dropna().values

    port_log_r = r @ weights
    port_simple_r = np.expm1(port_log_r)
    pnl = portfolio_value * port_simple_r
    losses = -pnl
    return losses


# ----------------------------
# Monte Carlo (correlated GBM)
# ----------------------------

def simulate_correlated_gbm_losses(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    weights: np.ndarray,
    horizon_days: int = 10,
    n_sims: int = 100_000,
    portfolio_value: float = 1_000_000,
    trading_days: int = 252,
    seed: int = 42,
    dist: str = "normal",  # "normal" or "t"
    t_df: int = 5
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_assets = len(weights)
    dt = horizon_days / trading_days

    drift = (mu_annual - 0.5 * np.diag(cov_annual)) * dt
    cov_h = cov_annual * dt
    L = np.linalg.cholesky(cov_h + 1e-12 * np.eye(n_assets))

    if dist == "normal":
        Z = rng.standard_normal(size=(n_sims, n_assets))
    elif dist == "t":
        if t_df <= 2:
            raise ValueError("t_df must be > 2 for finite variance.")
        Z = student_t.rvs(df=t_df, size=(n_sims, n_assets), random_state=rng)
        Z = Z * np.sqrt((t_df - 2) / t_df)  # scale to unit variance
    else:
        raise ValueError("dist must be 'normal' or 't'")

    log_r = drift + (Z @ L.T)

    port_log_r = log_r @ weights
    port_simple_r = np.expm1(port_log_r)
    losses = -portfolio_value * port_simple_r
    return losses


# ----------------------------
# Stress testing
# ----------------------------

def stress_modify_params(
    mu_annual: np.ndarray,
    cov_annual: np.ndarray,
    shock_mu_bps: float = -200.0,
    vol_mult: float = 1.5,
    corr_push: float = 0.15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple stress:
      - shift annual mu downward by shock_mu_bps (basis points)
      - multiply vol (cov) by vol_mult^2
      - increase off-diagonal correlations slightly (corr_push)
    """
    mu_stress = mu_annual + (shock_mu_bps / 10_000.0)

    vols = np.sqrt(np.diag(cov_annual))
    corr = cov_annual / np.outer(vols, vols)

    corr2 = corr.copy()
    for i in range(corr2.shape[0]):
        for j in range(corr2.shape[1]):
            if i != j:
                corr2[i, j] = np.clip(corr2[i, j] + corr_push, -0.95, 0.95)

    cov_stress = corr2 * np.outer(vols, vols)
    cov_stress = cov_stress * (vol_mult ** 2)

    return mu_stress, cov_stress


# ----------------------------
# Visualization helpers
# ----------------------------

def plot_loss_hist(losses: np.ndarray, title: str, alpha: float = 0.95):
    v, cv = var_cvar(losses, alpha)
    plt.figure()
    plt.hist(losses, bins=120)
    plt.axvline(v, linestyle="--")
    plt.axvline(cv, linestyle="--")
    plt.title(f"{title}\nVaR({int(alpha*100)}%)={v:,.0f} | CVaR={cv:,.0f}")
    plt.xlabel("Portfolio Loss")
    plt.ylabel("Frequency")
    plt.tight_layout()


def plot_sensitivity(
    base_mu: np.ndarray,
    base_cov: np.ndarray,
    weights: np.ndarray,
    horizon_days: int,
    portfolio_value: float,
    alpha: float = 0.95
):
    vol_scales = np.array([0.75, 1.0, 1.25, 1.5, 2.0])
    vars_, cvars_ = [], []

    for s in vol_scales:
        cov_s = base_cov * (s ** 2)
        losses = simulate_correlated_gbm_losses(
            base_mu, cov_s, weights,
            horizon_days=horizon_days,
            n_sims=50_000,
            portfolio_value=portfolio_value,
            seed=123,
            dist="normal"
        )
        v, cv = var_cvar(losses, alpha)
        vars_.append(v)
        cvars_.append(cv)

    plt.figure()
    plt.plot(vol_scales, vars_)
    plt.title("Sensitivity: Vol Scale vs VaR")
    plt.xlabel("Volatility Scale")
    plt.ylabel("VaR (loss)")
    plt.tight_layout()

    plt.figure()
    plt.plot(vol_scales, cvars_)
    plt.title("Sensitivity: Vol Scale vs CVaR")
    plt.xlabel("Volatility Scale")
    plt.ylabel("CVaR (loss)")
    plt.tight_layout()


def plot_backtest(bt: pd.DataFrame, title: str):
    plt.figure()
    plt.plot(bt.index, bt["realized_loss"].values, label="Realized Loss")
    plt.plot(bt.index, bt["VaR"].values, linestyle="--", label="VaR Forecast")

    breaches = bt[bt["breach"]]
    plt.scatter(breaches.index, breaches["realized_loss"].values, label="Breaches")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Loss ($)")
    plt.legend()
    plt.tight_layout()


# ----------------------------
# Main runner
# ----------------------------

def run_engine(
    tickers: list[str],
    weights: Dict[str, float],
    start: str = "2019-01-01",
    horizon_days: int = 10,
    alpha: float = 0.95,
    n_sims: int = 200_000,
    portfolio_value: float = 1_000_000
):
    prices = fetch_prices(tickers, start=start)

    tickers_used = list(prices.columns)
    w = np.array([weights[t] for t in tickers_used], dtype=float)
    w = w / w.sum()

    log_rets = compute_log_returns(prices)
    if log_rets.empty:
        raise RuntimeError("Log returns are empty. Data download failed or too few rows.")

    mu_annual, cov_annual = annualize_mu_cov(log_rets)

    # Historical risk
    hist_losses = historical_portfolio_losses(
        log_rets, w, horizon_days=horizon_days, portfolio_value=portfolio_value
    )
    hv, hcv = var_cvar(hist_losses, alpha)
    historical = RiskResults(hv, hcv, hist_losses)

    # Monte Carlo risk (Normal vs Student-t)
    mc_losses_norm = simulate_correlated_gbm_losses(
        mu_annual, cov_annual, w,
        horizon_days=horizon_days,
        n_sims=n_sims,
        portfolio_value=portfolio_value,
        dist="normal",
        seed=123
    )
    mv_n, mcv_n = var_cvar(mc_losses_norm, alpha)
    monte_carlo_normal = RiskResults(mv_n, mcv_n, mc_losses_norm)

    mc_losses_t = simulate_correlated_gbm_losses(
        mu_annual, cov_annual, w,
        horizon_days=horizon_days,
        n_sims=n_sims,
        portfolio_value=portfolio_value,
        dist="t",
        t_df=5,
        seed=123
    )
    mv_t, mcv_t = var_cvar(mc_losses_t, alpha)
    monte_carlo_t = RiskResults(mv_t, mcv_t, mc_losses_t)

    # Stress scenario
    mu_s, cov_s = stress_modify_params(mu_annual, cov_annual)
    stress_losses = simulate_correlated_gbm_losses(
        mu_s, cov_s, w,
        horizon_days=horizon_days,
        n_sims=150_000,
        portfolio_value=portfolio_value,
        seed=7,
        dist="normal"
    )
    sv, scv = var_cvar(stress_losses, alpha)
    stress = RiskResults(sv, scv, stress_losses)

    # Backtesting (Historical VaR)
    bt_hist = var_forecast_backtest_historical(
        log_rets, w,
        alpha=alpha,
        horizon_days=horizon_days,
        window=252,
        portfolio_value=portfolio_value
    )
    n = len(bt_hist)
    x = int(bt_hist["breach"].sum())
    lr, pval = kupiec_pof_test(n, x, alpha)

    # Print summary
    print("=== Portfolio Risk Engine ===")
    print("Tickers:", tickers_used)
    print("Weights:", {t: float(w[i]) for i, t in enumerate(tickers_used)})
    print(f"Horizon: {horizon_days} trading days | Portfolio Value: {portfolio_value:,.0f}\n")

    print(f"Historical VaR({int(alpha*100)}%): {historical.var:,.0f} | CVaR: {historical.cvar:,.0f}")
    print(f"Monte Carlo NORMAL VaR({int(alpha*100)}%): {monte_carlo_normal.var:,.0f} | CVaR: {monte_carlo_normal.cvar:,.0f}")
    print(f"Monte Carlo STUDENT-t(df=5) VaR({int(alpha*100)}%): {monte_carlo_t.var:,.0f} | CVaR: {monte_carlo_t.cvar:,.0f}")
    print(f"Stress Test VaR({int(alpha*100)}%): {stress.var:,.0f} | CVaR: {stress.cvar:,.0f}\n")

    print("=== VaR Backtest (Historical) ===")
    print(f"Forecasts: {n} | Breaches: {x} | Breach rate: {x/n:.3%} | Expected: {1-alpha:.3%}")
    print(f"Kupiec POF LR: {lr:.3f} | p-value: {pval:.4f}")

    # Plots
    plot_loss_hist(historical.loss_distribution, "Historical Loss Distribution", alpha)
    plot_loss_hist(monte_carlo_normal.loss_distribution, "Monte Carlo (Normal) Loss Distribution", alpha)
    plot_loss_hist(monte_carlo_t.loss_distribution, "Monte Carlo (Student-t, df=5) Loss Distribution", alpha)
    plot_loss_hist(stress.loss_distribution, "Stress Scenario Loss Distribution", alpha)

    plot_backtest(bt_hist, "Backtest: Historical VaR vs Realized Loss")
    plot_sensitivity(mu_annual, cov_annual, w, horizon_days, portfolio_value, alpha)

    plt.show()


if __name__ == "__main__":
    tickers = ["SPY", "TLT", "GLD", "QQQ"]
    weights = {"SPY": 0.45, "TLT": 0.25, "GLD": 0.10, "QQQ": 0.20}

    run_engine(
        tickers=tickers,
        weights=weights,
        start="2019-01-01",
        horizon_days=10,
        alpha=0.95,
        n_sims=200_000,
        portfolio_value=1_000_000
    )
