"""
SVI_preprocess.py
-----------------
Fetch option chain data, compute implied volatilities, and preprocess them into
log-moneyness, total variance, and weights for SVI calibration.
"""

import yfinance as yf
from datetime import datetime, date
import numpy as np
import pandas as pd
import models.IV_solver_v2 as sol
from greeks import get_vega

def get_T(expiry: str) -> float:
    """
    Compute time-to-expiry in years.

    Parameters
    ----------
    expiry : str
        Expiry date in 'YYYY-MM-DD' format.

    Returns
    -------
    float
        Time to expiry (>= 1e-6 to avoid division by zero).
    """
    exp_str = str(expiry)[:10]
    days = (datetime.strptime(exp_str, "%Y-%m-%d").date() - date.today()).days
    return max(days / 365.25, 1e-6)

def get_logmoneyness_and_totalvariance_and_weights(ticker: str, expiry: str, riskfree_rate: float = 0.0424):
    """
    Fetch option chain for a given ticker/expiry and compute
    log-moneyness, total variance, and weights for SVI calibration.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'TSLA').
    expiry : str
        Expiry date in 'YYYY-MM-DD' format.
    riskfree_rate : float, optional
        Risk-free interest rate (continuously compounded), default 0.0424.

    Returns
    -------
    k : np.ndarray
        Log-moneyness array (ln(K/F)).
    w : np.ndarray
        Total variance array (T * IV^2).
    wt : np.ndarray
        Normalized weights (vega * volume).
    """
    exp_str = str(expiry)[:10]
    T = get_T(exp_str)

    # Spot and forward price
    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")["Close"].dropna()
    if hist.empty:
        return np.array([]), np.array([]), np.array([])
    spot = float(hist.iloc[-1])
    forward = float(spot * np.exp(riskfree_rate * T))

    # Option chain
    chain = tk.option_chain(exp_str)
    calls = chain.calls[["strike","bid","ask","volume"]].copy()
    puts  = chain.puts [["strike","bid","ask","volume"]].copy()

    # Filters with volume guards
    vc = float(calls["volume"].sum()) if not calls.empty else 0.0
    vp = float(puts["volume"].sum())  if not puts.empty  else 0.0

    # Clean calls (bid/ask > 0, bid < ask, sufficient volume share)
    if not calls.empty:
        m = (calls["bid"] > 0) & (calls["ask"] > 0) & (calls["bid"] < calls["ask"])
        if vc > 0: m &= (calls["volume"]/vc) > 0.0025
        calls = calls.loc[m].copy()
        calls["mid"] = 0.5*(calls["bid"]+calls["ask"])
        # Only OTM calls (K >= F)
        df_calls_otm = calls.loc[calls["strike"] >= forward, ["strike","mid","volume"]].copy()
    else:
        df_calls_otm = pd.DataFrame(columns=["strike","mid","volume"])

    # Clean puts and convert to synthetic calls using put-call parity
    if not puts.empty:
        m = (puts["bid"] > 0) & (puts["ask"] > 0) & (puts["bid"] < puts["ask"])
        if vp > 0: m &= (puts["volume"]/vp) > 0.0025
        puts = puts.loc[m].copy()
        puts["mid"] = 0.5*(puts["bid"]+puts["ask"])
        disc = np.exp(-riskfree_rate * T)
        # Put-call parity: C = P + e^{-rT}(F - K)
        syn_mid = puts["mid"] + disc*(forward - puts["strike"])
        df_put_syn_otm = pd.DataFrame({
            "strike": puts["strike"].values,
            "mid": syn_mid.values,
            "volume": puts["volume"].values,
        })
        # Only OTM synthetic calls (K < F)
        df_put_syn_otm = df_put_syn_otm.loc[df_put_syn_otm["strike"] < forward]
    else:
        df_put_syn_otm = pd.DataFrame(columns=["strike","mid","volume"])

    # Combine synthetic calls and OTM calls
    mid = pd.concat([df_put_syn_otm, df_calls_otm], ignore_index=True)
    if mid.empty:
        return np.array([]), np.array([]), np.array([])
    mid = mid.sort_values("strike").reset_index(drop=True)

    # Prepare arrays
    K = mid["strike"].to_numpy(float)            # Strikes
    P = mid["mid"].to_numpy(float).reshape(-1,1) # Options mid prices (N,1)
    V = mid["volume"].to_numpy(float)            # Volume
    Tarr = np.full((K.size,1), T, float)         # Maturities (N,1)
    
    # Implied vols for each mid price
    IV = sol.implied_vol_array(P, spot, K, riskfree_rate, Tarr)  # expect (N,1) or (N,)
    IV = np.asarray(IV, float).reshape(-1)       # (N,)

    # One common validity mask
    valid = (
        np.isfinite(K) & (K>0) &
        np.isfinite(P).reshape(-1) & (P.reshape(-1)>0) &
        np.isfinite(IV) & (IV>0) & (IV < 5.0)    # clip extremes
    )
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([])

    K, V, IV = K[valid], V[valid], IV[valid]
    forward = float(forward)  # Ensure scalar

    # Final arrays: log-moneyness and total variance
    k = np.log(K / forward).astype(float).reshape(-1)
    w = (T * IV**2).astype(float).reshape(-1)

    # Weights = vega * volume (normalized)
    vega = np.array([max(get_vega(spot, K[i], riskfree_rate, T, IV[i]), 0.0) for i in range(K.size)], float)
    wt = vega * np.maximum(V, 0.0)
    s = wt.sum()
    wt = (wt / s) if (np.isfinite(s) and s>0) else np.full_like(k, 1.0/k.size)

    return k, w, wt
