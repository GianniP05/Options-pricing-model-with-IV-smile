import black_scholes as bs
from scipy.optimize import brentq
import numpy as np

def implied_vol_scalar(option_price, S, K, r, T, option_type='call', q=0):
    """
    Robust implied-vol solver with bounds‐check and logging.
    Returns np.nan if price is outside [intrinsic, max] or Brent fails.
    """
    # 1) cast everything to plain floats
    option_price = float(option_price)
    S            = float(S)
    K            = float(K)
    r            = float(r)
    T            = float(T)
    q            = float(q)

    # 2) define price-error function
    def bs_price_error(sigma):
        if option_type == 'call':
            return bs.call_BS(S, K, r, T, sigma, q) - option_price
        elif option_type == 'put':
            return bs.put_BS(S, K, r, T, sigma, q) - option_price
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    # 3) compute no‐arb bounds: intrinsic ≤ C ≤ S*e^{−qT}
    S_disc    = S * np.exp(-q * T)
    K_disc    = K * np.exp(-r * T)
    intrinsic = max(S_disc - K_disc, 0.0)
    max_price = S_disc

    if option_price < intrinsic or option_price > max_price:
        return np.nan

    # 4) check bracket endpoints
    lo, hi   = 1e-6, 5.0
    
    # 5) root‐find
    try:
        iv = brentq(bs_price_error, lo, hi)
        return round(iv, 4)
    except ValueError as e:
        print(f" Bracket fail for K={K:.2f}: {e}")
        return np.nan


def implied_vol_array(market_prices, S, K_array, r, T_array, option_type='call', q=0):
    """
    Vectorized wrapper: forces K_array and T_array into numpy arrays of floats,
    then calls implied_vol_scalar on each element.
    """
    K_arr = np.asarray(K_array, dtype=float)
    T_arr = np.asarray(T_array, dtype=float)

    N, M = market_prices.shape
    IVs  = np.full((N, M), np.nan, dtype=float)

    for i in range(N):
        Ki = K_arr[i]
        for j in range(M):
            price_ij = market_prices[i, j]
            Tij      = T_arr[j]
            IVs[i, j] = implied_vol_scalar(price_ij, S, Ki, r, Tij, option_type, q)

    return IVs
