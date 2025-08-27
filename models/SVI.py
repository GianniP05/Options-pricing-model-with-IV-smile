from models.SVI_preprocess import get_T,get_logmoneyness_and_totalvariance_and_weights
from scipy.optimize import differential_evolution, minimize
import numpy as np
import matplotlib.pyplot as plt

def get_SVI(ticker, expiry):
    """
    Calibrate the Stochastic Volatility Inspired (SVI) model for a given ticker and expiry.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "TSLA").
    expiry : str
        Option expiry date in 'YYYY-MM-DD' format.

    Returns
    -------
    IV_SVI : np.ndarray
        Implied volatilities from the calibrated SVI smile.
    k : np.ndarray
        Corresponding log-moneyness values used in the fit.
    """
    k_raw, w_raw, wt_raw = get_logmoneyness_and_totalvariance_and_weights(ticker, expiry)
    # Ensure 1-D arrays and mask invalid values
    k  = np.asarray(k_raw, dtype=float).reshape(-1)
    w  = np.asarray(w_raw, dtype=float).reshape(-1)
    wt = np.asarray(wt_raw, dtype=float).reshape(-1)

    valid = np.isfinite(k) & np.isfinite(w) & np.isfinite(wt) & (wt > 0)
    k, w, wt = k[valid], w[valid], wt[valid]
    assert k.size == w.size == wt.size, (k.size, w.size, wt.size)

    T_years = max(get_T(expiry), 1e-6)

    def parameters(m, sigma):
        """
        Compute linear parameters (a, b, rho) for given m and sigma.
        """
        y = (k - m) / sigma
        z = np.sqrt(y**2 + 1.0)
        M = np.vstack([np.ones_like(y), y, z])  # (3, n)
        G = M @ M.T                              # (3, 3)
        h = M @ w                                # (3,)
        a, d, c = np.linalg.solve(G, h)
        b   = np.sqrt(d**2 + c**2)
        rho = d / b
        return a, b, rho

    def obj_func(x):
        """
        Objective function (weighted squared error) for SVI calibration.
        """
        m, sigma = x
        a, b, rho = parameters(m, sigma)
        # Enforce constraints to ensure no-arbitrage feasibility
        if sigma <= 0 or not (-1 < rho < 1) or b > (4.0 / ((1.0 + abs(rho)) * T_years)):
            return 1e12
        # Compute SVI total variance
        w_svi = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        # Weighted squared error vs observed total variance
        return np.sum(wt * (w_svi - w)**2)

    # Parameter search bounds
    m_min, m_max = k.min(), k.max()
    sigma_min = max(0.05, 0.2 * k.std())
    bounds = [(m_min - 0.1, m_max + 0.1), (sigma_min, 1.0)]
    # Global search (DE) followed by local refinement (Nelder-Mead)
    res_g = differential_evolution(obj_func, bounds=bounds)
    res   = minimize(obj_func, x0=res_g.x, method="Nelder-Mead")
    # Recover optimal parameters
    m_opt, sigma_opt = res.x
    a, b, rho = parameters(m_opt, sigma_opt)
    # Compute fitted SVI IV curve
    w_svi = a + b * (rho * (k - m_opt) + np.sqrt((k - m_opt)**2 + sigma_opt**2))
    IV_SVI = np.sqrt(np.maximum(w_svi, 0.0) / T_years)
    return IV_SVI, k

def plot_SVI(ticker, expiry):
    """
    Plot the calibrated SVI volatility smile for a given ticker and expiry.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "TSLA").
    expiry : str
        Option expiry date in 'YYYY-MM-DD' format.
    """
    IV, k = get_SVI(ticker, expiry)
    k = np.asarray(k, float).reshape(-1)
    plt.plot(k, IV, "r", label="SVI fit")
    plt.xlabel("Log-moneyness"); plt.ylabel("Implied Vol")
    plt.title(f"SVI smile for {ticker} — {expiry}")
    plt.legend()
    plt.show()




