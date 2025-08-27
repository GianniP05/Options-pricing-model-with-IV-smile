import black_scholes as bs
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

N = stats.norm.cdf
n = stats.norm.pdf

def get_delta(S, K, r, T, sigma, q=0, option_type="call"):
    """
    Calculate the Delta of a European call/put option.

    Delta represents the sensitivity of the option price to a small change in the underlying asset price.

    Parameters:
    - S: Current stock price
    - K: Strike price
    - r: Risk-free interest rate (annualized)
    - T: Time to maturity (in years)
    - sigma: Volatility of underlying asset (annualized)
    - q: Continuous dividend yield (default 0)
    - option_type: str, optional (default='call') 
        Choose between call or put option.

    Returns:
    - Delta of the call/put option
    """
    if option_type == "call":
        return np.exp(-q * T) * N(bs.get_d1(S, K, r, T, sigma, q))
    elif option_type == "put":
        return np.exp(-q * T) * (N(bs.get_d1(S, K, r, T, sigma, q)) - 1)

def get_gamma(S, K, r, T, sigma, q=0):
    """
    Calculate the Gamma of a European option.

    Gamma measures the rate of change of Delta relative to changes in the underlying asset price.

    Parameters as above.

    Returns:
    - Gamma of the option
    """
    return (np.exp(-q * T) * n(bs.get_d1(S, K, r, T, sigma, q))) / (S * sigma * np.sqrt(T))

def get_theta_call(S, K, r, T, sigma, q=0, daily=True):
    """
    Calculate the Theta of a European call option.

    Theta represents the time decay of the option's value. If daily=True, returns Theta per trading day
    assuming 252 trading days per year.

    Parameters as above, plus:
    - daily: Boolean, if True returns Theta per trading day, else per year.

    Returns:
    - Theta of the call option (annualized or daily)
    """
    d1 = bs.get_d1(S, K, r, T, sigma, q)
    d2 = bs.get_d2(S, K, r, T, sigma, q)
    pdf_d1 = n(d1)
    cdf_d1 = N(d1)
    cdf_d2 = N(d2)
    
    term1 = -(S * np.exp(-q * T) * sigma * pdf_d1) / (2 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * cdf_d2
    term3 = q * S * np.exp(-q * T) * cdf_d1
    
    annual_theta = term1 + term2 + term3
    
    if daily:
        return annual_theta / 252
    else:
        return annual_theta

def get_theta_put(S, K, r, T, sigma, q=0, daily=True):
    """
    Calculate the Theta of a European put option.

    Parameters and return same as get_theta_call.

    Returns:
    - Theta of the put option (annualized or daily)
    """
    d1 = bs.get_d1(S, K, r, T, sigma, q)
    d2 = bs.get_d2(S, K, r, T, sigma, q)
    pdf_d1 = n(d1)
    cdf_neg_d1 = N(-d1)
    cdf_neg_d2 = N(-d2)
    
    term1 = -(S * np.exp(-q * T) * sigma * pdf_d1) / (2 * np.sqrt(T))
    term2 = r * K * np.exp(-r * T) * cdf_neg_d2
    term3 = -q * S * np.exp(-q * T) * cdf_neg_d1

    annual_theta = term1 + term2 + term3
    
    if daily:
        return annual_theta / 252
    else:
        return annual_theta

def get_vega(S, K, r, T, sigma, q=0):
    """
    Calculate the Vega of an option.

    Vega measures the sensitivity of the option price to changes in volatility.

    Parameters as above.

    Returns:
    - Vega per 1% change in volatility (scaled down by 100)
    """
    vega = S * np.exp(-q * T) * np.sqrt(T) * n(bs.get_d1(S, K, r, T, sigma, q))
    return vega / 100

def get_rho(S, K, r, T, sigma, q=0, option_type="call"):
    """
    Calculate the Rho of a European call/put option.

    Rho measures sensitivity of the option price to changes in the risk-free interest rate.

    Parameters as above, plus:
    - option_type: str, optional (default='call') 
        Choose between call or put option.

    Returns:
    - Rho per 1% change in interest rates (scaled down by 100)
    """
    if option_type == "call":
        rho = K * T * np.exp(-r * T) * N(bs.get_d2(S, K, r, T, sigma, q))
        return rho / 100
    elif option_type == "put":
        rho = -K * T * np.exp(-r * T) * N(-bs.get_d2(S, K, r, T, sigma, q))
        return rho / 100
    
def greeks_heatmap(S, K_array, r, T_array, sigma, q=0, greek="delta",option_type="call"):
    """
    Plots a heatmap for a selected Greek (Delta, Vega, or Theta) over a range of strike prices and maturities.

    Parameters as above, plus:
    - greek : str, optional (default='delta')
        The Greek to plot. Supported values: 'delta', 'vega', 'theta'.

    Returns:
    - None
        Displays a seaborn heatmap with strike prices on the y-axis,
        maturities on the x-axis, and the Greek values as cells.
    """
    greek_matrix = np.zeros((len(K_array), len(T_array)))
    if greek == "delta":
        deltas = []
        for i in range(len(K_array)):
            for j in range(len(T_array)):
                K = K_array[i]
                T = T_array[j]
                deltas.append(get_delta(S, K, r, T, sigma, q, option_type))
        
        greek_matrix = np.array(deltas).reshape(len(K_array), len(T_array))
    elif greek == "vega":
        vegas = []
        for i in range(len(K_array)):
            for j in range(len(T_array)):
                K = K_array[i]
                T = T_array[j]
                vegas.append(get_vega(S, K, r, T, sigma, q))
        
        greek_matrix = np.array(vegas).reshape(len(K_array), len(T_array))
    elif greek == "theta" and option_type == "call":
        thetas = []
        for i in range(len(K_array)):
            for j in range(len(T_array)):
                K = K_array[i]
                T = T_array[j]
                thetas.append(get_theta_call(S, K, r, T, sigma, q))
        
        greek_matrix = np.array(thetas).reshape(len(K_array), len(T_array))
    elif greek == "theta" and option_type == "put":
        thetas = []
        for i in range(len(K_array)):
            for j in range(len(T_array)):
                K = K_array[i]
                T = T_array[j]
                thetas.append(get_theta_put(S, K, r, T, sigma, q))
        
        greek_matrix = np.array(thetas).reshape(len(K_array), len(T_array))

    plt.figure()
    sns.heatmap(pd.DataFrame(greek_matrix, K_array, T_array), annot=True, cmap="viridis")
    if greek == "vega":
        plt.title(f"{greek}")
    else:
        plt.title(f"{greek} {option_type}")
    plt.xlabel("Maturity (Years)")
    plt.ylabel("Strike Price")
    plt.show()



