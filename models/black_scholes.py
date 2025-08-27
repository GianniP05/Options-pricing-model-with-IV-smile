from scipy import stats
import numpy as np

N = stats.norm.cdf

def get_d1(S, K, r, T, sigma, q=0):
    """
    Calculate d1 used in the Black-Scholes formula.

    Parameters:
    S (float): Current spot price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate (annualized, continuous compounding).
    T (float): Time to maturity in years.
    sigma (float): Annualized volatility of the underlying asset.
    q (float, optional): Continuous dividend yield (default is 0).

    Returns:
    float: Value of d1.
    """
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def get_d2(S, K, r, T, sigma, q=0):
    """
    Calculate d2 used in the Black-Scholes formula.

    Parameters:
    S, K, r, T, sigma, q : same as get_d1.

    Returns:
    float: Value of d2.
    """
    return get_d1(S, K, r, T, sigma, q) - sigma * np.sqrt(T)

def call_BS(S, K, r, T, sigma, q=0):
    """
    Calculate the Black-Scholes price of a European call option with continuous dividend yield.

    Parameters:
    S (float): Current spot price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate (annualized, continuous compounding).
    T (float): Time to maturity in years.
    sigma (float): Annualized volatility of the underlying asset.
    q (float, optional): Continuous dividend yield (default is 0).

    Returns:
    float: Theoretical price of the European call option.
    """
    d1 = get_d1(S, K, r, T, sigma, q)
    d2 = get_d2(S, K, r, T, sigma, q)
    return S * np.exp(-q * T) * N(d1) - K * np.exp(-r * T) * N(d2)

def put_BS(S, K, r, T, sigma, q=0):
    """
    Calculate the Black-Scholes price of a European put option with continuous dividend yield.

    Parameters:
    S (float): Current spot price of the underlying asset.
    K (float): Strike price of the option.
    r (float): Risk-free interest rate (annualized, continuous compounding).
    T (float): Time to maturity in years.
    sigma (float): Annualized volatility of the underlying asset.
    q (float, optional): Continuous dividend yield (default is 0).

    Returns:
    float: Theoretical price of the European put option.
    """
    d1 = get_d1(S, K, r, T, sigma, q)
    d2 = get_d2(S, K, r, T, sigma, q)
    return K * np.exp(-r * T) * N(-d2) - S * np.exp(-q * T) * N(-d1)

