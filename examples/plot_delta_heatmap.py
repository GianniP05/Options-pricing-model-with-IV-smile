from models.greeks import greeks_heatmap
import numpy as np

# Example parameters
S = 100                       # Spot price of the underlying
K_array = np.arange(80, 121, 10)   # Strike prices [80, 90, 100, 110, 120]
r = 0.02                      # Risk-free rate = 2%
T_array = np.array([0.25, 0.5, 1.0, 2.0])   # Maturities in years
sigma = 0.25                  # Volatility = 25%
q = 0.01                      # Dividend yield = 1%

# Run the heatmap for each Greek
greeks_heatmap(S, K_array, r, T_array, sigma, q, greek="delta", option_type="call")
