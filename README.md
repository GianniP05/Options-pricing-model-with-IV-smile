# Options-pricing-model-with-IV-smile
This repo contains a clean pipeline to fetch option chains, compute and plot implied vols and Greeks, and fit a single-expiry SVI smile. It’s a solid milestone toward a full volatility surface.
# Features
- Compute options theoretical price with the black scholes model
- Fetch options with `yfinance`
- Compute IVs (Newton solver) and Greeks (Δ, Γ, Θ, Vega, ρ)
- Clean OTM quotes; apply put-call parity to convert puts → synthetic calls
- Calibrate **SVI** (a, b, ρ, m, σ) per expiry via weighted least squares
- Plot the fitted IV smile (SVI curve) or the Greeks heatmap
# Installation
```bash
git clone https://github.com/<your-username>/Options-pricing-model-with-IV-smile.git
cd Options-pricing-model-with-IV-smile
pip install -r requirements.txt
```
# Quick start
```bash
from models.SVI import plot_SVI

# Example: plot TSLA IV smile for Aug 29, 2025
plot_SVI("TSLA", "2025-08-29")
```
# Repo structure
```bash
models/
  black_scholes.py     # Black–Scholes pricing model
  greeks.py            # Greeks: Δ, Γ, Θ, Vega, ρ
  IV_solver.py         # Implied vol solver
  SVI_preprocess.py    # Data fetch + preprocessing
  SVI.py               # SVI calibration + plotting
examples/
  plot_svi_demo.py     # Example script
notebooks/
  demo_SVI.ipynb       # Jupyter demo 
requirements.txt       # Required libraries
README.md
```
# Roadmap
- [ ] Extend to multi-expiry 3D volatility surface
- [ ] Add no-arbitrage checks
- [ ] Build a Streamlit UI

