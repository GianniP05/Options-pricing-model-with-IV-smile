import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.SVI import plot_SVI

# Plot the implied vol smile for the TSLA option that expires the 19th of September 2025
plot_SVI('TSLA', '2025-09-19')
