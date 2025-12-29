"""
visualize.py - Create validation visualizations

Your task: Create a scatter plot comparing volunteer vs professional
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import yaml

def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def create_validation_plot(matches_df, config):
    """
    Create scatter plot comparing volunteer vs. professional measurements
    
    TODO: Create publication-quality plot
    
    Your plot should include:
    1. Scatter plot of matched pairs
       - X-axis: Professional chloride (Pro_Value)
       - Y-axis: Volunteer chloride (Vol_Value)
    2. Linear regression line
    3. 1:1 reference line (perfect agreement)
    4. Statistics text box showing N, R², slope
    5. Proper labels and title
    
    Hints:
    - Use plt.subplots(figsize=(10, 8))
    - Use ax.scatter() for points
    - Use stats.linregress() for regression line
    - Use ax.plot() for lines
    - Use ax.text() for statistics box
    - Save with dpi=300 for publication quality
    """
    
    # TODO: Extract values
    vol_values = ... # TODO
    pro_values = ... # TODO
    
    # TODO: Calculate regression
    # HINT: slope, intercept, r_value, p_value, std_err = stats.linregress(...)
    
    # TODO: Create figure and axis
    fig, ax = ... # TODO
    
    # TODO: Create scatter plot
    # HINT: ax.scatter(pro_values, vol_values, ...)
    # Suggestion: Use alpha=0.6, color='steelblue', s=100
    
    # TODO: Add regression line
    # HINT: Create x_line = np.linspace(min, max)
    # HINT: Calculate y_line = slope * x_line + intercept
    # HINT: ax.plot(x_line, y_line, 'r-', ...)
    
    # TODO: Add 1:1 reference line
    # HINT: max_val = max(pro_values.max(), vol_values.max())
    # HINT: ax.plot([0, max_val], [0, max_val], 'k--', ...)
    
    # TODO: Add labels
    # ax.set_xlabel('Professional Chloride (mg/L)', ...)
    # ax.set_ylabel('Volunteer Chloride (mg/L)', ...)
    # ax.set_title(...)
    
    # TODO: Add statistics text box
    # HINT: stats_text = f'N = {len(matches_df)}\nR² = {r_squared:.3f}\n...'
    # HINT: ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, ...)
    
    # TODO: Add grid and legend
    
    # TODO: Save figure
    # HINT: output_path = Path(config['output_paths']['results']) / "validation_plot.png"
    # HINT: plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.close()

def main():
    """Create all visualizations"""
    
    config = load_config()
    
    # TODO: Load matched pairs
    
    # TODO: Create validation plot
    
    print("\n✅ Visualization complete")

if __name__ == "__main__":
    main()
