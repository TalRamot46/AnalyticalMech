import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# 1. Define the function f(x,y)
def HenonHils(x, y, E):
    return 1/2 * (x**2 + y**2) + x*y**2 - y**3/3 - E

def Toda(x, y, E):
    return 1/24 * (np.exp(2*y+2*np.sqrt(3)*x) + np.exp(2*y-2*np.sqrt(3)*x) + np.exp(-4*y)) - 1/8 - E

def plot_multi_energy_contours(func, X, Y, energies, ax, title, colormap='viridis', separatrix_E=None):
    print(f"Plotting {title} contours...")
    colors = plt.cm.get_cmap(colormap)(np.linspace(0, 1, len(energies)))
    
    for i, E in enumerate(energies):
        Z = func(X, Y, E)
        try:
            ax.contour(X, Y, Z, levels=[0], colors=[colors[i]], linewidths=1.5)
        except:
            print(f"Warning: Could not plot contour for E = {E}")
            continue
    
    # Plot separatrix if specified
    if separatrix_E is not None:
        Z_sep = func(X, Y, separatrix_E)
        try:
            contour_sep = ax.contour(X, Y, Z_sep, levels=[0], colors=['red'], linewidths=3)
            # Create a custom legend entry for the separatrix
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='red', lw=3, label=f'Separatrix (E={separatrix_E:.4f})')]
            ax.legend(handles=legend_elements, loc='upper right')
            print(f"Added separatrix contour at E = {separatrix_E}")
        except:
            print(f"Warning: Could not plot separatrix contour for E = {separatrix_E}")
    
    # Format the plot
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=energies.min(), vmax=energies.max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
    cbar.set_label('Energy (E)', rotation=270, labelpad=20)

def plot_single_contour(func, X, Y, E, title, save_path=None):
    """
    Plot a single contour at specified energy level
    
    Parameters:
    func: function to plot
    X, Y: meshgrid coordinates
    E: energy value
    title: plot title
    save_path: optional path to save the figure
    """
    Z = func(X, Y, E)
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=[0], colors='blue', linewidths=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Single contour plot saved to: {save_path}")

def create_output_directory():
    """Create output directory for saving figures"""
    output_dir = "Ex6/figures/contour_plots_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
    return output_dir

# 2. Create a grid of x and y values
x_min, x_max = -3, 3
y_min, y_max = -3, 3
x = np.linspace(x_min, x_max, 400)
y = np.linspace(y_min, y_max, 400)
X, Y = np.meshgrid(x, y)

# 3. Define different energy arrays for each function
E_henon = np.linspace(0.01, 0.3, 15)  # Energy levels for Henon-Heiles
E_toda = np.linspace(-0.1, 5, 15)   # Energy levels for Toda

# 4. Create output directory
output_dir = create_output_directory()

# 5. Create multi-energy contour plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

plot_multi_energy_contours(HenonHils, X, Y, E_henon, ax1, 
                          'Henon-Heiles Function Contours\n(Multiple Energy Levels)', 
                          'viridis',
                          separatrix_E=0.0523)

plot_multi_energy_contours(Toda, X, Y, E_toda, ax2, 
                          'Toda Function Contours\n(Multiple Energy Levels)', 
                          'plasma')

plt.tight_layout()

# Save the multi-energy contour plot
multi_energy_path = os.path.join(output_dir, "multi_energy_contours.png")
plt.savefig(multi_energy_path, dpi=300, bbox_inches='tight')
print(f"Multi-energy contour plot saved to: {multi_energy_path}")

# 6. Create and save single contour plot
single_contour_path = os.path.join(output_dir, "single_contour_henon_heiles.png")
plot_single_contour(HenonHils, X, Y, E=0.01, 
                   title='Original Plot: Henon-Heiles f(x,y) = 0 at E = 0.03',
                   save_path=single_contour_path)

print(f"\nAll plots have been saved to the '{output_dir}' directory.")