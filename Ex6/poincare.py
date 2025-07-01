import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from symplectic import forceHenon, forceToda

t_final = 1000
dt = 10 ** -1
NUM_ITERATIONS = int(t_final / dt)


def poincare_surface(x_0: float, y_0: float, px_0: float, py_0: float, force: callable) -> Tuple[List[float], List[float]]:
    y_poincare = []
    py_poincare = []
    t = np.zeros(shape=NUM_ITERATIONS + 1)
    x = np.zeros(shape=NUM_ITERATIONS + 1)
    y = np.zeros(shape=NUM_ITERATIONS + 1)
    px = np.zeros(shape=NUM_ITERATIONS + 1)
    py = np.zeros(shape=NUM_ITERATIONS + 1)
    t[0] = 0
    x[0] = x_0
    y[0] = y_0
    px[0] = px_0
    py[0] = py_0
    for n in range(NUM_ITERATIONS):
        px[n + 1] = px[n] + force(x[n], y[n])[0] * dt
        py[n + 1] = py[n] + force(x[n], y[n])[1] * dt
        x[n + 1] = x[n] + px[n + 1] * dt
        y[n + 1] = y[n] + py[n + 1] * dt
        t[n + 1] = (n + 1) * dt
        if x[n] <= 0 and x[n + 1] >=0:
            y_poincare.append(y[n])
            py_poincare.append(py[n])
    return y_poincare, py_poincare


def plot_poincare_surface(y, py, title, save_path, 
                               figsize=(15, 5), trajectory_color='blue', start_color='green', 
                               end_color='red', marker_size=80):
    
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    
    # Plot x(t)
    ax.scatter(y[1:-1], py[1:-1], color=trajectory_color, linewidth=2, label='x(t)')
    ax.scatter(y[0], py[0], color=start_color, s=marker_size, marker='x', 
                label=f'Start', zorder=5)
    ax.scatter(y[-1], py[-1], color=end_color, s=marker_size, marker='s', 
                label=f'End', zorder=5, edgecolors='black')
    ax.set_xlabel('y Position')
    ax.set_ylabel('p_y Momentum')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"poincare plot saved to: {save_path}")
    
    return fig, ax

def generate_initial_conditions(E, force):
    if force == forceHenon:
        print(E, np.roots([-E, 0, 1/2, 1/3])[0], np.roots([-E, 0, 1/2, 1/3])[1])
        return [
            (0.0, 0.0, np.sqrt(E), np.sqrt(E)),
            # (0.0, np.roots([-E, 0, 1/2, 1/3])[0], 0.0, 0.0),
            # (0.0, np.roots([-E, 0, 1/2, 1/3])[1], 0.0, 0.0),
        ]
    elif force == forceToda:
        return [
            (0.0, 0.0, np.sqrt(E), np.sqrt(E))
        ]


# Example usage with sample data
if __name__ == "__main__":
    Es_henon = [0.03,0.1,0.16, 0.2, 0.25]
    Es_toda = [0.03,0.1,0.16,0.5,5]
    energies = [Es_henon, Es_toda]
    forces = [forceHenon, forceToda]

    for i ,(Es, force) in enumerate(zip(energies, forces)):
        for E in Es:
            for initial_conditions in generate_initial_conditions(E, force):
                y, py = poincare_surface(
                    *initial_conditions, force=force
                )
            
                # Create output directory
                import os
                output_dir = "Ex6/figures/poincare_plots_output"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                

                # Plot 2D projections
                fig_2d, axes_2d = plot_poincare_surface(
                    y, py,
                    title=f"Poincare Surface E={E}",
                    save_path=os.path.join(output_dir, f"poincare_surface_{force.__name__}_E={E}.png")
                )
        
