import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

t_final = 20
alpha = 4
dt = 10 ** -3
NUM_ITERATIONS = int(t_final / dt)

def forceHenon(x: float, y: float) -> Tuple[float, float]:
    Fx = - (x + y ** 2)
    Fy = - (1 + 2*x)*y + y**2
    return Fx, Fy

def forceToda(x: float, y: float) -> Tuple[float, float]:
    Fx = - np.sqrt(3)/12 * (np.exp(2*y+2*np.sqrt(3)*x) - np.exp(2*y-2*np.sqrt(3)*x) + np.exp(-4*y))
    Fy = - 1/12 * (np.exp(2*y+2*np.sqrt(3)*x) + np.exp(2*y-2*np.sqrt(3)*x) - 2 * np.exp(-4*y))
    return Fx, Fy

def solve_numeric_symplectic(x_0: float, y_0: float, px_0: float, py_0: float, force: callable) -> Tuple[List[float], List[float]]:
    cross_points = []
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
        if (x[n] >= 0 and x[n + 1] <=0) or (x[n] <= 0 and x[n + 1] >= 0):
            cross_points.append(t[n])
    return t, x, y, px, py

def plot_3d_dynamics(t, x, y, title="3D Trajectory Dynamics", save_path=None, 
                     figsize=(12, 9), trajectory_color='blue', start_color='green', 
                     end_color='red', marker_size=100):
    """
    Plot 3D dynamics showing the trajectory (x(t), y(t)) in space with time as the third dimension
    
    Parameters:
    t: array of time values
    x: array of x coordinates
    y: array of y coordinates
    title: plot title
    save_path: optional path to save the figure
    figsize: figure size tuple
    trajectory_color: color of the trajectory line
    start_color: color of the starting point marker
    end_color: color of the ending point marker
    marker_size: size of start/end markers
    """
    
    # Create 3D figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory line
    ax.plot(x, y, t, color=trajectory_color, linewidth=2, alpha=0.8, label='Trajectory')
    
    # Mark starting point
    ax.scatter(x[0], y[0], t[0], color=start_color, s=marker_size, 
               marker='o', label=f'Start (t={t[0]:.3f})', alpha=0.9, edgecolors='black')
    
    # Mark ending point
    ax.scatter(x[-1], y[-1], t[-1], color=end_color, s=marker_size, 
               marker='s', label=f'End (t={t[-1]:.3f})', alpha=0.9, edgecolors='black')
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Time (t)')
    ax.set_title(title)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D dynamics plot saved to: {save_path}")
    
    return fig, ax


def plot_2d_projection_dynamics(t, x, y, title="2D Trajectory Projections", save_path=None, 
                               figsize=(15, 5), trajectory_color='blue', start_color='green', 
                               end_color='red', marker_size=80):
    """
    Plot 2D projections of the dynamics: x(t), y(t), and y(x)
    
    Parameters:
    t, x, y: arrays of time and coordinates
    title: main title for the subplot figure
    save_path: optional path to save the figure
    Other parameters: styling options
    """
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Plot x(t)
    ax1.plot(t, x, color=trajectory_color, linewidth=2, label='x(t)')
    ax1.scatter(t[0], x[0], color=start_color, s=marker_size, marker='o', 
                label=f'Start', zorder=5, edgecolors='black')
    ax1.scatter(t[-1], x[-1], color=end_color, s=marker_size, marker='s', 
                label=f'End', zorder=5, edgecolors='black')
    ax1.set_xlabel('Time (t)')
    ax1.set_ylabel('X Position')
    ax1.set_title('X vs Time')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot y(t)
    ax2.plot(t, y, color=trajectory_color, linewidth=2, label='y(t)')
    ax2.scatter(t[0], y[0], color=start_color, s=marker_size, marker='o', 
                label=f'Start', zorder=5, edgecolors='black')
    ax2.scatter(t[-1], y[-1], color=end_color, s=marker_size, marker='s', 
                label=f'End', zorder=5, edgecolors='black')
    ax2.set_xlabel('Time (t)')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Y vs Time')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot y(x) - phase space
    ax3.plot(x, y, color=trajectory_color, linewidth=2, label='Phase trajectory')
    ax3.scatter(x[0], y[0], color=start_color, s=marker_size, marker='o', 
                label=f'Start', zorder=5, edgecolors='black')
    ax3.scatter(x[-1], y[-1], color=end_color, s=marker_size, marker='s', 
                label=f'End', zorder=5, edgecolors='black')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('Phase Space (Y vs X)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_aspect('equal', adjustable='box')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"2D projections plot saved to: {save_path}")
    
    return fig, (ax1, ax2, ax3)


# Example usage with sample data
if __name__ == "__main__":
    E_henon = 0.001
    E_toda = 0.001
    energies = [E_henon, E_toda]
    forces = [forceHenon, forceToda]

    for i ,(E, force) in enumerate(zip(energies, forces)):
        t_sample, x_sample, y_sample, px_sample, py_sample = solve_numeric_symplectic(
            x_0=0, y_0=0, px_0=np.sqrt(E), py_0=np.sqrt(E), force=force
        )
    
        # Create output directory
        import os
        output_dir = "Ex6/figures/dynamics_plots_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")
        
        # Plot 3D dynamics
        fig_3d, ax_3d = plot_3d_dynamics(
            t_sample, x_sample, y_sample, 
            title=f"3D Spiral Trajectory Dynamics {force.__name__}",
            save_path=os.path.join(output_dir, f"3d_dynamics_example-{force.__name__}.png")
        )
        
        # Plot 2D projections
        fig_2d, axes_2d = plot_2d_projection_dynamics(
            t_sample, x_sample, y_sample,
            title=f"2D Projections of Spiral Dynamics {force.__name__}",
            save_path=os.path.join(output_dir, f"2d_projections_example-{force.__name__}.png")
        )
        
