import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple

t_final = 10
alpha = 4
dt = 10 ** -3
NUM_ITERATIONS = int(t_final / dt)

def plot_on_figure(
    x: np.ndarray,
    y: np.ndarray,
    label: str,
    fig: plt.Figure,
    color: str = "tab:blue",
    linewidth: float = 2.0,
) -> None:
    ax = fig.gca()
    ax.plot(x, y, label=label, color=color, linewidth=linewidth)
    ax.legend(fontsize=11)
    fig.tight_layout()

def plot_errorbars_on_figure(
    x: np.ndarray,
    x_err: np.ndarray,
    y: np.ndarray,
    y_err: np.ndarray,
    label: str,
    fig: plt.Figure,
    color: str,
    linewidth: float = 2.0,
) -> None:
    ax = fig.gca()
    ax.errorbar(
        x, y, xerr=x_err, yerr=y_err, color=color, linewidth=linewidth, fmt='.', capsize=5, label=label
    )
    ax.legend(fontsize=11)
    fig.tight_layout()

def solve_numeric_symplectic(x_0: float, p_0: float, num_iterations: int, dt: float) -> Tuple[List[float], List[float]]:
    cross_points = []
    t = np.zeros(shape=num_iterations + 1)
    x = np.zeros(shape=num_iterations + 1)
    p = np.zeros(shape=num_iterations + 1)
    t[0] = 0
    x[0] = x_0
    p[0] = p_0
    for n in range(num_iterations):
        p[n + 1] = p[n] - alpha * np.sign(x[n]) * dt
        x[n + 1] = x[n] + p[n + 1] * dt
        t[n + 1] = (n + 1) * dt
        if (x[n] >= 0 and x[n + 1] <=0) or (x[n] <= 0 and x[n + 1] >= 0):
            cross_points.append(t[n])
    cycle_time = 2 * np.average(np.diff(cross_points))
    cycle_time_err = np.std(np.diff(cross_points), ddof=1)
    frequecy = 2 * np.pi / cycle_time
    frequecy_err = 2 * np.pi * cycle_time_err / (cycle_time ** 2)
    return t, x, p, frequecy, frequecy_err

def simulation() -> None:
    phase_space_fig = plt.figure("Phase Space Trajectories", figsize=(8, 5))
    ax = phase_space_fig.gca()
    ax.set_title("Phase Space Trajectories", fontsize=14)
    ax.set_xlabel("Position (x)", fontsize=12)
    ax.set_ylabel("Momentum (p)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    trajectory_fig = plt.figure("Trajectories", figsize=(8, 5))
    ax = trajectory_fig.gca()
    ax.set_title("Trajectories", fontsize=14)
    ax.set_xlabel("Time (t)", fontsize=12)
    ax.set_ylabel("Position (x)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    frequencies = []
    frequency_errs = []
    energies = np.linspace(10, 50, 5)

    for E in energies:
        x_0 = 0.0
        p_0 = np.sqrt(2 * E)
        t, x, p, frequency, frequency_err = solve_numeric_symplectic(x_0, p_0, NUM_ITERATIONS, dt)
        plot_on_figure(
            x, p, f"E={E:.2f}", phase_space_fig, color=plt.cm.viridis(E / 50)
        )
        plot_on_figure(
            t, x, f"E={E:.2f}", trajectory_fig, color=plt.cm.viridis(E / 50)
        )
        frequencies.append(frequency)
        frequency_errs.append(frequency_err)

    frequencies_fig = plt.figure("Frequencies vs Energy", figsize=(8, 5))
    ax = frequencies_fig.gca()
    ax.set_title("Frequencies vs Energy", fontsize=14)
    ax.set_xlabel("Energy (E)", fontsize=12)
    ax.set_ylabel("Frequency (Hz)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    plot_errorbars_on_figure(
        energies, np.zeros_like(energies), frequencies, frequency_errs, "Simulated Frequencies", frequencies_fig, color='tab:red'
    )

    continuous_energies = np.linspace(10, 50, 100)
    theoretical_frequencies = np.pi * alpha / (2 * np.sqrt(2 * continuous_energies))
    plot_on_figure(
        continuous_energies, theoretical_frequencies, "Theoretical Frequency $\omega(I) = \\frac{\pi a}{2 \sqrt{2E}}$", frequencies_fig, color='tab:orange'
    )

if __name__ == "__main__":
    simulation()
    plt.show()
