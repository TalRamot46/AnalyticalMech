import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

t_final = 10
alpha = 4
dt = 10 ** -3
NUM_ITERATIONS = int(t_final / dt)

omega_0 = 1
E_sx = omega_0**2

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
        x, y, 
        xerr=x_err, yerr=y_err, 
        color=color, 
        linewidth=linewidth, 
        fmt='.', capsize=5, label=label
    )
    ax.legend(fontsize=11)
    fig.tight_layout()

def calc_force(x):
    return omega_0 ** 2 * np.sin(x)


def calc_frequency(E) :
    if E < -E_sx:
        raise ValueError("Energy must be greater than or " \
        "equal to minus of the seperatrix energy.")
    k = np.sqrt(1/2 * (1 + E / E_sx))
    m = k**2
    if k < 1: # Libration
        return omega_0 * np.pi / 2 * 1 / scipy.special.ellipk(m)
    else: # Rotation
        return omega_0 * np.pi * k / scipy.special.ellipk(1 /m)

def calc_frequency_first_order(E):
    if E < -E_sx:
        raise ValueError("Energy must be greater than or equal " \
        "to minus of the seperatrix energy.")
    return omega_0 - 1 / (8 * omega_0) * (E + E_sx) 
    

def plot_frequency_vs_energy():
    energies = np.linspace(-E_sx, E_sx + 2, 10000)
    freqs_theory = np.array([calc_frequency(E) for E in energies])
    freqs_linear_approx = np.array([calc_frequency_first_order(E) for E in energies])


    freq_energy_fig = plt.figure("Frequency vs Energy", figsize=(8, 5))
    ax = freq_energy_fig.gca()
    ax.set_title("Frequency vs Energy for a Pendulum", fontsize=14)
    ax.set_xlabel("Energy (E)", fontsize=12)
    ax.set_ylabel("Frequency ($\omega$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plot_on_figure(energies, freqs_theory, label="Exact Frequency", fig=freq_energy_fig, color="tab:blue")
    plot_on_figure(energies, freqs_linear_approx, label="First Order Approximation", fig=freq_energy_fig, color="tab:red")
    i = 0
    while np.abs(freqs_theory[i] - freqs_linear_approx[i]) / freqs_theory[i] < 0.1:
        i += 1
    ax.axvline(x=energies[i], color='tab:orange', linestyle='--', label=f'10% error ($E={energies[i]:.2f}$)') # Adding a vertical line for emphasis and label
    ax.legend()

# plot_frequency_vs_energy()
# plt.show()



def check_oscillation_type(x, p):
    E = p**2 / 2 - omega_0 ** 2 * np.cos(x)
    if E > E_sx:
        return "Rotation"
    else:
        return "Libration"

def solve_numeric_symplectic(x_0: float, p_0: float, num_iterations: int, dt: float) -> Tuple[List[float], List[float]]:
    cross_points = []
    t = np.zeros(shape=num_iterations + 1)
    x = np.zeros(shape=num_iterations + 1)
    p = np.zeros(shape=num_iterations + 1)
    t[0] = 0
    x[0] = x_0
    p[0] = p_0
    for n in range(num_iterations):
        # Update momentum using current position
        p[n + 1] = p[n] - calc_force(x[n]) * dt
        # Update position using the newly calculated momentum (symplectic property)
        x[n + 1] = x[n] + p[n + 1] * dt
        t[n + 1] = (n + 1) * dt
        if check_oscillation_type(x[0], p[0]) == "Libration":
            if (x[n] >= 0 and x[n + 1] <=0) or (x[n] <= 0 and x[n + 1] >= 0):
                cross_points.append(t[n])
        elif check_oscillation_type(x[0], p[0]) == "Rotation":
            if 1 < n < num_iterations - 1 and p[n-1] >= p[n] and p[n] <= p[n + 1]:
                cross_points.append(t[n])

    if check_oscillation_type(x[0], p[0]) == "Libration":
        cycle_time = 2 * np.average(np.diff(cross_points))
    elif check_oscillation_type(x[0], p[0]) == "Rotation":
        cycle_time = np.average(np.diff(cross_points))
    cycle_time_err = np.std(np.diff(cross_points), ddof=1)
    frequecy = 2 * np.pi / cycle_time
    frequecy_err = 2 * np.pi * cycle_time_err / (cycle_time ** 2)
    return t, x, p, frequecy, frequecy_err

def simulate_frequencies() -> None:
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
    
    libration_energies = np.linspace(-0.9 * E_sx, E_sx * 0.9, 3)
    rotation_energies = np.linspace(E_sx * 1.1, 2*E_sx, 5)
    energies = np.append(libration_energies, rotation_energies)
    continuous_energies = np.linspace(-E_sx + 10**-3, 2 * E_sx, 1000)
    theoretical_frequencies = [calc_frequency(E) for E in continuous_energies]

    frequencies = []
    frequency_errs = []

    for E in energies:
        x_0 = 0.0
        p_0 = np.sqrt(2 * (E + omega_0**2))
        t, x, p, frequency, frequency_err = solve_numeric_symplectic(x_0, p_0, NUM_ITERATIONS, dt)
        plot_on_figure(
            x, p, f"E={E:.2f}", phase_space_fig, color=plt.cm.viridis((E + E_sx) / (3 + 5))
        )
        plot_on_figure(
            t, x, f"E={E:.2f}", trajectory_fig, color=plt.cm.viridis(((E + E_sx) / (3 + 5)))
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
    plot_on_figure(
        continuous_energies, theoretical_frequencies, "Theoretical Frequency", frequencies_fig, color='tab:orange'
    )

# simulate_frequencies()
# plt.show()


def plot_frequency_vs_release_angle():
    release_angles = np.linspace(10**-1, np.pi - 10**-2, 100)
    energies = -omega_0 ** 2 * np.cos(release_angles)
    freqs_theory = np.array([calc_frequency(E) for E in energies])
    freqs_linear_approx = np.array([calc_frequency_first_order(E) for E in energies])

    freq_energy_fig = plt.figure("Frequency vs Release Angle", figsize=(8, 5))
    ax = freq_energy_fig.gca()
    ax.set_title("Frequency vs Release Angle for a Pendulum", fontsize=14)
    ax.set_xlabel("Release Angle (deg)", fontsize=12)
    ax.set_ylabel("Frequency ($\omega$)", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)
    plot_on_figure(release_angles * 180 / np.pi, freqs_theory, label="Exact Frequency", fig=freq_energy_fig, color="tab:blue")
    plot_on_figure(release_angles * 180 / np.pi, freqs_linear_approx, label="First Order Approximation", fig=freq_energy_fig, color="tab:red")
    
    i = 0
    while np.abs(freqs_theory[i] - omega_0) / omega_0 < 0.1:
        i += 1
    ax.axvline(x=release_angles[i] * 180 / np.pi, color='tab:orange', linestyle='--', label=f'10% error ($angle={release_angles[i] * 180 / np.pi:.2f})$') # Adding a vertical line for emphasis and label
    ax.legend()

plot_frequency_vs_release_angle()
plt.show()