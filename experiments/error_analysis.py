import numpy as np
from scipy.optimize import least_squares

# Speed of light (m/s)
c = 299792458

# -------------------------
# Define Pulsars
# -------------------------
pulsars = [
    {"name": "P1", "dir": np.array([1.0, 0.0, 0.0])},
    {"name": "P2", "dir": np.array([0.0, 1.0, 0.0])},
    {"name": "P3", "dir": np.array([0.0, 0.0, 1.0])},
    {"name": "P4", "dir": np.array([-1.0, -1.0, 0.5]) / np.linalg.norm([-1.0, -1.0, 0.5])},
]

# -------------------------
# True Position
# -------------------------
true_position = np.array([1.2e6, -2.5e6, 0.8e6])

# -------------------------
# Simulation Function
# -------------------------
def simulate_observations(position, pulsars, noise_std):
    times = []
    for p in pulsars:
        direction = p["dir"]
        delay = np.dot(position, direction) / c
        noisy_delay = delay + np.random.normal(0, noise_std)
        times.append(noisy_delay)
    return np.array(times)

# -------------------------
# Residual Function
# -------------------------
def residuals(pos, pulsars, observed):
    res = []
    for i, p in enumerate(pulsars):
        direction = p["dir"]
        predicted = np.dot(pos, direction) / c
        res.append(predicted - observed[i])
    return res

# -------------------------
# Run Analysis
# -------------------------
print("=== Error vs Noise Analysis ===")

initial_guess = np.array([0.0, 0.0, 0.0])

noise_levels = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6]

for noise in noise_levels:
    errors = []

    # Run multiple trials for stability
    for _ in range(10):
        observed = simulate_observations(true_position, pulsars, noise)
        result = least_squares(residuals, initial_guess, args=(pulsars, observed))
        est = result.x
        err = np.linalg.norm(est - true_position)
        errors.append(err)

    avg_error = np.mean(errors)
    print(f"Noise: {noise:.1e} → Avg Error: {avg_error:.2f} m")
