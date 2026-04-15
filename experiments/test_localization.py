import numpy as np
from scipy.optimize import least_squares

# Speed of light (m/s)
c = 299792458

# -------------------------
# Define Pulsars
# -------------------------
# Unit direction vectors (must be normalized)
pulsars = [
    {"name": "P1", "dir": np.array([1.0, 0.0, 0.0])},
    {"name": "P2", "dir": np.array([0.0, 1.0, 0.0])},
    {"name": "P3", "dir": np.array([0.0, 0.0, 1.0])},
    {"name": "P4", "dir": np.array([-1.0, -1.0, 0.5]) / np.linalg.norm([-1.0, -1.0, 0.5])},
]

# -------------------------
# True Position (unknown to solver)
# -------------------------
true_position = np.array([1.2e6, -2.5e6, 0.8e6])  # meters

# -------------------------
# Simulate Observed Timing
# -------------------------
def simulate_observations(position, pulsars, noise_std=1e-9):
    times = []
    for p in pulsars:
        direction = p["dir"]
        delay = np.dot(position, direction) / c
        noisy_delay = delay + np.random.normal(0, noise_std)
        times.append(noisy_delay)
    return np.array(times)

observed_times = simulate_observations(true_position, pulsars)

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
# Solve for Position
# -------------------------
initial_guess = np.array([0.0, 0.0, 0.0])

result = least_squares(residuals, initial_guess, args=(pulsars, observed_times))
estimated_position = result.x

# -------------------------
# Results
# -------------------------
error = np.linalg.norm(estimated_position - true_position)

print("=== Localization Test ===")
print(f"True Position:      {true_position}")
print(f"Estimated Position: {estimated_position}")
print(f"Error (meters):     {error:.3f}")
