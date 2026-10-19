import numpy as np
import matplotlib.pyplot as plt
from config import rng, N_NEURONS, NOISE_STD_POP, TRUE_ANGLES_DEG
from utils import wrap_angle_deg, angular_error_deg

pref_dirs_deg = np.linspace(0, 360, N_NEURONS, endpoint=False)
pref_dirs_rad = np.deg2rad(pref_dirs_deg)
pref_vecs = np.column_stack([np.cos(pref_dirs_rad), np.sin(pref_dirs_rad)])

true_angles_rad = np.deg2rad(TRUE_ANGLES_DEG)

# ------------------------------------------------------------
# TODO 1: Define cosine tuning function
# Hint:
# return max(0, cos(theta - phi))
# Use np.maximum so negative values become 0
# ------------------------------------------------------------
def cosine_tuning(theta, phi):
    return np.maximum(0, np.cos(theta - phi))

# ------------------------------------------------------------
# TODO 2: Decode each movement direction
# For each theta:
# - compute noiseless rates using cosine_tuning
# - add Gaussian noise
# - clip rates to be >= 0
# - compute population vector
# - decode angle using atan2
# - wrap decoded angle to [0, 360)
# ------------------------------------------------------------
decoded_angles_deg = []
all_rates_matrix = []

for theta in true_angles_rad:
    rates = cosine_tuning(theta, pref_dirs_rad)
    rates += rng.normal(0, NOISE_STD_POP, N_NEURONS)
    rates = np.clip(rates, 0, None)

    pv = np.sum(rates[:, None] * pref_vecs, axis=0)
    decoded = np.rad2deg(np.arctan2(pv[1], pv[0]))
    decoded = wrap_angle_deg(decoded)

    decoded_angles_deg.append(decoded)
    all_rates_matrix.append(rates)

decoded_angles_deg = np.array(decoded_angles_deg)
all_rates_matrix = np.array(all_rates_matrix)

# ------------------------------------------------------------
# TODO 3: Compute angular decoding error
# Hint:
# use angular_error_deg(decoded_angles_deg, TRUE_ANGLES_DEG)
# then compute mean absolute error and max absolute error
# ------------------------------------------------------------
angular_errors = angular_error_deg(decoded_angles_deg, TRUE_ANGLES_DEG)
mae = np.mean(np.abs(angular_errors))
max_err = np.max(np.abs(angular_errors))

print(f"Mean absolute error : {mae:.2f} degrees")
print(f"Max absolute error  : {max_err:.2f} degrees")

# ------------------------------------------------------------
# TODO 4: Plot decoding results
# Left: polar plot of true and decoded directions
# Right: bar chart of angular errors
# ------------------------------------------------------------
fig = plt.figure(figsize=(13, 5))

ax_polar = fig.add_subplot(1, 2, 1, projection='polar')

for i in range(len(TRUE_ANGLES_DEG)):
    t = np.deg2rad(TRUE_ANGLES_DEG[i])
    d = np.deg2rad(decoded_angles_deg[i])
    ax_polar.plot([t, t], [0, 1], 'b-', alpha=0.6, label='True' if i == 0 else None)
    ax_polar.plot([d, d], [0, 0.8], 'r--', alpha=0.8, label='Decoded' if i == 0 else None)
ax_polar.legend(loc='lower left')

ax_polar.set_title("Population Vector Decoding")

ax_err = fig.add_subplot(1, 2, 2)

ax_err.bar(TRUE_ANGLES_DEG, angular_errors, width=10)
ax_err.axhline(0, color='k')

ax_err.set_xlabel("True angle (°)")
ax_err.set_ylabel("Error (°)")
ax_err.set_title("Decoding Error per Direction")
ax_err.grid(True)

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# TODO 5: Plot population activity heatmap
# Hint:
# rows = true angles
# cols = neurons
# use cmap='viridis'
# ------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(9, 6))

im = ax2.imshow(all_rates_matrix, aspect='auto', cmap='viridis')

ax2.set_xlabel("Neuron preferred direction (°)")
ax2.set_ylabel("True movement angle (°)")
ax2.set_title("Population Activity Heatmap")

ax2.set_xticks(np.arange(len(pref_dirs_deg)))
ax2.set_xticklabels([f"{int(d)}" for d in pref_dirs_deg])

ax2.set_yticks(np.arange(len(TRUE_ANGLES_DEG)))
ax2.set_yticklabels([f"{int(a)}" for a in TRUE_ANGLES_DEG])

cbar = plt.colorbar(im, ax=ax2)
cbar.set_label("Firing rate")

plt.tight_layout()
plt.show()