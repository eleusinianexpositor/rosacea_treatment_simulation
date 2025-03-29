import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import List, Dict # For type hinting
# Import model and parameters
from model import rosacea_model
from parameters import parameters

# --- Simulation Settings ---
sim_duration_single = 28  # Simulate for 28 days (adjust as needed)
sim_points_per_day_single = 10
patch_load_il10_single = 10.0
output_plot_filename_single = "rosacea_single_patch_simulation.png"

# --- Initial Conditions ---
# Calculate untreated steady state to start near chronic levels
LL37_ss_init = (parameters['basal_LL37'] + parameters['k_chronic_stimulus']) / parameters['d_LL37']
NLRP3_ss_init = parameters['k_LL37_nlrp3'] * LL37_ss_init / parameters['d_NLRP3']
ProInflamm_ss_init = (parameters['k_LL37_inflamm'] * LL37_ss_init + parameters['k_nlrp3_inflamm'] * NLRP3_ss_init) / parameters['d_ProInflamm']
VEGF_ss_init = (parameters['k_LL37_vegf'] * LL37_ss_init + parameters['k_inflamm_vegf'] * ProInflamm_ss_init) / parameters['d_VEGF']
Erythema_ss_init = (parameters['k_vegf_eryth'] * VEGF_ss_init + parameters['k_inflamm_eryth'] * ProInflamm_ss_init) / parameters['d_Erythema']

# Initial conditions for UNTREATED case (starts at steady state, no patch)
y0_untreated = np.array([
    LL37_ss_init, ProInflamm_ss_init, NLRP3_ss_init, VEGF_ss_init, Erythema_ss_init,
    0.0,  # IL10_patch
    0.0,  # IL10_tissue
    0.0   # IL10_Effect
])

# Initial conditions for TREATED case (starts same, but adds patch load)
y0_treated = np.array(y0_untreated) # Copy untreated state
y0_treated[5] = patch_load_il10_single # Add the patch load at t=0 (index 5)

# --- Time Vector ---
t_span = np.linspace(0, sim_duration_single, int(sim_duration_single * sim_points_per_day_single) + 1)

# --- Solve ODEs for both scenarios ---
print("Running untreated simulation...")
sol_untreated = odeint(rosacea_model, y0_untreated, t_span, args=(parameters,))
print("Running single patch simulation...")
sol_treated = odeint(rosacea_model, y0_treated, t_span, args=(parameters,))
print("Simulations complete.")

# --- Plotting ---
print("Generating plots...")
plt.style.use('seaborn-v0_8-talk')
fig, axes = plt.subplots(3, 1, figsize=(15, 13), sharex=True)

# Plot 1: Core Inflammatory Mediators
axes[0].plot(t_span, sol_treated[:, 0], label='LL37 (Treated)', linestyle='-', color='blue', linewidth=2)
axes[0].plot(t_span, sol_untreated[:, 0], label='LL37 (Untreated)', linestyle='--', color='lightblue', linewidth=2)
axes[0].plot(t_span, sol_treated[:, 1], label='ProInflamm (Treated)', linestyle='-', color='red', linewidth=2)
axes[0].plot(t_span, sol_untreated[:, 1], label='ProInflamm (Untreated)', linestyle='--', color='salmon', linewidth=2)
axes[0].plot(t_span, sol_treated[:, 2], label='NLRP3 (Treated)', linestyle='-', color='green', linewidth=2)
axes[0].plot(t_span, sol_untreated[:, 2], label='NLRP3 (Untreated)', linestyle='--', color='lightgreen', linewidth=2)
axes[0].set_ylabel('Activity / Level')
axes[0].set_title('Core Inflammatory Mediators (Single IL-10 Patch vs. Untreated)')
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.7)
axes[0].set_ylim(bottom=0)

# Plot 2: Intervention Delivery, Effect & Vascular Factors
ax2b = axes[1].twinx()
# Primary axis (Treated only for IL-10 related, compare VEGF)
l1, = axes[1].plot(t_span, sol_treated[:, 6], label='IL-10 (Tissue)', linestyle='-', color='purple', linewidth=2) # Index 6
l2, = axes[1].plot(t_span, sol_treated[:, 7], label='IL-10 Effect', linestyle='-', color='magenta', linewidth=2) # Index 7
l3, = axes[1].plot(t_span, sol_treated[:, 3], label='VEGF (Treated)', linestyle='-', color='cyan', linewidth=2) # Index 3
l4, = axes[1].plot(t_span, sol_untreated[:, 3], label='VEGF (Untreated)', linestyle='--', color='paleturquoise', linewidth=2) # Index 3
axes[1].set_ylabel('Concentration / Activity / Effect Level')
axes[1].set_ylim(bottom=0)
# Secondary axis (Treated only)
l5, = ax2b.plot(t_span, sol_treated[:, 5], label='IL-10 (Patch)', linestyle=':', color='purple', alpha=0.7, linewidth=2) # Index 5
ax2b.set_ylabel('Amount in Patch')
ax2b.set_ylim(bottom=0)
axes[1].set_title('Intervention Delivery, Effect & Vascular Factors')
lines = [l1, l2, l3, l4, l5] # Include untreated VEGF
axes[1].legend(lines, [l.get_label() for l in lines], loc='center right')
axes[1].grid(True, linestyle=':', alpha=0.7)

# Plot 3: Clinical Outcome (Erythema)
axes[2].plot(t_span, sol_treated[:, 4], label='Erythema (Treated)', linestyle='-', color='darkred', linewidth=2.5) # Index 4
axes[2].plot(t_span, sol_untreated[:, 4], label='Erythema (Untreated)', linestyle='--', color='lightcoral', linewidth=2.5) # Index 4
axes[2].set_ylabel('Severity Score')
axes[2].set_title('Clinical Outcome: Erythema (Single Patch vs. Untreated)')
axes[2].set_xlabel('Time (days)')
axes[2].legend()
axes[2].grid(True, linestyle=':', alpha=0.7)
axes[2].set_ylim(bottom=0)

plt.tight_layout(pad=1.5)

# --- Save Plot ---
try:
    plt.savefig(output_plot_filename_single, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved successfully as '{output_plot_filename_single}'")
except Exception as e:
    print(f"\nError saving plot: {e}")

plt.show() # Display the plot

print("\nSimulation complete.")