import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any # For type hinting

# --- Core Model Definition ---
# Identical to the model used in the interval finding script

def rosacea_model(y: np.ndarray, t: float, p: Dict[str, float]) -> List[float]:
    """
    Defines the system of Ordinary Differential Equations (ODEs) for the
    rosacea inflammation model treated with an IL-10 microneedle patch.

    Includes:
    - Persistent chronic stimulus driving inflammation.
    - IL-10 delivery from a patch (release + degradation).
    - A persistent downstream effect variable for IL-10 suppression.

    Args:
        y: Array of current state variable values:
           [LL37, ProInflamm, NLRP3, VEGF, Erythema, IL10_patch, IL10_tissue, IL10_Effect]
        t: Current time (days).
        p: Dictionary of parameter values.

    Returns:
        List of derivatives (dY/dt) for each state variable.
    """
    LL37, ProInflamm, NLRP3, VEGF, Erythema, IL10_patch, IL10_tissue, IL10_Effect = y
    chronic_stimulus = p['k_chronic_stimulus']
    dLL37_dt = p['basal_LL37'] + chronic_stimulus \
               - p['d_LL37'] * LL37 \
               - p['k_il10_effect_supp_LL37'] * IL10_Effect * LL37
    dProInflamm_dt = (p['k_LL37_inflamm'] * LL37 + p['k_nlrp3_inflamm'] * NLRP3) \
                     - p['d_ProInflamm'] * ProInflamm \
                     - p['k_il10_effect_supp_inflamm'] * IL10_Effect * ProInflamm
    dNLRP3_dt = p['k_LL37_nlrp3'] * LL37 - p['d_NLRP3'] * NLRP3
    dVEGF_dt = p['k_LL37_vegf'] * LL37 + p['k_inflamm_vegf'] * ProInflamm - p['d_VEGF'] * VEGF
    dErythema_dt = p['k_vegf_eryth'] * VEGF + p['k_inflamm_eryth'] * ProInflamm - p['d_Erythema'] * Erythema
    release_IL10 = p['k_release_IL10'] * IL10_patch
    degradation_IL10_patch = p['k_degrade_patch_IL10'] * IL10_patch
    dIL10_patch_dt = -release_IL10 - degradation_IL10_patch
    dIL10_tissue_dt = release_IL10 - p['d_IL10_tissue'] * IL10_tissue
    dIL10_Effect_dt = p['k_produce_IL10_effect'] * IL10_tissue - p['d_IL10_Effect'] * IL10_Effect
    if LL37 <= 0 and dLL37_dt < 0: dLL37_dt = 0
    if ProInflamm <= 0 and dProInflamm_dt < 0: dProInflamm_dt = 0
    if NLRP3 <= 0 and dNLRP3_dt < 0: dNLRP3_dt = 0
    if VEGF <= 0 and dVEGF_dt < 0: dVEGF_dt = 0
    if Erythema <= 0 and dErythema_dt < 0: dErythema_dt = 0
    if IL10_patch <= 0: IL10_patch = 0; dIL10_patch_dt = 0
    if IL10_tissue <= 0 and dIL10_tissue_dt < 0: dIL10_tissue_dt = 0
    if IL10_Effect <= 0 and dIL10_Effect_dt < 0: dIL10_Effect_dt = 0
    return [dLL37_dt, dProInflamm_dt, dNLRP3_dt, dVEGF_dt, dErythema_dt,
            dIL10_patch_dt, dIL10_tissue_dt, dIL10_Effect_dt]

# --- Main Execution Block ---

if __name__ == "__main__":

    # --- Model Parameters (Identical to the interval finding script) ---
    parameters = {
        # Chronic Stimulus & Basal
        'k_chronic_stimulus': 0.95, # Represents persistent underlying trigger (microbial, genetic etc.). Value set high relative to LL37 decay to maintain high chronic LL37 levels (~1.2).
        'basal_LL37': 0.01,         # Represents low-level constitutive production. Assumed small compared to chronic stimulus in disease state.

        # Decay Rates (1/day) - Inverse related to half-life (t_1/2 = ln(2)/decay_rate)
        'd_LL37': 0.8,              # Assumes LL37 peptide half-life is moderate (~21 hrs).
        'd_ProInflamm': 2.5,        # Assumes rapid turnover for pro-inflammatory cytokines (~7 hr half-life).
        'd_NLRP3': 1.5,             # Represents moderate deactivation/turnover rate of the active inflammasome complex (~11 hr half-life).
        'd_VEGF': 0.6,              # Assumes VEGF has a longer half-life than inflammatory cytokines (~28 hrs).
        'd_Erythema': 0.07,         # Represents very slow clinical resolution of redness (t_1/2 ~10 days), ensuring persistence.
        'd_IL10_tissue': 2.5,       # Assumes rapid clearance/decay for IL-10 molecule in tissue (~7 hr half-life).
        'd_IL10_Effect': 0.7,       # Decay rate for the downstream effect (t_1/2 ~ 1 day), slower than IL-10 molecule but faster than initial guess.

        # Production/Activation Rates (Relative values tuned for plausible steady state)
        'k_LL37_inflamm': 1.0,      # Rate ProInflamm production is driven by LL37.
        'k_nlrp3_inflamm': 0.8,     # Rate ProInflamm production is driven by NLRP3.
        'k_LL37_nlrp3': 0.5,        # Rate LL37 activates NLRP3.
        'k_LL37_vegf': 0.2,         # Rate VEGF production is driven by LL37.
        'k_inflamm_vegf': 0.4,      # Rate VEGF production is driven by ProInflamm cytokines.
        'k_vegf_eryth': 0.15,       # Rate Erythema score increases due to VEGF.
        'k_inflamm_eryth': 0.15,    # Rate Erythema score increases due to ProInflamm cytokines.
        'k_produce_IL10_effect': 1.0, # Rate the downstream effect is generated by IL10_tissue.

        # Patch Release & Degradation (1/day)
        'k_release_IL10': 0.4,      # Release rate from patch (first-order simplification, t_1/2 ~1.7 days). Represents multi-day release goal.
        'k_degrade_patch_IL10': 0.05,# Estimated degradation rate of IL-10 within hydrogel (t_1/2 ~14 days). Educated guess.

        # Therapeutic Effects (mediated by IL10_Effect)
        'k_il10_effect_supp_LL37': 1.8,     # Strength effect suppresses LL37. Tuned for potency.
        'k_il10_effect_supp_inflamm': 1.8,  # Strength effect suppresses ProInflamm. Tuned for potency.
    }

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