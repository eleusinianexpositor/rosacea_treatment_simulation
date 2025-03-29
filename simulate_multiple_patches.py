import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import time
from typing import List, Dict, Tuple, Any # For type hinting
# Import model and parameters
from model import rosacea_model
from parameters import parameters

# --- Simulation Execution Function ---

def simulate_repeated_patches(interval_days: int,
                              y0_initial: np.ndarray,
                              t_total_sim: float,
                              pts_per_day: int,
                              params_sim: Dict[str, float],
                              patch_load: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs the ODE simulation with patches reapplied periodically.

    Args:
        interval_days: The number of days between patch applications.
        y0_initial: The initial state vector (before first patch).
        t_total_sim: Total duration of the simulation (days).
        pts_per_day: Number of simulation points per day for output.
        params_sim: Dictionary of model parameters.
        patch_load: The amount of IL-10 loaded into a fresh patch.

    Returns:
        A tuple containing:
        - t_results: NumPy array of time points.
        - sol_results: NumPy array of corresponding solution states.
    """
    all_t = [0.0]
    y0_current = np.array(y0_initial) # Make a mutable copy
    # Apply the *first* patch at time 0
    y0_current[5] = patch_load # Index 5 is IL10_patch
    all_sol = [y0_current]

    num_intervals = int(np.ceil(t_total_sim / interval_days))
    points_per_interval = max(2, int(interval_days * pts_per_day)) # Ensure at least 2 points

    for i in range(num_intervals):
        t_start = i * interval_days
        t_end = (i + 1) * interval_days
        # Adjust last interval if needed
        if t_end > t_total_sim:
            t_end = t_total_sim
            points_this_interval = max(2, int((t_end - t_start) * pts_per_day))
            if t_end <= t_start: break # Avoid zero-length intervals
        else:
            points_this_interval = points_per_interval

        # Time points for this specific interval
        t_interval = np.linspace(t_start, t_end, points_this_interval)

        # Initial condition for this interval is the end state of the last one
        y0_interval = all_sol[-1]

        # Reapply patch at the beginning of intervals > 0
        if i > 0:
            y0_interval = np.array(y0_interval) # Ensure mutable copy
            y0_interval[5] = patch_load # Reset IL10_patch (index 5)

        # Solve for the current interval
        sol_interval = odeint(rosacea_model, y0_interval, t_interval, args=(params_sim,))

        # Store results, skipping the first point (overlap with previous end point)
        all_t.extend(t_interval[1:])
        all_sol.extend(sol_interval[1:])

    return np.array(all_t), np.array(all_sol)

# --- Evaluation Function ---

def evaluate_intervals(test_intervals: range,
                       y0: np.ndarray,
                       t_total_eval: float,
                       pts_per_day_eval: int,
                       params_eval: Dict[str, float],
                       patch_load_eval: float) -> List[Dict[str, Any]]:
    """
    Simulates multiple patch application intervals and calculates metrics.

    Args:
        test_intervals: A range of intervals (in days) to test.
        y0: Initial state vector (before first patch).
        t_total_eval: Total simulation time for evaluation.
        pts_per_day_eval: Points per day for simulation.
        params_eval: Model parameters dictionary.
        patch_load_eval: IL-10 load per patch.

    Returns:
        A list of dictionaries, each containing results for one interval.
    """
    results = []
    late_phase_start_time = t_total_eval / 2.0 # Evaluate second half

    print("\nEvaluating different reapplication intervals...")
    print("-" * 40)
    start_time_eval = time.time()

    for interval in test_intervals:
        print(f"  Simulating interval: {interval} days...")
        t_sim, sol_sim = simulate_repeated_patches(
            interval, y0, t_total_eval, pts_per_day_eval, params_eval, patch_load_eval
        )

        late_phase_indices = np.where(t_sim >= late_phase_start_time)[0]

        if len(late_phase_indices) > 1: # Need at least 2 points for stats
            # Calculate metrics over the late phase
            erythema_late = sol_sim[late_phase_indices, 4] # Index 4 = Erythema
            ll37_late = sol_sim[late_phase_indices, 0]     # Index 0 = LL37

            mean_erythema = np.mean(erythema_late)
            max_erythema = np.max(erythema_late)
            final_erythema = erythema_late[-1]
            ll37_amplitude = np.max(ll37_late) - np.min(ll37_late)

            results.append({
                'interval': interval,
                'mean_erythema': mean_erythema,
                'max_erythema': max_erythema,
                'final_erythema': final_erythema,
                'll37_amplitude': ll37_amplitude
            })
            print(f"    -> Mean Ery={mean_erythema:.3f}, Max Ery={max_erythema:.3f}, LL37 Amp={ll37_amplitude:.3f}")
        else:
            print(f"    -> Simulation too short or late phase too small to evaluate.")

    end_time_eval = time.time()
    print("-" * 40)
    print(f"Evaluation finished in {end_time_eval - start_time_eval:.2f} seconds.")
    return results

# --- Result Analysis and Plotting ---

def analyze_and_plot(results: List[Dict[str, Any]],
                       y0: np.ndarray,
                       t_total_final: float,
                       pts_per_day_final: int,
                       params_final: Dict[str, float],
                       patch_load_final: float,
                       save_filename: str = "rosacea_simulation_plot.png"):
    """
    Analyzes evaluation results, selects an interval, runs the final
    simulation, plots the results, and saves the plot.

    Args:
        results: List of dictionaries from evaluate_intervals.
        y0: Initial state vector.
        t_total_final: Total time for the final simulation run.
        pts_per_day_final: Points per day for the final run.
        params_final: Model parameters dictionary.
        patch_load_final: IL-10 load per patch.
        save_filename: Filename to save the plot image.
    """
    print("\n--- Interval Evaluation Summary ---")
    print("Interval | Mean Erythema | Max Erythema | LL37 Amplitude")
    print(" (days)  |  (late phase) | (late phase) | (late phase) ")
    print("---------|---------------|--------------|---------------")
    if not results:
        print(" No results to display.")
        plot_interval = None
    else:
        results.sort(key=lambda x: x['interval'])
        for res in results:
            print(f"   {res['interval']:<5} |     {res['mean_erythema']:.3f}     |    {res['max_erythema']:.3f}    |     {res['ll37_amplitude']:.3f}")

        # --- Selection Logic: Choose interval with lowest mean erythema ---
        # (Can be easily changed to prioritize lowest max, lowest amplitude, etc.)
        best_result = min(results, key=lambda x: x['mean_erythema'])
        plot_interval = best_result['interval']
        print(f"\nSelected interval for plotting (lowest mean erythema): {plot_interval} days")
        print("(Review table to choose differently based on priorities)")

    # --- Simulate and Plot with the Selected Interval ---
    if plot_interval is not None:
        print(f"\nRunning final simulation with {plot_interval}-day interval...")
        t_final, sol_final = simulate_repeated_patches(
            plot_interval, y0, t_total_final, pts_per_day_final, params_final, patch_load_final
        )

        # --- Plotting Code ---
        print("Generating plots...")
        plt.style.use('seaborn-v0_8-talk')
        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        late_phase_start_time = t_total_final / 2.0

        # Plot 1: Core Inflammatory Mediators
        axes[0].plot(t_final, sol_final[:, 0], label='LL37', linestyle='-', color='blue', linewidth=2)
        axes[0].plot(t_final, sol_final[:, 1], label='ProInflamm', linestyle='-', color='red', linewidth=2)
        axes[0].plot(t_final, sol_final[:, 2], label='NLRP3', linestyle='-', color='green', linewidth=2)
        axes[0].set_ylabel('Activity / Level')
        axes[0].set_title(f'Core Inflammatory Mediators (Repeated IL-10 Patching: {plot_interval} days)')
        axes[0].legend()
        axes[0].grid(True, linestyle=':', alpha=0.7)
        axes[0].set_ylim(bottom=0)

        # Plot 2: Intervention Delivery, Effect & Vascular Factors
        ax2b = axes[1].twinx()
        l1, = axes[1].plot(t_final, sol_final[:, 6], label='IL-10 (Tissue)', linestyle='-', color='purple', linewidth=2) # Index 6
        l2, = axes[1].plot(t_final, sol_final[:, 7], label='IL-10 Effect', linestyle='-', color='magenta', linewidth=2) # Index 7
        l3, = axes[1].plot(t_final, sol_final[:, 3], label='VEGF', linestyle='-', color='cyan', linewidth=2) # Index 3
        axes[1].set_ylabel('Concentration / Activity / Effect Level')
        axes[1].set_ylim(bottom=0)
        l5, = ax2b.plot(t_final, sol_final[:, 5], label='IL-10 (Patch)', linestyle=':', color='purple', alpha=0.7, linewidth=2) # Index 5
        ax2b.set_ylabel('Amount in Patch')
        ax2b.set_ylim(bottom=0)
        axes[1].set_title('Intervention Delivery, Effect & Vascular Factors')
        lines = [l1, l2, l3, l5]
        axes[1].legend(lines, [l.get_label() for l in lines], loc='center right')
        axes[1].grid(True, linestyle=':', alpha=0.7)

        # Plot 3: Clinical Outcome (Erythema)
        axes[2].plot(t_final, sol_final[:, 4], label=f'Erythema ({plot_interval}-day interval)', linestyle='-', color='darkred', linewidth=2.5) # Index 4
        late_phase_indices_final = np.where(t_final >= late_phase_start_time)[0]
        mean_erythema_final = 0.0 # Default value
        if len(late_phase_indices_final) > 0:
            mean_erythema_final = np.mean(sol_final[late_phase_indices_final, 4])
            axes[2].axhline(mean_erythema_final, color='grey', linestyle='--', label=f'Mean (t>{late_phase_start_time:.0f}d)={mean_erythema_final:.2f}')

        axes[2].set_ylabel('Severity Score')
        axes[2].set_title('Clinical Outcome: Erythema')
        axes[2].set_xlabel('Time (days)')
        num_apps = int(np.ceil(t_total_final / plot_interval))
        patch_applied_label_added = False
        for i in range(num_apps):
            app_time = i * plot_interval
            if app_time <= t_final[-1]:
                label = ""
                if not patch_applied_label_added:
                   label = 'Patch Applied'
                   patch_applied_label_added = True
                axes[0].axvline(app_time, color='grey', linestyle=':', alpha=0.6)
                axes[1].axvline(app_time, color='grey', linestyle=':', alpha=0.6)
                axes[2].axvline(app_time, color='grey', linestyle=':', alpha=0.6, label=label)

        handles, labels = axes[2].get_legend_handles_labels()
        axes[2].legend(handles, labels)
        axes[2].grid(True, linestyle=':', alpha=0.7)
        axes[2].set_ylim(bottom=0)

        plt.tight_layout(pad=1.5)

        # --- Save Plot ---
        try:
            plt.savefig(save_filename, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved successfully as '{save_filename}'")
        except Exception as e:
            print(f"\nError saving plot: {e}")

        plt.show() # Display the plot

        print(f"\nFinal Erythema at day {t_final[-1]:.1f} with {plot_interval}-day interval: {sol_final[-1, 4]:.4f}")

    else:
        print("\nNo suitable interval selected for plotting.")

# --- Simulation Settings ---
sim_duration = 60           # Total simulation time (days)
sim_points_per_day = 100     # Simulation time resolution
patch_load_il10 = 10.0      # Amount of IL-10 in a fresh patch (arbitrary units)
intervals_to_test = range(3, 11) # Test intervals from 3 to 10 days
output_plot_filename = "rosacea_IL10_simulation.png" # File to save plot

# --- Initial Conditions (start near untreated steady state) ---
LL37_ss_init = (parameters['basal_LL37'] + parameters['k_chronic_stimulus']) / parameters['d_LL37']
NLRP3_ss_init = parameters['k_LL37_nlrp3'] * LL37_ss_init / parameters['d_NLRP3']
ProInflamm_ss_init = (parameters['k_LL37_inflamm'] * LL37_ss_init + parameters['k_nlrp3_inflamm'] * NLRP3_ss_init) / parameters['d_ProInflamm']
VEGF_ss_init = (parameters['k_LL37_vegf'] * LL37_ss_init + parameters['k_inflamm_vegf'] * ProInflamm_ss_init) / parameters['d_VEGF']
Erythema_ss_init = (parameters['k_vegf_eryth'] * VEGF_ss_init + parameters['k_inflamm_eryth'] * ProInflamm_ss_init) / parameters['d_Erythema']

initial_conditions = np.array([
    LL37_ss_init, ProInflamm_ss_init, NLRP3_ss_init, VEGF_ss_init, Erythema_ss_init,
    0.0,  # IL10_patch (starts empty before first application)
    0.0,  # IL10_tissue
    0.0   # IL10_Effect (starts at zero)
])

# --- Run Evaluation and Plotting ---
evaluation_results = evaluate_intervals(
    test_intervals=intervals_to_test,
    y0=initial_conditions,
    t_total_eval=sim_duration,
    pts_per_day_eval=sim_points_per_day,
    params_eval=parameters,
    patch_load_eval=patch_load_il10
)

analyze_and_plot(
    results=evaluation_results,
    y0=initial_conditions,
    t_total_final=sim_duration,
    pts_per_day_final=sim_points_per_day,
    params_final=parameters,
    patch_load_final=patch_load_il10,
    save_filename=output_plot_filename
)

print("\nSimulation complete.")