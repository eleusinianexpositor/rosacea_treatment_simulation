import numpy as np
from typing import List, Dict # For type hinting

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
    # Unpack state variables for clarity
    LL37, ProInflamm, NLRP3, VEGF, Erythema, IL10_patch, IL10_tissue, IL10_Effect = y

    # --- Calculate Rates of Change (Derivatives) ---

    # Chronic stimulus drives inflammation upstream
    chronic_stimulus = p['k_chronic_stimulus']

    # LL37 Dynamics: Driven by stimulus, decays, suppressed by IL-10 effect
    dLL37_dt = p['basal_LL37'] + chronic_stimulus \
               - p['d_LL37'] * LL37 \
               - p['k_il10_effect_supp_LL37'] * IL10_Effect * LL37

    # ProInflamm Dynamics: Driven by LL37 & NLRP3, decays, suppressed by IL-10 effect
    dProInflamm_dt = (p['k_LL37_inflamm'] * LL37 + p['k_nlrp3_inflamm'] * NLRP3) \
                     - p['d_ProInflamm'] * ProInflamm \
                     - p['k_il10_effect_supp_inflamm'] * IL10_Effect * ProInflamm

    # NLRP3 Dynamics: Activated by LL37, decays
    dNLRP3_dt = p['k_LL37_nlrp3'] * LL37 - p['d_NLRP3'] * NLRP3

    # VEGF Dynamics: Driven by LL37 & ProInflamm, decays
    dVEGF_dt = p['k_LL37_vegf'] * LL37 + p['k_inflamm_vegf'] * ProInflamm - p['d_VEGF'] * VEGF

    # Erythema Dynamics: Driven by VEGF & ProInflamm, decays slowly
    dErythema_dt = p['k_vegf_eryth'] * VEGF + p['k_inflamm_eryth'] * ProInflamm - p['d_Erythema'] * Erythema

    # IL-10 Patch Dynamics: Depleted by release and degradation
    release_IL10 = p['k_release_IL10'] * IL10_patch
    degradation_IL10_patch = p['k_degrade_patch_IL10'] * IL10_patch
    dIL10_patch_dt = -release_IL10 - degradation_IL10_patch

    # IL-10 Tissue Dynamics: Gains from release, decays
    dIL10_tissue_dt = release_IL10 - p['d_IL10_tissue'] * IL10_tissue

    # IL-10 Effect Dynamics: Produced by IL-10 tissue, decays slowly
    dIL10_Effect_dt = p['k_produce_IL10_effect'] * IL10_tissue - p['d_IL10_Effect'] * IL10_Effect

    # --- Non-Negativity Constraints (Numerical Stability) ---
    # Prevent variables from becoming negative due to numerical integration steps
    if LL37 <= 0 and dLL37_dt < 0: dLL37_dt = 0
    if ProInflamm <= 0 and dProInflamm_dt < 0: dProInflamm_dt = 0
    if NLRP3 <= 0 and dNLRP3_dt < 0: dNLRP3_dt = 0
    if VEGF <= 0 and dVEGF_dt < 0: dVEGF_dt = 0
    if Erythema <= 0 and dErythema_dt < 0: dErythema_dt = 0
    if IL10_patch <= 0: IL10_patch = 0; dIL10_patch_dt = 0 # Stop depletion if empty
    if IL10_tissue <= 0 and dIL10_tissue_dt < 0: dIL10_tissue_dt = 0
    if IL10_Effect <= 0 and dIL10_Effect_dt < 0: dIL10_Effect_dt = 0

    return [dLL37_dt, dProInflamm_dt, dNLRP3_dt, dVEGF_dt, dErythema_dt,
            dIL10_patch_dt, dIL10_tissue_dt, dIL10_Effect_dt]