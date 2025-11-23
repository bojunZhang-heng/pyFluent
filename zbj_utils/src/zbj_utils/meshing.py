import math
import textwrap

# --------------------------------------- Usage --------------------------------------- 
"""
import zbj_utils.meshing as meshing
meshing.print_yplus_info()
meshing.calculate_delta_y(10.0, 1.225, 1.8e-5, 1.0, 30.0)
"""
# --- 1. Parameter Info Function ---
def print_yplus_info():
    """
    Prints the required parameters, units, and order for the main calculation function.
    """
    print("=" * 70)
    print("YPLUS MESH THICKNESS CALCULATOR - PARAMETER INFO")
    print("=" * 70)
    
    # Define parameter list
    parameters = [
        ("U_f", "Free stream velocity", "m/s"),
        ("rho", "Density", "kg/m^3"),
        ("mu", "Dynamic viscosity", "Pa·s"),
        ("L", "Characteristic Length", "m"),
        ("Y+", "Target Dimensionless distance", "dimensionless")
    ]
    
    print("The main calculation function requires 5 parameters:")
    
    # Print formatted parameter info
    for name, desc, unit in parameters:
        print(f"  - {name:<12}: {desc:<25} (Unit: {unit})")
    
    print("\nExample Call: calculate_delta_y(10.0, 1.225, 1.8e-5, 1.0, 30.0)")
    print("=" * 70)


# --- 2. Main Calculation Function ---
def calculate_delta_y(U_f, rho, mu, L, y_plus_target):
    """
    Calculates the Reynolds number, friction coefficient, wall shear stress, 
    friction velocity, and the first layer mesh thickness (Delta Y).
    
    Parameters:
    U_f (float): Free stream velocity (m/s)
    rho (float): Density (kg/m^3)
    mu (float): Dynamic viscosity (Pa·s)
    L (float): Characteristic length (m)
    y_plus_target (float): Target Y+ value
    """
    
    # Check for valid parameters
    if any(p <= 0 for p in [U_f, rho, mu, L, y_plus_target]):
        print("ERROR: All physical parameters must be positive.")
        return

    # --- 1. Calculate Reynolds Number (Re) ---
    Re = rho * U_f * L / mu
    
    # --- 2. Calculate Friction Coefficient (Cf) using 5 empirical formulas ---
    # We calculate 5 common values and select the maximum (most conservative) value.
    
    try:
        # 1. Blasius Laminar Boundary Layer
        Cf_1 = 0.664 / math.sqrt(Re)
        
        # 2. Prandtl-Schlichting Turbulent Formula
        Cf_2 = 0.0592 / (Re ** (1/5))
        
        # 3. Power Law Turbulent (1/7th Power Law)
        Cf_3 = 0.026 / (Re ** (1/7))
        
        # 4. Karman-Schoenherr Log Law (Wide range Turbulent approximation)
        Cf_4 = (2 * math.log10(Re) - 0.65)**(-2.3)
        
        # 5. White's Relation (Common Turbulent approximation)
        Cf_5 = 0.455 / (math.log10(Re))**2.58
        
        Cf_values = [Cf_1, Cf_2, Cf_3, Cf_4, Cf_5]
        
    except ValueError as e:
        print(f"ERROR: A mathematical error occurred during friction coefficient calculation (e.g., log(Re)): {e}")
        return

    # --- 3. Determine Conservative and Average Cf ---
    Cf_conservative = max(Cf_values)
    Cf_average = sum(Cf_values) / len(Cf_values)
    
    # --- 4. Calculate Wall Shear Stress, Friction Velocity, and Delta Y ---
    
    results = {}
    
    # Helper function to calculate derived values based on Cf
    def calculate_derived_values(Cf, label):
        tau_w = 0.5 * Cf * rho * (U_f ** 2)
        u_tau = math.sqrt(tau_w / rho)
        delta_y = (y_plus_target * mu) / (rho * u_tau)
        return {
            f"tau_w_{label}": tau_w,
            f"u_tau_{label}": u_tau,
            f"delta_y_{label}": delta_y
        }

    # Calculate conservative values
    results.update(calculate_derived_values(Cf_conservative, "conservative"))
    
    # Calculate average values
    results.update(calculate_derived_values(Cf_average, "average"))
    
    # --- 5. Formatted Output ---
    print("\n" + "=" * 70)
    print("YPLUS MESH THICKNESS CALCULATION RESULTS")
    print("=" * 70)
    
    # Section 1: Inputs and Re
    print("--- BASE PARAMETERS AND REYNOLDS NUMBER ---")
    print(f"  - Target Y+: {y_plus_target:.2f}")
    print(f"  - Freestream Velocity (U_f): {U_f:.2f} m/s")
    print(f"  - Reynolds Number (Re): {Re:.2e}")
    
    # Section 2: Friction Coefficients
    print("\n--- FRICTION COEFFICIENT (C_f) ESTIMATION ---")
    print("-" * 35)
    for i, Cf in enumerate(Cf_values, 1):
        print(f"  - Method {i}: {Cf:.5e}")
    
    print(f"\n- Most Conservative C_f (Maximum): {Cf_conservative:.5e}")
    print(f"- Average C_f (Mean): {Cf_average:.5e}")
    
    # Section 3: Core Calculation Comparison Table
    print("\n--- CORE CALCULATION COMPARISON ---")
    print("-" * 70)
    print(f"| {'Calculation':<15} | {'Conservative Value (Max C_f)':<27} | {'Average Value (Avg C_f)':<25} |")
    print("|" + "-"*16 + "+" + "-"*29 + "+" + "-"*27 + "|")
    
    # Wall Shear Stress
    tau_w_cons = results["tau_w_conservative"]
    tau_w_avg = results["tau_w_average"]
    print(f"| {'Shear Stress (tau_w)':<15} | {tau_w_cons:.4f} Pa{textwrap.indent(textwrap.dedent(''), 25)} | {tau_w_avg:.4f} Pa{textwrap.indent(textwrap.dedent(''), 25)} |")
    
    # Friction Velocity
    u_tau_cons = results["u_tau_conservative"]
    u_tau_avg = results["u_tau_average"]
    print(f"| {'Friction Velocity (u_tau)':<15} | {u_tau_cons:.4f} m/s{textwrap.indent(textwrap.dedent(''), 24)} | {u_tau_avg:.4f} m/s{textwrap.indent(textwrap.dedent(''), 24)} |")
    
    # First Layer Thickness
    delta_y_cons = results["delta_y_conservative"] * 1000 # Convert to mm
    delta_y_avg = results["delta_y_average"] * 1000 # Convert to mm
    
    print("\n--- FIRST LAYER THICKNESS (Delta Y) ---")
    print("-" * 70)
    print(f"| {'Conservative Delta Y (Recommended)':<36} | {delta_y_cons:.4f} mm ({delta_y_cons*1e3:.1f} um)")
    print(f"| {'Average Delta Y':<36} | {delta_y_avg:.4f} mm ({delta_y_avg*1e3:.1f} um)")
    
    print("\nSummary: The conservative Delta Y (smallest value) should be used for safety margin and mesh quality.")
    print("=" * 70)
