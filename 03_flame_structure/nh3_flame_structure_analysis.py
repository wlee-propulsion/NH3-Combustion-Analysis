"""
NH3 Flame Structure Analysis
Laminar flame speed and flame thickness characterization for ammonia-air mixtures
Transport model comparison and flame structure analysis
Author: Wonhyeong Lee
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MECHANISM = (SCRIPT_DIR.parent / "MEI_2019.yaml").as_posix()  # NH3 mechanism
FUEL = "NH3"
CONDITIONS = {
    'phi_range': [0.6, 0.8, 1.0, 1.2],  # Equivalence ratio range
    'pressure': 1.0,  # Pressure [bar]
    'temperature_unburned': 800.0,  # Unburned gas temperature [K]
    'domain_width': 0.08,  # Computational domain width [m]
    'transport_models': ["mixture-averaged", "multicomponent"],  # Transport models
    'enable_soret': False  # Thermal diffusion effects
}

def get_output_dir():
    out = SCRIPT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def get_velocity_at_inlet(flame):
    """
    Get inlet velocity with Cantera version compatibility
    """
    if hasattr(flame, "velocity"): # Cantera 2.x
        return float(flame.velocity[0])
    if hasattr(flame, "u"): # Cantera 3.x
        return float(flame.u[0])
    # Fallback method
    return float(flame.value("velocity", 0))


def get_species_profile(flame, species_name):
    """
    Safely retrieve species mole fraction profile
    Returns zero array if species not found
    """
    gas = flame.gas
    try:
        species_index = gas.species_index(species_name)
        if species_index >= 0:
            return flame.X[species_index, :]
    except Exception:
        pass
    return np.zeros_like(flame.grid)


def safe_log10(x, floor=1e-300):
    """
    Safe logarithm calculation avoiding NaN and infinity
    """
    x = np.asarray(x)
    x = np.nan_to_num(x, nan=floor, neginf=floor, posinf=floor)
    x = np.clip(x, floor, None)
    return np.log10(x)


# =============================================================================
# FLAME ANALYSIS FUNCTIONS
# =============================================================================
def calculate_flame_speed(mechanism, fuel, phi, pressure_bar, T_unburned, transport, width):
    """
    Calculate laminar flame speed using free flame solver

    Parameters:
        mechanism: Cantera mechanism file path
        fuel: Fuel species name
        phi: Equivalence ratio
        pressure_bar: Pressure [bar]
        T_unburned: Unburned gas temperature [K]
        transport: Transport model ("mixture-averaged" or "multicomponent")
        width: Domain width [m]

    Returns:
        tuple: (flame_speed, flame_object) or (None, None) if failed
    """
    try:
        # Setup gas mixture
        gas = ct.Solution(mechanism, "gas")
        gas.set_equivalence_ratio(phi, f"{fuel}:1", "O2:1, N2:3.76")
        gas.TP = T_unburned, pressure_bar * 1e5
        gas.transport_model = transport

        # Create free flame
        flame = ct.FreeFlame(gas, width=width)
        flame.set_refine_criteria(ratio=3.0, slope=0.06, curve=0.12)

        if CONDITIONS['enable_soret']:
            flame.soret_enabled = True

        # Staged solution for better convergence
        flame.energy_enabled = False
        flame.solve(loglevel=0, auto=True)
        flame.energy_enabled = True
        flame.solve(loglevel=0, auto=True)

        # Extract flame speed
        flame_speed = get_velocity_at_inlet(flame)
        return flame_speed, flame

    except Exception as e:
        print(f"Error for $\phi$={phi}, transport={transport}: {e}")
        return None, None


def calculate_thermal_thickness(flame):
    """
    Calculate thermal flame thickness using temperature gradient method
    delta_T = (T_burned - T_unburned) / max(dT/dx)
    """
    grid = flame.grid
    temperature = flame.T
    dT_dx = np.gradient(temperature, grid)
    T_unburned, T_burned = temperature[0], temperature[-1]
    return (T_burned - T_unburned) / max(np.max(dT_dx), 1e-30)


def calculate_flame_metrics(flame):
    """
    Calculate dimensionless flame coordinate system metrics
    Returns normalized coordinate kci = (x - x*) / delta_T
    """
    grid = flame.grid
    temperature = flame.T
    dT_dx = np.gradient(temperature, grid)

    # Find flame front position (maximum temperature gradient)
    flame_front_idx = int(np.argmax(dT_dx))
    x_star = grid[flame_front_idx]

    # Calculate thermal thickness
    T_unburned, T_burned = temperature[0], temperature[-1]
    delta_T = (T_burned - T_unburned) / max(dT_dx[flame_front_idx], 1e-30)

    # Normalized coordinate
    xi = (grid - x_star) / max(delta_T, 1e-12)

    # 10-90% temperature rise zone
    T_10 = T_unburned + 0.10 * (T_burned - T_unburned)
    T_90 = T_unburned + 0.90 * (T_burned - T_unburned)
    idx_10 = int(np.argmin(np.abs(temperature - T_10)))
    idx_90 = int(np.argmin(np.abs(temperature - T_90)))

    return xi, x_star, delta_T, (grid[idx_10], grid[idx_90])


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_flame_profiles(flame_mix, flame_multi, phi):
    """
    Plot normalized flame profiles comparing transport models
    """
    # Calculate metrics for both flames
    xi_mix, x_star_mix, delta_mix, (z10_mix, z90_mix) = calculate_flame_metrics(flame_mix)
    T_mix = flame_mix.T
    OH_mix = get_species_profile(flame_mix, "OH")

    xi_multi, x_star_multi, delta_multi, (z10_multi, z90_multi) = calculate_flame_metrics(flame_multi)
    T_multi = flame_multi.T
    OH_multi = get_species_profile(flame_multi, "OH")

    # Create plot with dual y-axes
    fig, ax_temp = plt.subplots()
    ax_species = ax_temp.twinx()

    # Temperature profiles
    ax_temp.plot(xi_mix, T_mix, linewidth=2, label="T, Mix-avg")
    ax_temp.plot(xi_multi, T_multi, linewidth=2, linestyle="--", label="T, Multi")
    ax_temp.set_xlabel(r"Normalized coordinate $\xi=(x-x^*)/\delta_T$")
    ax_temp.set_ylabel("Temperature [K]")

    # OH profiles (log scale)
    ax_species.plot(
        xi_mix,
        safe_log10(OH_mix),
        color='red',
        label=r'$\log_{10}(\mathrm{OH})$, Mix-avg'
    )
    ax_species.plot(
        xi_multi,
        safe_log10(OH_multi),
        color='red',
        linestyle="--",
        label=r'$\log_{10}(\mathrm{OH})$, Multi'
    )

    ax_species.set_ylabel(r'$\log_{10}$ mole fraction, OH')

    # 10-90% temperature rise zones
    ax_temp.axvspan((z10_mix - x_star_mix) / delta_mix, (z90_mix - x_star_mix) / delta_mix,
                    color='black', alpha=0.1, label="10–90% T-rise, Mix-avg")
    ax_temp.axvspan((z10_multi - x_star_multi) / delta_multi, (z90_multi - x_star_multi) / delta_multi,
                    color='gray', alpha=0.1)

    # Formatting
    ax_temp.set_xlim(-4, 6)
    ax_temp.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax_temp.get_legend_handles_labels()
    lines2, labels2 = ax_species.get_legend_handles_labels()
    ax_temp.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    title = (f"NH3–Air Flame Structure, $\phi$={phi:.2f}\n"
             r"$x^*$ at max(dT/dx), $\delta_T$ = thermal thickness")
    ax_temp.set_title(title)

    plt.tight_layout()

    # Save plot
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)
    filename = f"nh3_flame_profile_phi_{str(phi).replace('.', 'p')}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_species_fluxes(flame, phi, transport):
    """
    Plot species molar fluxes if available
    Optional analysis for transport phenomena
    """
    try:
        grid = flame.grid
        # Try different methods to access fluxes
        try:
            H_flux = flame.fluxes("H")
            H2_flux = flame.fluxes("H2")
            flux_available = True
        except Exception:
            if hasattr(flame, "species_mole_fluxes"):
                gas = flame.gas
                flux_matrix = flame.species_mole_fluxes
                H_flux = flux_matrix[gas.species_index("H"), :]
                H2_flux = flux_matrix[gas.species_index("H2"), :]
                flux_available = True
            else:
                flux_available = False

        if flux_available:
            plt.figure()
            plt.plot(grid * 1000, H_flux, label="H flux")
            plt.plot(grid * 1000, H2_flux, label="H2 flux")
            plt.xlabel("Position [mm]")
            plt.ylabel(r'Molar flux [mol m$^{-2}$ s$^{-1}$]')
            plt.title(f"Species Fluxes: $\phi$={phi}, {transport}")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()

            output_dir = get_output_dir()
            output_dir.mkdir(exist_ok=True)
            filename = f"nh3_flux_phi_{str(phi).replace('.', 'p')}_{transport}.png"
            plt.savefig(output_dir / filename, dpi=150)
            plt.close()

    except Exception:
        pass  # Skip flux plotting if not available


# =============================================================================
# MAIN ANALYSIS
# =============================================================================
def analyze_nh3_flame_structure():
    """
    Perform comprehensive NH3 flame structure analysis
    """
    print("NH3 Flame Structure Analysis")
    print("=" * 50)
    print(f"Mechanism: {Path(MECHANISM).name}")
    print(f"Conditions: T_u={CONDITIONS['temperature_unburned']}K, p={CONDITIONS['pressure']}bar")
    print(f"Transport models: {CONDITIONS['transport_models']}")
    print()

    # Storage for results
    results = {transport: [] for transport in CONDITIONS['transport_models']}
    thickness_results = {transport: [] for transport in CONDITIONS['transport_models']}
    flame_pairs = {}  # Store flame objects for profile comparison

    # Calculate for each equivalence ratio
    for phi in CONDITIONS['phi_range']:
        print(f"Analyzing phi = {phi:.2f}")

        mix_success = multi_success = False

        # Mixture-averaged transport
        flame_speed_mix, flame_mix = calculate_flame_speed(
            MECHANISM, FUEL, phi, CONDITIONS['pressure'],
            CONDITIONS['temperature_unburned'], "mixture-averaged",
            CONDITIONS['domain_width']
        )

        if flame_speed_mix is not None:
            results["mixture-averaged"].append((phi, flame_speed_mix))
            thickness_mix = calculate_thermal_thickness(flame_mix)
            thickness_results["mixture-averaged"].append((phi, thickness_mix))
            plot_species_fluxes(flame_mix, phi, "mixture-averaged")
            mix_success = True
            print(f"  Mix-avg: S_L = {flame_speed_mix:.4f} m/s, delta_T = {thickness_mix * 1000:.2f} mm")
        else:
            results["mixture-averaged"].append((phi, np.nan))
            thickness_results["mixture-averaged"].append((phi, np.nan))
            print(f"  Mix-avg: Failed")

        # Multicomponent transport
        flame_speed_multi, flame_multi = calculate_flame_speed(
            MECHANISM, FUEL, phi, CONDITIONS['pressure'],
            CONDITIONS['temperature_unburned'], "multicomponent",
            CONDITIONS['domain_width']
        )

        if flame_speed_multi is not None:
            results["multicomponent"].append((phi, flame_speed_multi))
            thickness_multi = calculate_thermal_thickness(flame_multi)
            thickness_results["multicomponent"].append((phi, thickness_multi))
            plot_species_fluxes(flame_multi, phi, "multicomponent")
            multi_success = True
            print(f"  Multi:   S_L = {flame_speed_multi:.4f} m/s, delta_T = {thickness_multi * 1000:.2f} mm")
        else:
            results["multicomponent"].append((phi, np.nan))
            thickness_results["multicomponent"].append((phi, np.nan))
            print(f"  Multi:   Failed")

        # Store flame pairs for profile comparison
        if mix_success and multi_success:
            flame_pairs[phi] = (flame_mix, flame_multi)
            plot_flame_profiles(flame_mix, flame_multi, phi)

        print()

    return results, thickness_results, flame_pairs


def plot_summary_results(results, thickness_results):
    """
    Create summary plots for flame speed and thickness
    """
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)

    # Flame speed plot
    plt.figure()
    for transport, data in results.items():
        phi_values, flame_speeds = zip(*data)
        plt.plot(phi_values, flame_speeds, marker='o', linewidth=2,
                 label=transport, markersize=8)

    plt.xlabel("Equivalence Ratio $\phi$")
    plt.ylabel("Laminar Flame Speed $S_L$ [m/s]")
    soret_note = ", Soret effects included" if CONDITIONS['enable_soret'] else ""
    plt.title(f"NH3–Air Laminar Flame Speed\n"
              f"$T_u$ = {CONDITIONS['temperature_unburned']}K, p = {CONDITIONS['pressure']}bar{soret_note}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "nh3_flame_speed.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Thermal thickness plot
    plt.figure()
    for transport, data in thickness_results.items():
        phi_values, thicknesses = zip(*data)
        thickness_mm = np.array(thicknesses) * 1000  # Convert to mm
        plt.plot(phi_values, thickness_mm, marker='s', linewidth=2,
                 label=transport, markersize=8)

    plt.xlabel("Equivalence Ratio $\phi$")
    plt.ylabel("Thermal Flame Thickness $\delta_T$ [mm]")
    plt.title(f"NH3–Air Flame Thickness\n"
              f"$T_u$ = {CONDITIONS['temperature_unburned']}K, p = {CONDITIONS['pressure']}bar")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "nh3_flame_thickness.png", dpi=150, bbox_inches='tight')
    plt.show()


# =============================================================================
# EXECUTION
# =============================================================================
def main():
    """Main analysis routine for NH3 flame structure characterization"""
    print("Starting NH3 Flame Structure Analysis...")
    print("Focus: Laminar flame speed and transport model comparison")
    print()

    # Core analysis
    results, thickness_results, flame_pairs = analyze_nh3_flame_structure()

    if any(results.values()):
        # Create summary plots
        plot_summary_results(results, thickness_results)

        print("Key Findings:")
        # Find maximum flame speeds
        for transport, data in results.items():
            valid_data = [(phi, SL) for phi, SL in data if not np.isnan(SL)]
            if valid_data:
                max_result = max(valid_data, key=lambda x: x[1])
                print(f"  {transport}: Max S_L = {max_result[1]:.4f} m/s at phi = {max_result[0]}")

        print(f"\nAnalysis complete. Results saved in 'results/' directory.")
        print(f"Profile comparisons available for phi = {list(flame_pairs.keys())}")

    else:
        print("Analysis failed. Check mechanism file path and Cantera installation.")


if __name__ == "__main__":
    main()