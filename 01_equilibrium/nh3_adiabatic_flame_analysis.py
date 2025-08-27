"""
NH3 Adiabatic Flame Temperature and Equilibrium Analysis
Thermochemical equilibrium characterization for ammonia combustion systems
Comparative analysis with methane CH4 for fuel transition assessment
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
    'pressure': ct.one_atm,  # [Pa]
    'temperature': 300.0,  # Initial temperature [K]
    'phi_range': [0.6, 0.8, 1.0, 1.2, 1.4]  # Equivalence ratio range
}

def get_output_dir():
    out = SCRIPT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def calculate_adiabatic_conditions(mechanism, fuel, phi, T_initial, p_initial):
    """
    Calculate adiabatic flame temperature and equilibrium composition

    Parameters:
        mechanism: Cantera mechanism file path
        fuel: Fuel species name
        phi: Equivalence ratio
        T_initial: Initial temperature [K]
        p_initial: Initial pressure [Pa]

    Returns:
        dict: Contains T_ad, major species, and thermodynamic properties
    """
    try:
        # Initial mixture
        gas = ct.Solution(mechanism)
        gas.set_equivalence_ratio(phi, f"{fuel}:1", "O2:1, N2:3.76")
        gas.TP = T_initial, p_initial

        # Store unburned properties
        h_reactants = gas.enthalpy_mass  # [J/kg]
        T_unburned = gas.T

        # Adiabatic equilibrium (constant H, P)
        gas.equilibrate('HP')

        T_adiabatic = gas.T
        h_products = gas.enthalpy_mass

        # Extract major species (>1% mole fraction)
        major_species = []
        for i, name in enumerate(gas.species_names):
            if gas.X[i] > 0.001: # 0.1% threshold
                major_species.append((name, gas.X[i]))

        # Sort by mole fraction
        major_species.sort(key=lambda x: x[1], reverse=True)

        return {
            'phi': phi,
            'T_unburned': T_unburned,
            'T_adiabatic': T_adiabatic,
            'delta_T': T_adiabatic - T_unburned,
            'h_reactants': h_reactants,
            'h_products': h_products,
            'major_species': major_species[:8],  # Top 8 species
            'pressure': gas.P
        }

    except Exception as e:
        print(f"Error for φ={phi}: {e}")
        return None


def analyze_nh3_equilibrium():
    """
    Perform NH3 adiabatic flame analysis
    Focuses on temperature rise and nitrogen chemistry products
    """
    results = []

    print("NH3-Air Adiabatic Flame Analysis")
    print("=" * 50)
    print(f"Mechanism: {Path(MECHANISM).name}")
    print(f"Initial conditions: T={CONDITIONS['temperature']} K, p={CONDITIONS['pressure'] / ct.one_atm:.1f} atm")

    for phi in CONDITIONS['phi_range']:
        result = calculate_adiabatic_conditions(
            MECHANISM, FUEL, phi,
            CONDITIONS['temperature'], CONDITIONS['pressure']
        )

        if result:
            results.append(result)

            # Print summary
            print(f"φ = {phi:.2f}")
            print(f"  T_ad = {result['T_adiabatic']:.1f} K  (ΔT = +{result['delta_T']:.1f} K)")
            print(f"  Major species:")
            for name, X in result['major_species'][:5]:  # Top 5
                print(f"    {name:>6s}: X = {X:.4f}")
            print()

    return results


def plot_adiabatic_results(results):
    """
    Create plots for NH3 adiabatic flame analysis
    """
    if not results:
        print("No results to plot")
        return

    # Extract data
    phi_vals = [r['phi'] for r in results]
    T_ad_vals = [r['T_adiabatic'] for r in results]
    delta_T_vals = [r['delta_T'] for r in results]

    # Create output directory
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Adiabatic flame temperature vs equivalence ratio
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Temperature plot
    ax1.plot(phi_vals, T_ad_vals, 'ro-', linewidth=2, markersize=8)
    ax1.set_xlabel('Equivalence Ratio φ')
    ax1.set_ylabel('Adiabatic Flame Temperature [K]')
    ax1.set_title(f'{FUEL}–Air Adiabatic Flame Temperature')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(min(T_ad_vals) - 50, max(T_ad_vals) + 50)

    # Add temperature rise annotations
    for phi, T_ad, dT in zip(phi_vals, T_ad_vals, delta_T_vals):
        ax1.annotate(f'{T_ad:.0f}K\n(+{dT:.0f})',
                     (phi, T_ad), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    # Temperature rise plot  
    ax2.plot(phi_vals, delta_T_vals, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Equivalence Ratio φ')
    ax2.set_ylabel('Temperature Rise ΔT [K]')
    ax2.set_title('Temperature Rise from Combustion')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "nh3_adiabatic_temperature.png", dpi=150, bbox_inches='tight')
    plt.show()

    # Plot 2: Major species composition at stoichiometric conditions
    # Find stoichiometric result (φ=1.0)
    stoich_result = next((r for r in results if abs(r['phi'] - 1.0) < 0.01), None)

    if stoich_result:
        species_names, mole_fractions = zip(*stoich_result['major_species'])

        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(species_names)))
        bars = plt.bar(range(len(species_names)), mole_fractions, color=colors)

        plt.xlabel('Species')
        plt.ylabel('Mole Fraction')
        plt.title(f'{FUEL}–Air Equilibrium Products (φ=1.0, T_ad={stoich_result["T_adiabatic"]:.0f}K)')
        plt.xticks(range(len(species_names)), species_names, rotation=45)
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for i, (bar, X) in enumerate(zip(bars, mole_fractions)):
            if X > 0.001:  # Only label significant species
                plt.text(bar.get_x() + bar.get_width() / 2, X, f'{X:.3f}',
                         ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / "nh3_equilibrium_species.png", dpi=150, bbox_inches='tight')
        plt.show()


def compare_with_methane():
    """
    Compare NH3 vs CH4 adiabatic flame temperatures
    """
    try:
        # CH4 calculation with GRI-Mech
        gri_mech = 'gri30.yaml'
        ch4_results = []

        for phi in CONDITIONS['phi_range']:
            result = calculate_adiabatic_conditions(
                gri_mech, "CH4", phi,
                CONDITIONS['temperature'], CONDITIONS['pressure']
            )
            if result:
                ch4_results.append(result)

        # Plot comparison
        if ch4_results:
            phi_vals = CONDITIONS['phi_range']
            nh3_temps = [r['T_adiabatic'] for r in analyze_nh3_equilibrium()]
            ch4_temps = [r['T_adiabatic'] for r in ch4_results]

            plt.figure(figsize=(8, 6))
            plt.plot(phi_vals, nh3_temps, 'ro-', label='NH3-Air', linewidth=2)
            plt.plot(phi_vals, ch4_temps, 'bo-', label='CH4-Air', linewidth=2)
            plt.xlabel('Equivalence Ratio φ')
            plt.ylabel('Adiabatic Flame Temperature [K]')
            plt.title('Fuel Comparison: Adiabatic Flame Temperature')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            output_dir = get_output_dir()
            output_dir.mkdir(exist_ok=True)
            plt.savefig(output_dir / "fuel_comparison.png", dpi=150)
            plt.show()

    except Exception as e:
        print(f"CH4 comparison failed: {e}")


def main():
    """Main analysis routine for NH3 adiabatic flame characterization"""
    print("Starting NH3 Adiabatic Flame Analysis...")
    print("Focus: Equilibrium temperature and nitrogen chemistry products")

    results = analyze_nh3_equilibrium()

    if results:
        # Create visualizations
        plot_adiabatic_results(results)

        # Fuel comparison with methane
        print("Performing fuel comparison with CH4...")
        compare_with_methane()

        print(f"\nAnalysis complete. Results saved in 'results/' directory.")
        print(f"Key finding: NH3 peak adiabatic temperature ≈ {max(r['T_adiabatic'] for r in results):.0f} K")

        # Extract nitrogen-containing products for stoichiometric case
        stoich = next((r for r in results if abs(r['phi'] - 1.0) < 0.01), None)
        if stoich:
            n_species = [s for s in stoich['major_species'] if 'N' in s[0] and s[0] not in ['N2']]
            print(f"Nitrogen products (φ=1.0): {[s[0] for s in n_species]}")

    else:
        print("Analysis failed. Check mechanism file path and Cantera installation.")


if __name__ == "__main__":
    main()