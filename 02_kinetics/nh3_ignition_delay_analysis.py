"""
NH3 Ignition Delay Time Analysis
Autoignition characteristics of ammonia-air mixtures under varying conditions
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
    'phi_range': [0.6, 0.8, 1.0, 1.2, 1.4],
    'temperature_range': [1200, 1300, 1400, 1500],
    'pressure': ct.one_atm,                           # Initial pressure [Pa]
    'max_time': 0.5                                   # Simulation time [s]
}

def get_output_dir():
    out = SCRIPT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out
# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================
def calculate_ignition_delay(mechanism, fuel, phi, T_initial, p_initial, max_time):
    """
    Calculate ignition delay time using constant pressure reactor
    Improved detection criteria for reliable ignition identification

    Parameters:
        mechanism: Cantera mechanism file path
        fuel: Fuel species name
        phi: Equivalence ratio
        T_initial: Initial temperature [K]
        p_initial: Initial pressure [Pa]
        max_time: Maximum simulation time [s]

    Returns:
        float: Ignition delay time [s] or None if ignition not detected
    """
    try:
        # Setup gas mixture
        gas = ct.Solution(mechanism)
        gas.set_equivalence_ratio(phi, f"{fuel}:1", "O2:1.0, N2:3.76")
        gas.TP = T_initial, p_initial

        # Create constant pressure reactor
        reactor = ct.IdealGasReactor(gas)
        reactor_network = ct.ReactorNet([reactor])

        # Set tight tolerances for accurate ignition detection
        reactor_network.rtol = 1e-9
        reactor_network.atol = 1e-15

        # Storage arrays
        times = []
        temperatures = []

        # Initial conditions for ignition detection
        T_initial_recorded = reactor.T
        ignition_threshold = T_initial_recorded + 400  # 400K temperature rise threshold

        # Time integration with adaptive stepping
        dt = 1e-6  # Start with smaller time step
        while reactor_network.time < max_time:
            reactor_network.advance(reactor_network.time + dt)
            times.append(reactor_network.time)
            temperatures.append(reactor.T)

            # Check for ignition (rapid temperature rise)
            if len(temperatures) > 10:
                recent_dT = temperatures[-1] - temperatures[-10]
                if recent_dT > 200:  # Rapid temperature rise detected
                    break

            # Adaptive time stepping
            if len(temperatures) > 1:
                dT_dt = (temperatures[-1] - temperatures[-2]) / dt
                if abs(dT_dt) > 1000:  # High temperature gradient
                    dt = min(dt, 1e-7)  # Use smaller steps
                else:
                    dt = min(dt * 1.1, 1e-5)  # Gradually increase step size

        # Convert to numpy arrays
        times = np.array(times)
        temperatures = np.array(temperatures)

        # Check if ignition occurred
        if np.max(temperatures) < ignition_threshold:
            print(f"  No ignition detected (max T: {np.max(temperatures):.0f}K)")
            return None, None, None, None

        # Calculate temperature time derivative
        dT_dt = np.gradient(temperatures, times)

        # Find ignition delay (maximum temperature rise rate)
        ignition_idx = np.argmax(dT_dt)
        ignition_delay = times[ignition_idx]

        # Validate ignition delay (should be reasonable)
        if ignition_delay > max_time * 0.9:
            print(f"  Late ignition detected at {ignition_delay*1000:.1f}ms - may be spurious")
            return None, None, None, None

        return ignition_delay, times, temperatures, dT_dt

    except Exception as e:
        print(f"Error for $\phi$={phi}, T={T_initial}K: {e}")
        return None, None, None, None

def analyze_nh3_ignition():
    """
    Perform NH3 ignition delay analysis
    Covers range of equivalence ratios and temperatures
    """
    results = []

    print("NH3-Air Ignition Delay Analysis")
    print("=" * 50)
    print(f"Mechanism: {Path(MECHANISM).name}")
    print(f"Fuel: {FUEL}")
    print(f"Pressure: {CONDITIONS['pressure']/ct.one_atm:.1f} atm")
    print(f"phi range: {CONDITIONS['phi_range']}")
    print(f"T range: {CONDITIONS['temperature_range']} K")
    print()

    total_cases = len(CONDITIONS['phi_range']) * len(CONDITIONS['temperature_range'])
    case_count = 0

    for phi in CONDITIONS['phi_range']:
        for T0 in CONDITIONS['temperature_range']:
            case_count += 1
            print(f"Case {case_count}/{total_cases}: phi={phi}, T0={T0}K", end=" → ")

            tau, times, temps, dT_dt = calculate_ignition_delay(
                MECHANISM, FUEL, phi, T0,
                CONDITIONS['pressure'], CONDITIONS['max_time']
            )

            if tau is not None:
                results.append({
                    'phi': phi,
                    'T_initial': T0,
                    'ignition_delay': tau,
                    'max_temperature': np.max(temps) if temps is not None else None
                })
                print(f"tau = {tau:.4f} s ({tau*1000:.1f} ms)")
            else:
                print("No ignition detected")

    print(f"\nAnalysis complete: {len(results)}/{total_cases} cases converged")
    return results

def plot_ignition_delay_results(results):
    """
    Create plots for NH3 ignition delay analysis
    """
    if not results:
        print("No results to plot")
        return

    # Create output directory
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)

    # Extract unique phi and temperature values
    phi_values = sorted(set(r['phi'] for r in results))
    temp_values = sorted(set(r['T_initial'] for r in results))

    # Create comprehensive figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Ignition delay vs temperature for different phi
    for phi in phi_values:
        phi_data = [r for r in results if r['phi'] == phi]
        if phi_data:
            T_vals = [r['T_initial'] for r in phi_data]
            tau_vals = [r['ignition_delay'] * 1000 for r in phi_data]  # Convert to ms

            ax1.semilogy(T_vals, tau_vals, 'o-', linewidth=2, markersize=6, label=f'$\phi$ = {phi}')

    ax1.set_xlabel('Initial Temperature [K]')
    ax1.set_ylabel('Ignition Delay Time [ms]')
    ax1.set_title(f'{FUEL}-Air Ignition Delay')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Arrhenius plot (ln(tau) vs 1000/T) for stoichiometric
    stoich_data = [r for r in results if abs(r['phi'] - 1.0) < 0.01]
    if stoich_data:
        inv_T = [1000/r['T_initial'] for r in stoich_data]
        ln_tau = [np.log(r['ignition_delay']) for r in stoich_data]

        ax2.plot(inv_T, ln_tau, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel(r'$1000/T$ [K$^{-1}$]')
        ax2.set_ylabel(r'$ln(\tau)$ [$ln(s)$]')
        ax2.set_title('Arrhenius Plot ($\phi$ = 1.0)')
        ax2.grid(True, alpha=0.3)

        # Add activation energy estimate
        if len(inv_T) > 1:
            coeffs = np.polyfit(inv_T, ln_tau, 1)
            E_a_estimate = -coeffs[0] * 8.314  # R = 8.314 J/mol/K
            ax2.text(0.05, 0.95, f'Estimated $E_{{a}}$ $\\approx$ {E_a_estimate:.0f} kJ/mol',
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))

    # Plot 3: Ignition delay vs equivalence ratio at different temperatures
    colors = ['blue', 'green', 'red']
    for i, T0 in enumerate([1200, 1300, 1400]):  # Selected temperatures
        temp_data = [r for r in results if r['T_initial'] == T0]
        if temp_data:
            phi_vals = sorted([r['phi'] for r in temp_data])
            tau_vals = [r['ignition_delay'] * 1000 for r in sorted(temp_data, key=lambda x: x['phi'])]

            ax3.plot(phi_vals, tau_vals, 'o-', linewidth=2, markersize=6,
                    color=colors[i], label=f'T = {T0}K')

            # Find minimum ignition delay for this temperature
            min_idx = np.argmin(tau_vals)
            min_phi = phi_vals[min_idx]
            min_tau = tau_vals[min_idx]

            # Annotate minimum point
            ax3.annotate(f'Min: phi={min_phi}\n tau={min_tau:.1f}ms',
                        xy=(min_phi, min_tau), xytext=(10, 10),
                        textcoords='offset points', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
                        arrowprops=dict(arrowstyle='->', color=colors[i]))

    ax3.set_xlabel('Equivalence Ratio $\phi$')
    ax3.set_ylabel('Ignition Delay Time [ms]')
    ax3.set_title('Effect of Equivalence Ratio on Ignition Delay')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Use log scale for better visibility

    # Plot 4: 3D surface plot showing ignition delay landscape
    # Create meshgrid for 3D plotting
    phi_unique = sorted(set(r['phi'] for r in results))
    temp_unique = sorted(set(r['T_initial'] for r in results))

    if len(phi_unique) >= 3 and len(temp_unique) >= 3:
        # Create 2D grid
        PHI, TEMP = np.meshgrid(phi_unique, temp_unique)
        TAU = np.zeros_like(PHI)

        # Fill the grid with ignition delay data
        for i, T in enumerate(temp_unique):
            for j, phi in enumerate(phi_unique):
                result = next((r for r in results if r['phi'] == phi and r['T_initial'] == T), None)
                if result:
                    TAU[i, j] = result['ignition_delay'] * 1000  # Convert to ms
                else:
                    TAU[i, j] = np.nan

        # Create contour plot with appropriate scale
        # Set reasonable contour levels based on actual data range
        tau_min, tau_max = np.nanmin(TAU), np.nanmax(TAU)
        levels = np.linspace(tau_min, tau_max, 15)  # 15 levels covering actual data range

        contour = ax4.contourf(PHI, TEMP, TAU, levels=levels, cmap='viridis_r')
        contour_lines = ax4.contour(PHI, TEMP, TAU, levels=8, colors='white', alpha=0.7, linewidths=1)
        ax4.clabel(contour_lines, inline=True, fontsize=8, fmt='%.0f ms')

        ax4.set_xlabel('Equivalence Ratio $\phi$')
        ax4.set_ylabel('Temperature [K]')
        ax4.set_title('Ignition Delay Landscape')

        # Add colorbar with proper range
        cbar = plt.colorbar(contour, ax=ax4, shrink=0.8)
        cbar.set_label('Ignition Delay [ms]')

        # Set colorbar limits to actual data range
        cbar.mappable.set_clim(tau_min, tau_max)

        # Find and mark global minimum
        min_idx = np.nanargmin(TAU)
        min_i, min_j = np.unravel_index(min_idx, TAU.shape)
        ax4.plot(PHI[min_i, min_j], TEMP[min_i, min_j], 'r*', markersize=15,
                label=f'Global min: {TAU[min_i, min_j]:.1f}ms')
        ax4.legend()

    else:
        # Fallback: Temperature sensitivity analysis
        sensitivity_data = []
        for phi in phi_values:
            phi_data = sorted([r for r in results if r['phi'] == phi], key=lambda x: x['T_initial'])
            if len(phi_data) > 2:
                T_vals = [r['T_initial'] for r in phi_data]
                tau_vals = [r['ignition_delay'] for r in phi_data]

                # Calculate sensitivity at middle temperature
                mid_idx = len(T_vals) // 2
                if mid_idx > 0 and mid_idx < len(T_vals) - 1:
                    T_mid = T_vals[mid_idx]
                    dln_tau_dT_inv = np.gradient(np.log(tau_vals), [1/T for T in T_vals])[mid_idx]
                    sensitivity_data.append((phi, T_mid, abs(dln_tau_dT_inv)))

        if sensitivity_data:
            phi_sens, T_sens, S_sens = zip(*sensitivity_data)
            bars = ax4.bar(phi_sens, S_sens, color='lightblue', edgecolor='navy', alpha=0.7)
            ax4.set_xlabel('Equivalence Ratio $\phi$')
            ax4.set_ylabel('|Temperature Sensitivity|')
            ax4.set_title('Ignition Temperature Sensitivity')
            ax4.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, sens in zip(bars, S_sens):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(S_sens)*0.01,
                        f'{sens:.0f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "nh3_ignition_delay_overview.png", dpi=150, bbox_inches='tight')
    plt.show()

def create_summary_table(results):
    """
    Create summary table of ignition delay results
    """
    print("\nIgnition Delay Summary Table")
    print("=" * 60)
    print("phi   |  1100K  |  1200K  |  1300K  |  1400K  |  1500K")
    print("------|---------|---------|---------|---------|--------")

    phi_values = sorted(set(r['phi'] for r in results))
    temp_values = sorted(set(r['T_initial'] for r in results))

    for phi in phi_values:
        row = f"{phi:4.2f}  |"
        for T in temp_values:
            # Find result for this phi and T
            result = next((r for r in results if r['phi'] == phi and r['T_initial'] == T), None)
            if result:
                tau_ms = result['ignition_delay'] * 1000
                row += f"  {tau_ms:5.1f}  |"
            else:
                row += "   ---   |"
        print(row)

    print("\nNote: Times in milliseconds [ms]")

# =============================================================================
# EXECUTION
# =============================================================================
def main():
    """Main analysis routine for NH3 ignition delay characterization"""
    print("Starting NH3 Ignition Delay Analysis...")
    print("Focus: Autoignition characteristics for industrial safety assessment")
    print()

    # Core analysis
    results = analyze_nh3_ignition()

    if results:
        # Create visualizations
        plot_ignition_delay_results(results)

        # Summary table
        create_summary_table(results)

        # Key findings
        print(f"\nKey Findings:")
        min_delay = min(r['ignition_delay'] for r in results) * 1000
        max_delay = max(r['ignition_delay'] for r in results) * 1000
        print(f"  Ignition delay range: {min_delay:.1f} - {max_delay:.1f} ms")

        # Find conditions for minimum delay
        min_result = min(results, key=lambda x: x['ignition_delay'])
        print(f"  Fastest ignition: phi={min_result['phi']}, T={min_result['T_initial']}K")
        print(f"  → tau = {min_result['ignition_delay']*1000:.1f} ms")

        print(f"\nAnalysis complete. Results saved in 'results/' directory.")

    else:
        print("Analysis failed. Check mechanism file path and conditions.")

if __name__ == "__main__":
    main()