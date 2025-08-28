"""
NH3 Nitrogen Reaction Pathway Analysis
Chemical pathway visualization for NOx formation/reduction in ammonia combustion
Author: Wonhyeong Lee
"""

import cantera as ct
import numpy as np
import matplotlib.pyplot as plt
from subprocess import run
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
MECHANISM = (SCRIPT_DIR.parent / "MEI_2019.yaml").as_posix()  # NH3 mechanism
FUEL = "NH3"
CONDITIONS = {
    'phi': 1.0,                    # Stoichiometric for complete analysis
    'pressure': ct.one_atm,        # [Pa]
    'temperature': 1300.0,         # [K] Typical industrial furnace temperature
    'residence_time': 0.1          # [s] Typical industrial residence time
}

# Analysis focus
ELEMENTS_OF_INTEREST = ['N']  # Primary: Nitrogen pathways
NOX_SPECIES = ['NO', 'NO2', 'N2O', 'HNO', 'NH', 'NH2']  # Key nitrogen species

def get_output_dir():
    out = SCRIPT_DIR / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out

# =============================================================================
# REACTOR SETUP AND ANALYSIS
# =============================================================================
def setup_nh3_reactor():
    """
    Setup NH3 combustion reactor for pathway analysis
    Simulates industrial furnace conditions with controlled residence time
    """
    try:
        # Initialize gas mixture
        gas = ct.Solution(MECHANISM)
        gas.set_equivalence_ratio(CONDITIONS['phi'], f"{FUEL}:1", "O2:1, N2:3.76")
        gas.TP = CONDITIONS['temperature'], CONDITIONS['pressure']

        print("Initial NH3-Air Mixture Conditions:")
        print(f"  phi = {CONDITIONS['phi']}")
        print(f"  T = {CONDITIONS['temperature']} K")
        print(f"  p = {CONDITIONS['pressure']/ct.one_atm:.1f} atm")
        print(f"  Fuel composition: {FUEL}")
        print()

        # Create ideal gas reactor (constant pressure)
        reactor = ct.IdealGasReactor(gas)
        reactor_network = ct.ReactorNet([reactor])

        # Set tolerances for stiff chemistry
        reactor_network.rtol = 1e-9
        reactor_network.atol = 1e-15

        return gas, reactor, reactor_network

    except Exception as e:
        print(f"Reactor setup failed: {e}")
        return None, None, None

def analyze_nitrogen_evolution(reactor, network, max_time):
    """
    Track nitrogen species evolution during combustion
    """
    # Initialize storage
    times = [0.0]
    temperatures = [reactor.T]
    nitrogen_species_history = {species: [0.0] for species in NOX_SPECIES}

    # Get initial concentrations
    gas = reactor.thermo
    for species in NOX_SPECIES:
        try:
            idx = gas.species_index(species)
            nitrogen_species_history[species][0] = gas.X[idx]
        except ValueError:
            nitrogen_species_history[species][0] = 0.0

    print("Tracking nitrogen species evolution...")
    print("Time [ms]   T [K]    NO      NO2     N2O     NH2")
    print("-" * 50)

    # Time integration
    dt = max_time / 1000  # 1000 steps
    while network.time < max_time:
        network.advance(network.time + dt)

        times.append(network.time)
        temperatures.append(reactor.T)

        # Record nitrogen species
        for species in NOX_SPECIES:
            try:
                idx = gas.species_index(species)
                nitrogen_species_history[species].append(gas.X[idx])
            except ValueError:
                nitrogen_species_history[species].append(0.0)

        # Print periodic updates
        if len(times) % 200 == 0:  # Every 200 steps
            t_ms = network.time * 1000
            NO_ppm = nitrogen_species_history['NO'][-1] * 1e6
            NO2_ppm = nitrogen_species_history['NO2'][-1] * 1e6
            N2O_ppm = nitrogen_species_history['N2O'][-1] * 1e6
            NH2_ppm = nitrogen_species_history['NH2'][-1] * 1e6
            print(f"{t_ms:6.1f}   {reactor.T:6.0f}   {NO_ppm:6.1f}   {NO2_ppm:6.1f}   {N2O_ppm:6.1f}   {NH2_ppm:6.1f}")

    return times, temperatures, nitrogen_species_history

def check_graphviz_installation():
    """
    Check if graphviz is installed and provide installation instructions
    """
    try:
        result = run(['dot', '-V'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Graphviz found: {result.stderr.strip()}")
            return True
    except FileNotFoundError:
        pass

    print("Graphviz not found. Installation required for pathway diagrams:")

    return False

def create_reaction_pathway_diagram(gas, element='N'):
    """
    Generate reaction pathway diagram focusing on nitrogen chemistry
    """
    print(f"\nGenerating reaction pathway diagram for element: {element}")

    # Check graphviz availability
    if not check_graphviz_installation():
        print("Skipping pathway diagram generation")
        return None, None

    try:
        # Create pathway diagram
        diagram = ct.ReactionPathDiagram(gas, element)
        diagram.title = f'NH3 Combustion: {element} Reaction Pathways'
        diagram.label_threshold = 0.01  # Show pathways with >1% flux

        # Create output directory
        output_dir = get_output_dir()
        output_dir.mkdir(exist_ok=True)

        # Output files
        dot_file = output_dir / f'nh3_{element.lower()}_pathways.dot'
        img_file = output_dir / f'nh3_{element.lower()}_pathways.png'

        # Write DOT file
        diagram.write_dot(str(dot_file))
        print(f"DOT file written: results/{Path(dot_file).name}")

        # Display pathway data summary
        print(f"Pathway diagram settings:")
        print(f"  Element tracked: {element}")
        print(f"  Flux threshold: {diagram.label_threshold*100:.1f}%")
        print(f"  Title: {diagram.title}")

        # Convert DOT to PNG using graphviz
        try:
            cmd = ['dot', str(dot_file), '-Tpng', f'-o{img_file}', '-Gdpi=300']
            result = run(cmd, capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                print(f"Pathway diagram generated: results/{Path(img_file).name}")

                # Verify file was created and has reasonable size
                if img_file.exists() and img_file.stat().st_size > 1000:  # >1KB
                    print("Pathway diagram generation successful")
                    return str(dot_file), str(img_file)
                else:
                    print("Warning: Generated image file seems too small")

            else:
                print(f"Graphviz error: {result.stderr}")

        except Exception as e:
            print(f"Graphviz execution failed: {e}")

    except Exception as e:
        print(f"Pathway diagram creation failed: {e}")

    return str(dot_file) if 'dot_file' in locals() else None, None

def extract_major_pathways(gas, element='N'):
    """
    Extract and analyze major reaction pathways numerically
    Alternative analysis when graphviz visualization unavailable
    """
    print(f"\nAnalyzing major {element} reaction pathways numerically...")

    try:
        diagram = ct.ReactionPathDiagram(gas, element)

        # Get net production rates for nitrogen species
        net_rates = gas.net_production_rates  # [mol/m³/s]

        # Focus on nitrogen species
        nitrogen_rates = {}
        for i, species_name in enumerate(gas.species_names):
            if element.upper() in species_name and abs(net_rates[i]) > 1e-10:
                nitrogen_rates[species_name] = net_rates[i]

        # Sort by absolute production rate
        sorted_rates = sorted(nitrogen_rates.items(), key=lambda x: abs(x[1]), reverse=True)

        print("Major nitrogen species production/consumption rates:")
        print("Species      Rate [mol/m^3/s]    Type")
        print("-" * 40)

        for species, rate in sorted_rates[:10]:  # Top 10
            rate_type = "Production" if rate > 0 else "Consumption"
            print(f"{species:>8s}   {rate:12.2e}   {rate_type}")

        return sorted_rates

    except Exception as e:
        print(f"Numerical pathway analysis failed: {e}")
        return []

def plot_nitrogen_species_evolution(times, temperatures, species_history):
    """
    Plot time evolution of key nitrogen species
    Focus on NOx formation kinetics and intermediate radicals
    """
    output_dir = get_output_dir()
    output_dir.mkdir(exist_ok=True)

    # Convert times to milliseconds for better readability
    times_ms = np.array(times) * 1000

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))

    # Temperature evolution (top subplot)
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(times_ms, temperatures, 'k-', linewidth=2)
    ax1.set_xlabel('Time [ms]')
    ax1.set_ylabel('Temperature [K]')
    ax1.set_title('Temperature Evolution')
    ax1.grid(True, alpha=0.3)

    # NOx species evolution (log scale)
    ax2 = plt.subplot(3, 2, 2)
    nox_main = ['NO', 'NO2', 'N2O']
    colors = ['red', 'blue', 'green']

    for species, color in zip(nox_main, colors):
        if species in species_history:
            concentrations = np.array(species_history[species])
            ppm_values = np.maximum(concentrations * 1e6, 1e-3)  # Convert to ppm
            ax2.semilogy(times_ms, ppm_values, color=color, linewidth=2, label=species)

    ax2.set_xlabel('Time [ms]')
    ax2.set_ylabel('Concentration [ppm]')
    ax2.set_title('NOx Species Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # NH radical species
    ax3 = plt.subplot(3, 2, 3)
    nh_species = ['NH', 'NH2']
    colors_nh = ['orange', 'purple']

    for species, color in zip(nh_species, colors_nh):
        if species in species_history:
            concentrations = np.array(species_history[species])
            ppm_values = np.maximum(concentrations * 1e6, 1e-3)
            ax3.semilogy(times_ms, ppm_values, color=color, linewidth=2, label=species)

    ax3.set_xlabel('Time [ms]')
    ax3.set_ylabel('Concentration [ppm]')
    ax3.set_title('NH Radical Evolution')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # NOx formation rates (derivative)
    ax4 = plt.subplot(3, 2, 4)
    if 'NO' in species_history and len(times_ms) > 1:
        NO_ppm = np.array(species_history['NO']) * 1e6
        dNO_dt = np.gradient(NO_ppm, times_ms)  # ppm/ms
        ax4.plot(times_ms[1:], dNO_dt[1:], 'red', linewidth=2, label='NO formation rate')
        ax4.set_xlabel('Time [ms]')
        ax4.set_ylabel('dC/dt [ppm/ms]')
        ax4.set_title('NO Formation Rate')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)

    # Final NOx levels comparison
    ax5 = plt.subplot(3, 2, 5)
    final_concentrations = {}
    for species in nox_main:
        if species in species_history:
            final_concentrations[species] = species_history[species][-1] * 1e6  # ppm

    if final_concentrations:
        species_names = list(final_concentrations.keys())
        concentrations = list(final_concentrations.values())

        bars = ax5.bar(species_names, concentrations, color=['red', 'blue', 'green'][:len(species_names)])
        ax5.set_ylabel('Final Concentration [ppm]')
        ax5.set_title(f'Final NOx Levels (t = {times_ms[-1]:.1f} ms)')
        ax5.grid(True, alpha=0.3)

        # Add value labels
        for bar, conc in zip(bars, concentrations):
            if conc > 0.01:
                ax5.text(bar.get_x() + bar.get_width()/2, conc,
                        f'{conc:.2f}', ha='center', va='bottom')

    # Nitrogen balance check
    ax6 = plt.subplot(3, 2, 6)
    # Calculate total nitrogen in major species over time
    N_species = ['N2', 'NO', 'NO2', 'N2O', 'NH3', 'NH2', 'NH']
    total_N = np.zeros(len(times))

    for species in N_species:
        if species in species_history:
            # Account for number of N atoms per molecule
            n_atoms = 2 if species == 'N2' or species == 'N2O' else 1
            if species == 'NH3':
                n_atoms = 1
            contribution = np.array(species_history.get(species, np.zeros(len(times)))) * n_atoms
            total_N += contribution
        elif species == 'N2':  # N2 should be major product
            # Estimate N2 from nitrogen balance if not tracked
            initial_N = 2 * 0.79  # Approximate N2 in air
            total_N += initial_N

    if np.any(total_N > 0):
        ax6.plot(times_ms, total_N, 'k-', linewidth=2, label='Total N balance')
        ax6.set_xlabel('Time [ms]')
        ax6.set_ylabel('Total N mole fraction')
        ax6.set_title('Nitrogen Balance Check')
        ax6.grid(True, alpha=0.3)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, 'N2 data not available\nfor balance calculation',
                ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Nitrogen Balance Check')

    plt.tight_layout()
    plot_file = output_dir / "nh3_nitrogen_evolution.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    return str(plot_file)

def analyze_final_composition(gas):
    """
    Comprehensive analysis of final equilibrium composition
    Focus on nitrogen products and NOx emission levels
    """
    print("\nFinal Composition Analysis:")
    print("=" * 50)

    # All nitrogen-containing species above detection threshold
    nitrogen_species_final = []
    total_N_atoms = 0

    for i, name in enumerate(gas.species_names):
        if 'N' in name and gas.X[i] > 1e-12:  # Detection threshold
            mole_frac = gas.X[i]
            ppm = mole_frac * 1e6
            mass_frac = gas.Y[i]
            nitrogen_species_final.append((name, mole_frac, ppm, mass_frac))

    # Sort by mole fraction
    nitrogen_species_final.sort(key=lambda x: x[1], reverse=True)

    print("Nitrogen-containing species (>0.001 ppm):")
    print("Species    Mole Frac    Conc [ppm]    Mass Frac")
    print("-" * 50)

    for name, X, ppm, Y in nitrogen_species_final:
        if ppm > 0.001:
            print(f"{name:>7s}   {X:10.3e}   {ppm:9.3f}   {Y:10.3e}")

    # Calculate total NOx emissions
    nox_species = ['NO', 'NO2', 'N2O']
    nox_total = 0
    nox_breakdown = {}

    for name, X, ppm, Y in nitrogen_species_final:
        if name in nox_species:
            nox_total += ppm
            nox_breakdown[name] = ppm

    print(f"\nNOx Emission Assessment:")
    print(f"  Total NOx: {nox_total:.2f} ppm")

    if nox_breakdown:
        dominant_nox = max(nox_breakdown.items(), key=lambda x: x[1])
        print(f"  Dominant NOx: {dominant_nox[0]} ({dominant_nox[1]:.2f} ppm)")

        for species, conc in nox_breakdown.items():
            percentage = (conc / nox_total * 100) if nox_total > 0 else 0
            print(f"    {species}: {conc:.3f} ppm ({percentage:.1f}%)")

    # Industrial relevance assessment
    print(f"\nIndustrial Relevance:")
    if nox_total > 100:
        print(f"  HIGH NOx emissions - requires control measures")
    elif nox_total > 10:
        print(f"  MODERATE NOx emissions - monitoring recommended")
    else:
        print(f"  LOW NOx emissions - acceptable for most applications")

    return nitrogen_species_final, nox_breakdown

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    """
    Complete NH3 nitrogen pathway analysis workflow
    Combines kinetic evolution, pathway visualization, and emission assessment
    """
    print("NH3 Nitrogen Reaction Pathway Analysis")
    print("=" * 60)
    print("Objective: Understand NOx formation in NH3 industrial combustion")
    print(f"Mechanism: {Path(MECHANISM).name}")
    print(f"Analysis conditions: phi={CONDITIONS['phi']}, T={CONDITIONS['temperature']}K")
    print()

    # Setup reactor system
    gas, reactor, network = setup_nh3_reactor()
    if not reactor:
        print("Failed to setup reactor. Check mechanism file path:")
        print(f"  Current path: {MECHANISM}")
        print("  Make sure MEI_2019.yaml is available")
        return

    # Track nitrogen species evolution over time
    print("Step 1: Nitrogen species evolution analysis")
    times, temperatures, species_history = analyze_nitrogen_evolution(
        reactor, network, CONDITIONS['residence_time']
    )

    # Generate comprehensive plots
    print("\nStep 2: Creating visualization plots")
    evolution_plot_file = plot_nitrogen_species_evolution(times, temperatures, species_history)

    # Reaction pathway diagram generation
    print("\nStep 3: Reaction pathway analysis")
    final_gas = reactor.thermo
    dot_file, img_file = create_reaction_pathway_diagram(final_gas, 'N')

    # If pathway diagram failed, do numerical analysis
    if not img_file:
        print("\nFallback: Numerical pathway analysis")
        major_pathways = extract_major_pathways(final_gas, 'N')

    # Final composition and emission analysis
    print("\nStep 4: Final composition analysis")
    nitrogen_species, nox_emissions = analyze_final_composition(final_gas)

    # Summary report
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Simulation time: {CONDITIONS['residence_time']*1000:.1f} ms")
    print(f"Final temperature: {reactor.T:.0f} K")
    print(f"Initial temperature: {CONDITIONS['temperature']} K")
    print(f"Temperature rise: {reactor.T - CONDITIONS['temperature']:.0f} K")

    if nox_emissions:
        total_nox = sum(nox_emissions.values())
        print(f"Total NOx emissions: {total_nox:.2f} ppm")
        print(f"Primary NOx species: {max(nox_emissions.items(), key=lambda x: x[1])[0]}")

    print(f"\nOutput files generated:")
    print(f"  Nitrogen evolution plots: results/{Path(evolution_plot_file).name}")
    if img_file:
        print(f"  Reaction pathway diagram: results/{Path(img_file).name}")
    if dot_file:
        print(f"  DOT source file: results/{Path(dot_file).name}")


    return {
        'times': times,
        'temperatures': temperatures,
        'species_history': species_history,
        'final_composition': nitrogen_species,
        'nox_emissions': nox_emissions
    }

if __name__ == "__main__":
    results = main()