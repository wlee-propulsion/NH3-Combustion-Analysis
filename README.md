# NH3 Combustion Analysis

Comprehensive ammonia combustion modeling suite using Cantera for industrial decarbonization applications.

## Overview

Four analysis modules covering the complete spectrum of NH3 combustion characterization:

- **01_equilibrium**: Thermodynamic equilibrium analysis with CH4 comparison
- **02_kinetics**: Ignition delay time characterization  
- **03_flame_structure**: Laminar flame speed and transport model comparison
- **04_pathways**: Nitrogen reaction pathways and NOx formation analysis

## Requirements

### Python Dependencies
pip install cantera numpy matplotlib

### External Dependencies
- **Graphviz** (for reaction pathway diagrams in module 4)
  - macOS: `brew install graphviz`
  - Ubuntu/Debian: `sudo apt install graphviz`
  - Windows: Download from https://graphviz.org/download/

### Mechanism Files
- MEI_2019.yaml for NH3 chemistry (place in project root)
- GRI-Mech 3.0 (included with Cantera)

## Usage

Run each analysis module:

| Module | Command |
|--------|---------|
| **Equilibrium** | `cd 01_equilibrium && python nh3_adiabatic_flame_analysis.py` |
| **Kinetics** | `cd 02_kinetics && python nh3_ignition_delay_analysis.py` |
| **Flame Structure** | `cd 03_flame_structure && python nh3_flame_structure_analysis.py` |
| **Pathways** | `cd 04_pathways && python nh3_nitrogen_pathway_analysis.py` |

Results are saved in `results/` subdirectories within each module.

## Key Results

- NH3 adiabatic flame temperatures and equilibrium composition
- Ignition delay characteristics across temperature and equivalence ratio ranges
- Laminar flame speeds and thermal thickness measurements  
- Nitrogen chemistry pathways and NOx emission predictions

## Applications

Data essential for:
- Industrial furnace design and retrofitting
- Burner development for NH3 fuel systems
- CFD model validation
- NOx emission control strategy development

## Author

Wonhyeong Lee - Aerospace Engineering Graduate, TU Berlin