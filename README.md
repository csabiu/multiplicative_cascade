# Multiplicative Cascade

A Python package for generating and analyzing multifractal fields using multiplicative cascade processes.

## Overview

This package implements multifractal analysis tools based on multiplicative cascades, as described in [Wikipedia: Multiplicative cascade](https://en.wikipedia.org/wiki/Multiplicative_cascade). It provides functions for:

- Generating multifractal density fields
- Measuring fractal dimensions using box-counting methods
- Creating mock particle distributions from density fields

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/csabiu/multiplicative_cascade.git
cd multiplicative_cascade
pip install numpy scipy
```

## Main Functions

### `multifrac(probabilities, dim=2, size=2, levels=4, add_power=False)`

Generate a multiplicative cascade fractal field.

**Parameters:**
- `probabilities`: Probability weights for the cascade (should sum to ~1.0)
- `dim`: Dimensionality (1, 2, or 3)
- `size`: Subdivision factor at each level (default: 2)
- `levels`: Number of cascade levels (default: 4)
- `add_power`: Apply increasing power weights at each level (default: False)

**Returns:**
- `cascade_field`: The generated multifractal field
- `theoretical_slope`: Theoretical fractal dimension

**Example:**
```python
import numpy as np
from multifrac import multifrac

# Create a 2D multifractal field
weights = np.array([0.1, 0.2, 0.3, 0.4])
field, theory_dim = multifrac(weights, dim=2, size=2, levels=5)
print(f"Field shape: {field.shape}")
print(f"Theoretical dimension: {theory_dim:.4f}")
```

### `fractal_dimension(field, threshold=None, method='binary', q=2, ...)`

Measure the fractal dimension of a 2D or 3D field using box-counting methods.

**Parameters:**
- `field`: Input field (2D or 3D array)
- `threshold`: Threshold for binary method (default: median of non-zero values)
- `method`: 'binary' (threshold-based) or 'mass' (intensity-based)
- `q`: Order of generalized dimension for mass method (default: 2)
- `weights`: Optional spatial weight array
- `mask`: Optional boolean mask for observed regions
- `min_observed_fraction`: Minimum observed fraction for a box (default: 0.5)
- `min_box_size`: Minimum box size in pixels (default: 2)
- `max_box_size`: Maximum box size (default: half of smallest dimension)
- `num_scales`: Number of box sizes to test (default: 10)
- `return_diagnostics`: Return detailed fit information (default: False)

**Returns:**
- `dimension`: Estimated fractal dimension
- `diagnostics`: (optional) Dictionary with fit details

**Methods:**
1. **Binary box-counting** (`method='binary'`): Counts boxes where mean > threshold
2. **Mass-based** (`method='mass'`): Accounts for field intensity using partition functions

**Example:**
```python
from multifrac import fractal_dimension

# Binary box-counting (default)
dim_binary = fractal_dimension(field)

# Mass-based method
dim_mass = fractal_dimension(field, method='mass', q=2)

# With spatial weighting
weights = np.exp(-field)
dim_weighted = fractal_dimension(field, weights=weights)

# With mask for incomplete observations
mask = (observations > 0)  # True where data exists
dim_masked = fractal_dimension(field, mask=mask, min_observed_fraction=0.3)
```

### `mock(density, boxsize=100, Npart=100)`

Generate mock particle distributions from a density field using Poisson sampling.

**Parameters:**
- `density`: Input density field (2D or 3D)
- `boxsize`: Physical size of the simulation box (default: 100)
- `Npart`: Target number of particles (default: 100)

**Returns:**
- `points`: Particle positions sampled from density field
- `randoms`: Uniform random positions (10× Npart)

**Example:**
```python
from multifrac import mock

# Generate particles from density field
particles, randoms = mock(field, boxsize=100, Npart=1000)
print(f"Generated {particles.shape[0]} particles")
```

## Theory

### Multiplicative Cascade

A multiplicative cascade creates a multifractal by recursively subdividing space and multiplying by random probability weights. The theoretical correlation dimension D₂ is given by:

```
D₂ = -log₂(Σᵢ pᵢ²)
```

where pᵢ are the normalized probability weights.

### Box-Counting Methods

1. **Binary method**: Counts occupied boxes at different scales
   - N(ε) ~ ε^(-D)

2. **Mass method**: Uses partition functions
   - Σᵢ mᵢ^q ~ ε^((q-1)D_q)
   - D_q is the generalized dimension of order q

Common q values:
- q=0: Capacity dimension
- q=1: Information dimension
- q=2: Correlation dimension (recommended)

## Examples

See the test scripts for detailed examples:
- `test_fractal_dimension.py`: Basic 2D/3D fractal dimension measurement
- `test_mass_dimension.py`: Comparison of binary vs mass-based methods
- `test_unified_dimension.py`: Advanced features (weights, masking)

Run tests:
```bash
python test_fractal_dimension.py
python test_mass_dimension.py
python test_unified_dimension.py
```

## Features

- **Multi-dimensional**: Supports 1D, 2D, and 3D fields
- **Flexible analysis**: Binary and mass-based box-counting methods
- **Spatial weighting**: Emphasize specific regions in the analysis
- **Masking support**: Handle incomplete observations with automatic corrections
- **Generalized dimensions**: Compute D_q for different orders q
- **Diagnostic output**: Detailed fit quality metrics and parameters

## References

- [Multiplicative cascade - Wikipedia](https://en.wikipedia.org/wiki/Multiplicative_cascade)
- Box-counting dimension for multifractal analysis
- Generalized dimensions and partition functions

## License

See LICENSE file for details.
