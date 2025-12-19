#!/usr/bin/env python3
"""Test script for unified fractal_dimension function with weights and masking."""

import numpy as np
from multifrac import multifrac, fractal_dimension


def test_method_parameter():
    """Test that method parameter works correctly."""
    print("=" * 60)
    print("Testing method parameter (binary vs mass)")
    print("=" * 60)

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights, dim=2, size=2, levels=5)

    print("\nBinary method (default):")
    dim_binary = fractal_dimension(field, method='binary')

    print("\nMass method (q=2):")
    dim_mass = fractal_dimension(field, method='mass', q=2)

    print(f"\nBinary dimension: {dim_binary:.4f}")
    print(f"Mass dimension (q=2): {dim_mass:.4f}")

    return dim_binary, dim_mass


def test_spatial_weighting():
    """Test spatial weighting functionality."""
    print("\n" + "=" * 60)
    print("Testing spatial weighting")
    print("=" * 60)

    weights_prob = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights_prob, dim=2, size=2, levels=5)

    # Create spatial weights that emphasize the center
    y, x = np.ogrid[:field.shape[0], :field.shape[1]]
    center_y, center_x = field.shape[0] // 2, field.shape[1] // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    spatial_weights = np.exp(-distance / (field.shape[0] / 4))

    print("\nWithout spatial weights:")
    dim_unweighted = fractal_dimension(field, method='mass', q=2)

    print("\nWith center-focused weights:")
    dim_weighted = fractal_dimension(field, method='mass', q=2, weights=spatial_weights)

    print(f"\nUnweighted dimension: {dim_unweighted:.4f}")
    print(f"Weighted dimension: {dim_weighted:.4f}")

    return dim_unweighted, dim_weighted


def test_mask_random():
    """Test masking with random missing data."""
    print("\n" + "=" * 60)
    print("Testing masking (random missing data)")
    print("=" * 60)

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights, dim=2, size=2, levels=5)

    # Create a random mask with 80% observed
    np.random.seed(42)
    mask = np.random.random(field.shape) > 0.2  # 80% observed

    print(f"\nObserved fraction: {np.mean(mask):.2%}")

    print("\nWithout mask (full data):")
    dim_full = fractal_dimension(field, method='mass', q=2)

    print("\nWith random mask (80% observed):")
    dim_masked = fractal_dimension(
        field, method='mass', q=2, mask=mask, min_observed_fraction=0.3
    )

    print(f"\nFull data dimension: {dim_full:.4f}")
    print(f"Masked data dimension: {dim_masked:.4f}")
    print(f"Difference: {abs(dim_full - dim_masked):.4f}")

    return dim_full, dim_masked


def test_mask_rectangular():
    """Test masking with a rectangular unobserved region."""
    print("\n" + "=" * 60)
    print("Testing masking (rectangular missing region)")
    print("=" * 60)

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights, dim=2, size=2, levels=5)

    # Create a mask with a rectangular unobserved region
    mask = np.ones(field.shape, dtype=bool)
    # Mask out the bottom-right quadrant
    mask[field.shape[0]//2:, field.shape[1]//2:] = False

    print(f"\nObserved fraction: {np.mean(mask):.2%}")

    print("\nWithout mask (full data):")
    dim_full = fractal_dimension(field, method='binary')

    print("\nWith rectangular mask (75% observed):")
    dim_masked = fractal_dimension(
        field, method='binary', mask=mask, min_observed_fraction=0.5
    )

    print(f"\nFull data dimension: {dim_full:.4f}")
    print(f"Masked data dimension: {dim_masked:.4f}")
    print(f"Difference: {abs(dim_full - dim_masked):.4f}")

    return dim_full, dim_masked


def test_mask_3d():
    """Test masking in 3D."""
    print("\n" + "=" * 60)
    print("Testing masking (3D field)")
    print("=" * 60)

    weights = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15])
    field, _ = multifrac(weights, dim=3, size=2, levels=3)

    # Create a random mask with 70% observed
    np.random.seed(123)
    mask = np.random.random(field.shape) > 0.3  # 70% observed

    print(f"\nField shape: {field.shape}")
    print(f"Observed fraction: {np.mean(mask):.2%}")

    print("\nWithout mask (full data):")
    dim_full = fractal_dimension(field, method='mass', q=2)

    print("\nWith random mask (70% observed):")
    dim_masked = fractal_dimension(
        field, method='mass', q=2, mask=mask, min_observed_fraction=0.3
    )

    print(f"\nFull data dimension: {dim_full:.4f}")
    print(f"Masked data dimension: {dim_masked:.4f}")

    return dim_full, dim_masked


def test_combined_weights_and_mask():
    """Test using both weights and mask together."""
    print("\n" + "=" * 60)
    print("Testing combined weights and mask")
    print("=" * 60)

    weights_prob = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights_prob, dim=2, size=2, levels=5)

    # Create spatial weights
    y, x = np.ogrid[:field.shape[0], :field.shape[1]]
    center_y, center_x = field.shape[0] // 2, field.shape[1] // 2
    distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    spatial_weights = np.exp(-distance / (field.shape[0] / 4))

    # Create a random mask
    np.random.seed(42)
    mask = np.random.random(field.shape) > 0.2

    print(f"\nObserved fraction: {np.mean(mask):.2%}")

    print("\nBaseline (no weights, no mask):")
    dim_baseline = fractal_dimension(field, method='mass', q=2)

    print("\nWith both weights and mask:")
    dim_combined = fractal_dimension(
        field, method='mass', q=2,
        weights=spatial_weights, mask=mask,
        min_observed_fraction=0.3
    )

    print(f"\nBaseline dimension: {dim_baseline:.4f}")
    print(f"Combined dimension: {dim_combined:.4f}")

    return dim_baseline, dim_combined


def test_min_observed_fraction():
    """Test effect of min_observed_fraction parameter."""
    print("\n" + "=" * 60)
    print("Testing min_observed_fraction parameter")
    print("=" * 60)

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights, dim=2, size=2, levels=5)

    # Create a sparse mask (50% observed)
    np.random.seed(42)
    mask = np.random.random(field.shape) > 0.5

    print(f"\nObserved fraction: {np.mean(mask):.2%}")

    results = []
    for min_frac in [0.1, 0.3, 0.5, 0.7]:
        print(f"\nmin_observed_fraction = {min_frac}:")
        try:
            dim = fractal_dimension(
                field, method='binary', mask=mask,
                min_observed_fraction=min_frac
            )
            results.append((min_frac, dim))
        except ValueError as e:
            print(f"  Error: {e}")
            results.append((min_frac, None))

    print("\nSummary:")
    for min_frac, dim in results:
        if dim is not None:
            print(f"  min_frac={min_frac}: D={dim:.4f}")
        else:
            print(f"  min_frac={min_frac}: Failed")

    return results


def test_diagnostics():
    """Test diagnostics output with new features."""
    print("\n" + "=" * 60)
    print("Testing diagnostics output")
    print("=" * 60)

    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, _ = multifrac(weights, dim=2, size=2, levels=5)

    np.random.seed(42)
    mask = np.random.random(field.shape) > 0.2

    print("\nWith mask and diagnostics:")
    dim, diag = fractal_dimension(
        field, method='mass', q=2, mask=mask,
        min_observed_fraction=0.3, return_diagnostics=True
    )

    print("\nDiagnostics:")
    for key, value in diag.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: array of length {len(value)}")
        else:
            print(f"  {key}: {value}")

    return diag


if __name__ == "__main__":
    print("Testing unified fractal_dimension function")
    print("with weights and masking support\n")

    # Run all tests
    test_method_parameter()
    test_spatial_weighting()
    test_mask_random()
    test_mask_rectangular()
    test_mask_3d()
    test_combined_weights_and_mask()
    test_min_observed_fraction()
    test_diagnostics()

    # Summary
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
