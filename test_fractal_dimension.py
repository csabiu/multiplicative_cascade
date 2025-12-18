#!/usr/bin/env python3
"""Test script for fractal_dimension function."""

import numpy as np
from multifrac import multifrac, fractal_dimension


def test_2d_field():
    """Test fractal dimension on a 2D field."""
    print("=" * 60)
    print("Testing 2D Fractal Field")
    print("=" * 60)

    # Create a 2D fractal field
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    field, theoretical_slope = multifrac(weights, dim=2, size=2, levels=5)

    print(f"\nField shape: {field.shape}")
    print(f"Field range: [{field.min():.6f}, {field.max():.6f}]")
    print(f"Field mean: {field.mean():.6f}")

    # Measure fractal dimension
    print("\nMeasuring fractal dimension...")
    dimension, diag = fractal_dimension(field, return_diagnostics=True)

    print(f"\nDiagnostic information:")
    print(f"  Number of scales tested: {len(diag['box_sizes'])}")
    print(f"  Box sizes: {diag['box_sizes']}")
    print(f"  Box counts: {diag['box_counts']}")
    print(f"  Threshold used: {diag['threshold_used']:.6f}")

    return dimension


def test_3d_field():
    """Test fractal dimension on a 3D field."""
    print("\n" + "=" * 60)
    print("Testing 3D Fractal Field")
    print("=" * 60)

    # Create a 3D fractal field
    weights = np.array([0.05, 0.15, 0.25, 0.20, 0.15, 0.10, 0.05, 0.05])
    field, theoretical_slope = multifrac(weights, dim=3, size=2, levels=4)

    print(f"\nField shape: {field.shape}")
    print(f"Field range: [{field.min():.6f}, {field.max():.6f}]")
    print(f"Field mean: {field.mean():.6f}")

    # Measure fractal dimension
    print("\nMeasuring fractal dimension...")
    dimension, diag = fractal_dimension(field, return_diagnostics=True)

    print(f"\nDiagnostic information:")
    print(f"  Number of scales tested: {len(diag['box_sizes'])}")
    print(f"  Box sizes: {diag['box_sizes']}")
    print(f"  Box counts: {diag['box_counts']}")
    print(f"  Threshold used: {diag['threshold_used']:.6f}")

    return dimension


def test_uniform_field():
    """Test on a uniform field (should give dimension equal to field dimensionality)."""
    print("\n" + "=" * 60)
    print("Testing Uniform Field (Sanity Check)")
    print("=" * 60)

    # Create a uniform field
    field = np.ones((64, 64))
    print(f"\nField shape: {field.shape}")
    print(f"Field is uniform with value: {field[0, 0]}")

    # Measure fractal dimension with explicit threshold
    print("\nMeasuring fractal dimension...")
    dimension = fractal_dimension(field, threshold=0.5)

    print(f"\nExpected dimension: ~2.0 (fills all of 2D space)")
    print(f"Measured dimension: {dimension:.4f}")

    return dimension


if __name__ == "__main__":
    print("Testing fractal_dimension function\n")

    # Run tests
    dim_2d = test_2d_field()
    dim_3d = test_3d_field()
    dim_uniform = test_uniform_field()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"2D fractal field dimension: {dim_2d:.4f}")
    print(f"3D fractal field dimension: {dim_3d:.4f}")
    print(f"Uniform field dimension: {dim_uniform:.4f}")
    print("\nAll tests completed successfully!")
