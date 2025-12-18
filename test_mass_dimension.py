#!/usr/bin/env python3
"""Test script to compare binary and mass-based box counting methods."""

import numpy as np
from multifrac import multifrac, fractal_dimension, fractal_dimension_mass

# Create a multifractal field with varying intensities
print("="*60)
print("Testing Mass-Based Box Counting")
print("="*60)

# Test with a non-uniform cascade (more interesting multifractal)
weights = np.array([0.1, 0.2, 0.3, 0.4])
print(f"\nGenerating 2D multifractal field with weights: {weights}")
print(f"Sum of weights: {np.sum(weights):.2f}")

field, theoretical_slope = multifrac(weights, dim=2, size=2, levels=5)
print(f"\nField shape: {field.shape}")
print(f"Field min: {field.min():.6f}, max: {field.max():.6f}")
print(f"Field mean: {field.mean():.6f}, sum: {field.sum():.6f}")

print("\n" + "="*60)
print("Binary Box Counting (standard method)")
print("="*60)
dim_binary = fractal_dimension(field, num_scales=15)

print("\n" + "="*60)
print("Mass-Based Box Counting (q=2)")
print("="*60)
dim_mass = fractal_dimension_mass(field, num_scales=15, q=2)

print("\n" + "="*60)
print("Comparison of Different q Values")
print("="*60)
print("\nTesting different orders of generalized dimensions:")
print("(Note: q=1 requires special handling and is omitted here)")
for q in [0, 2, 3]:
    print(f"\nq = {q}:")
    dim_q = fractal_dimension_mass(field, num_scales=15, q=q)

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"Theoretical slope: {theoretical_slope:.4f}")
print(f"Binary box counting dimension: {dim_binary:.4f}")
print(f"Mass-based dimension (q=2): {dim_mass:.4f}")
print("\nThe mass-based method accounts for field intensity,")
print("while binary method only looks at spatial occupancy.")

# Test with 3D field
print("\n" + "="*60)
print("Testing 3D Field")
print("="*60)
weights_3d = np.array([0.05, 0.1, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15])
field_3d, _ = multifrac(weights_3d, dim=3, size=2, levels=3)
print(f"\n3D Field shape: {field_3d.shape}")

print("\nBinary method (3D):")
dim_binary_3d = fractal_dimension(field_3d, num_scales=8)

print("\nMass-based method (3D, q=2):")
dim_mass_3d = fractal_dimension_mass(field_3d, num_scales=8, q=2)

print("\n" + "="*60)
print("Test completed successfully!")
print("="*60)
