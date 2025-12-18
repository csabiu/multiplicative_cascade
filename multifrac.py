import numpy as np
from scipy.stats import poisson


def multifrac(probabilities, dim=2, size=2, levels=4, add_power=False):
    """
    Generate a multiplicative cascade fractal field.

    This function creates a multifractal density field using a multiplicative
    cascade process. At each level, the field is subdivided and multiplied by
    randomly permuted probability weights.

    Parameters
    ----------
    probabilities : array-like
        Probability weights for the cascade. Should sum to approximately 1.0
        for proper normalization. The number of elements should equal size**dim.
    dim : int, optional
        Dimensionality of the output field (1, 2, or 3). Default is 2.
    size : int, optional
        Subdivision factor at each level (e.g., 2 means each cell splits into
        2^dim subcells). Default is 2.
    levels : int, optional
        Number of cascade levels to generate. The final grid will have
        size**(levels+1) cells per dimension. Default is 4.
    add_power : bool, optional
        If True, applies increasing power weights at each level to modify
        the fractal dimension. Default is False.

    Returns
    -------
    cascade_field : ndarray
        The generated multiplicative cascade field with shape determined by
        (size**(levels+1),) * dim
    theoretical_slope : float
        The theoretical power law slope (fractal dimension)

    Raises
    ------
    ValueError
        If dim is not in [1, 2, 3] or if parameters are invalid

    Examples
    --------
    >>> weights = np.array([0.25, 0.25, 0.25, 0.25])
    >>> field, slope = multifrac(weights, dim=2, size=2, levels=3)
    >>> print(field.shape)
    (16, 16)
    """
    # Input validation
    if dim not in [1, 2, 3]:
        raise ValueError(f'Dimension must be 1, 2, or 3. Got: {dim}')

    if size <= 0:
        raise ValueError(f'Size must be positive. Got: {size}')

    if levels <= 0:
        raise ValueError(f'Levels must be positive. Got: {levels}')

    probabilities = np.asarray(probabilities)
    expected_length = size ** dim
    if len(probabilities) != expected_length:
        raise ValueError(
            f'Length of probabilities ({len(probabilities)}) must equal '
            f'size^dim ({expected_length})'
        )

    # Initialize grid shape
    grid_shape = np.tile(size, dim)
    power_increment = 1.0 if add_power else 0.0

    # Initialize the cascade field with randomly permuted probabilities
    cascade_field = np.random.permutation(probabilities).reshape(grid_shape)

    # Perform the multiplicative cascade
    for level in range(levels):
        grid_size = size ** (level + 2)
        new_field = np.zeros(np.tile(grid_size, dim))

        # Apply subdivision based on dimensionality
        if dim == 1:
            for i in range(0, grid_size, size):
                power = power_increment * level + 1
                new_field[i:i+size] = (
                    np.random.permutation(probabilities).reshape(grid_shape) ** power
                )
        elif dim == 2:
            for i in range(0, grid_size, size):
                for j in range(0, grid_size, size):
                    power = power_increment * level + 1
                    new_field[i:i+size, j:j+size] = (
                        np.random.permutation(probabilities).reshape(grid_shape) ** power
                    )
        elif dim == 3:
            for i in range(0, grid_size, size):
                for j in range(0, grid_size, size):
                    for k in range(0, grid_size, size):
                        power = power_increment * level + 1
                        new_field[i:i+size, j:j+size, k:k+size] = (
                            np.random.permutation(probabilities).reshape(grid_shape) ** power
                        )

        # Expand the existing field to match the new grid size
        for d in range(dim):
            cascade_field = cascade_field.repeat(size, axis=d)

        # Multiply the fields
        cascade_field = cascade_field * new_field

    # Calculate theoretical power law slope (fractal dimension)
    normalized_prob = probabilities / np.sum(probabilities)
    if add_power:
        # Empirical formula for add_power mode
        # The 0.5 offset is based on experimental observations
        theoretical_slope = np.log2(np.sum(normalized_prob ** levels)) / (levels + 0.5)
    else:
        # Standard box-counting dimension formula
        theoretical_slope = np.log2(np.sum(normalized_prob ** 2))

    print(f'Power law slope (theory) = {theoretical_slope:.6f}')

    return cascade_field, theoretical_slope


def fractal_dimension(field, threshold=None, min_box_size=2, max_box_size=None,
                      num_scales=10, return_diagnostics=False):
    """
    Measure the fractal dimension of a 2D or 3D field using the box-counting method.

    This function estimates the fractal (box-counting) dimension by counting how
    many boxes of varying sizes contain values above a threshold. The dimension
    is computed from the power-law relationship: N(ε) ~ ε^(-D), where N is the
    number of occupied boxes and ε is the box size.

    Parameters
    ----------
    field : ndarray
        Input field (2D or 3D array) to analyze.
    threshold : float, optional
        Threshold value for determining occupied boxes. Boxes with mean values
        above this threshold are counted as occupied. If None, uses the median
        of non-zero values. Default is None.
    min_box_size : int, optional
        Minimum box size to use (in pixels/voxels). Must be >= 2. Default is 2.
    max_box_size : int, optional
        Maximum box size to use. If None, uses half the smallest field dimension.
        Default is None.
    num_scales : int, optional
        Number of different box sizes to test. Default is 10.
    return_diagnostics : bool, optional
        If True, returns additional diagnostic information including box sizes,
        counts, and fit quality. Default is False.

    Returns
    -------
    dimension : float
        The estimated fractal dimension.
    diagnostics : dict, optional
        Only returned if return_diagnostics=True. Contains:
        - 'box_sizes': array of box sizes used
        - 'box_counts': array of occupied box counts
        - 'slope': fitted slope (-dimension)
        - 'intercept': y-intercept of the fit
        - 'r_squared': coefficient of determination for the fit

    Raises
    ------
    ValueError
        If field is not 2D or 3D, or if parameters are invalid

    Examples
    --------
    >>> # Create a simple fractal field
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> field, _ = multifrac(weights, dim=2, size=2, levels=5)
    >>> dimension = fractal_dimension(field)
    >>> print(f"Fractal dimension: {dimension:.3f}")

    >>> # Get detailed diagnostics
    >>> dim, diag = fractal_dimension(field, return_diagnostics=True)
    >>> print(f"R² of fit: {diag['r_squared']:.4f}")
    """
    # Input validation
    field = np.asarray(field)
    dim = len(field.shape)

    if dim not in [2, 3]:
        raise ValueError(f'Field must be 2D or 3D. Got shape: {field.shape}')

    if min_box_size < 2:
        raise ValueError(f'min_box_size must be >= 2. Got: {min_box_size}')

    # Set default threshold if not provided
    if threshold is None:
        non_zero = field[field > 0]
        if len(non_zero) > 0:
            threshold = np.median(non_zero)
        else:
            threshold = 0.0

    # Determine max box size
    min_field_dim = min(field.shape)
    if max_box_size is None:
        max_box_size = min_field_dim // 2

    if max_box_size > min_field_dim:
        raise ValueError(
            f'max_box_size ({max_box_size}) cannot exceed minimum field '
            f'dimension ({min_field_dim})'
        )

    if min_box_size >= max_box_size:
        raise ValueError(
            f'min_box_size ({min_box_size}) must be less than max_box_size '
            f'({max_box_size})'
        )

    # Generate logarithmically-spaced box sizes
    box_sizes = np.logspace(
        np.log10(min_box_size),
        np.log10(max_box_size),
        num=num_scales,
        dtype=int
    )
    box_sizes = np.unique(box_sizes)  # Remove duplicates from rounding

    box_counts = []

    # Count occupied boxes for each box size
    for box_size in box_sizes:
        count = 0

        if dim == 2:
            # 2D case
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    # Extract box, handling edge cases
                    box = field[i:i+box_size, j:j+box_size]
                    # Count box as occupied if mean exceeds threshold
                    if np.mean(box) > threshold:
                        count += 1

        else:  # dim == 3
            # 3D case
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    for k in range(0, field.shape[2], box_size):
                        # Extract box, handling edge cases
                        box = field[i:i+box_size, j:j+box_size, k:k+box_size]
                        # Count box as occupied if mean exceeds threshold
                        if np.mean(box) > threshold:
                            count += 1

        box_counts.append(count)

    box_counts = np.array(box_counts)

    # Remove any zero counts (would cause log issues)
    valid_mask = box_counts > 0
    box_sizes_valid = box_sizes[valid_mask]
    box_counts_valid = box_counts[valid_mask]

    if len(box_sizes_valid) < 2:
        raise ValueError(
            'Not enough valid data points for fractal dimension estimation. '
            'Try adjusting threshold or box size parameters.'
        )

    # Fit log(N) vs log(1/ε) to get dimension
    # N(ε) ~ ε^(-D) => log(N) = -D * log(ε) + const
    log_scales = np.log(box_sizes_valid)
    log_counts = np.log(box_counts_valid)

    # Linear regression
    coeffs = np.polyfit(log_scales, log_counts, 1)
    slope, intercept = coeffs

    # The fractal dimension is the negative of the slope
    dimension = -slope

    # Calculate R² for goodness of fit
    log_counts_pred = slope * log_scales + intercept
    ss_res = np.sum((log_counts - log_counts_pred) ** 2)
    ss_tot = np.sum((log_counts - np.mean(log_counts)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f'Fractal dimension (box-counting): {dimension:.4f}')
    print(f'Fit quality (R²): {r_squared:.4f}')

    if return_diagnostics:
        diagnostics = {
            'box_sizes': box_sizes_valid,
            'box_counts': box_counts_valid,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'threshold_used': threshold
        }
        return dimension, diagnostics

    return dimension


def fractal_dimension_mass(field, min_box_size=2, max_box_size=None,
                           num_scales=10, q=2, return_diagnostics=False):
    """
    Measure the mass-based fractal dimension of a 2D or 3D field.

    This function estimates the fractal dimension by accounting for the field
    intensity (mass) rather than just spatial occupancy. The field is normalized
    to sum to 1, treating it as a mass/probability distribution. The generalized
    dimension D_q is computed from: Sum_i(m_i^q) ~ ε^((q-1)*D_q), where m_i is
    the mass in box i and ε is the box size.

    Parameters
    ----------
    field : ndarray
        Input field (2D or 3D array) to analyze. Should contain non-negative values.
    min_box_size : int, optional
        Minimum box size to use (in pixels/voxels). Must be >= 2. Default is 2.
    max_box_size : int, optional
        Maximum box size to use. If None, uses half the smallest field dimension.
        Default is None.
    num_scales : int, optional
        Number of different box sizes to test. Default is 10.
    q : float, optional
        Order of the generalized dimension. Common values:
        - q=0: Capacity dimension (counts non-empty boxes)
        - q=1: Information dimension (entropy-based, experimental)
        - q=2: Correlation dimension (default, recommended)
        - q=3: Higher-order dimension
        Higher q values emphasize high-density regions. Default is 2.
    return_diagnostics : bool, optional
        If True, returns additional diagnostic information including box sizes,
        partition sums, and fit quality. Default is False.

    Returns
    -------
    dimension : float
        The estimated mass-based fractal dimension D_q.
    diagnostics : dict, optional
        Only returned if return_diagnostics=True. Contains:
        - 'box_sizes': array of box sizes used
        - 'partition_sums': array of partition function values
        - 'slope': fitted slope ((q-1)*D_q)
        - 'intercept': y-intercept of the fit
        - 'r_squared': coefficient of determination for the fit
        - 'q': the q value used

    Raises
    ------
    ValueError
        If field is not 2D or 3D, contains negative values, or parameters are invalid

    Examples
    --------
    >>> # Create a multifractal field
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> field, _ = multifrac(weights, dim=2, size=2, levels=5)
    >>> dimension = fractal_dimension_mass(field, q=2)
    >>> print(f"Mass-based fractal dimension: {dimension:.3f}")

    >>> # Compare different q values
    >>> d0 = fractal_dimension_mass(field, q=0)  # Capacity dimension
    >>> d2 = fractal_dimension_mass(field, q=2)  # Correlation dimension
    """
    # Input validation
    field = np.asarray(field)
    dim = len(field.shape)

    if dim not in [2, 3]:
        raise ValueError(f'Field must be 2D or 3D. Got shape: {field.shape}')

    if np.any(field < 0):
        raise ValueError('Field must contain non-negative values for mass-based analysis')

    if min_box_size < 2:
        raise ValueError(f'min_box_size must be >= 2. Got: {min_box_size}')

    # Normalize the field to sum to 1 (treat as mass distribution)
    total_mass = np.sum(field)
    if total_mass == 0:
        raise ValueError('Field has zero total mass')

    field_norm = field / total_mass

    # Determine max box size
    min_field_dim = min(field.shape)
    if max_box_size is None:
        max_box_size = min_field_dim // 2

    if max_box_size > min_field_dim:
        raise ValueError(
            f'max_box_size ({max_box_size}) cannot exceed minimum field '
            f'dimension ({min_field_dim})'
        )

    if min_box_size >= max_box_size:
        raise ValueError(
            f'min_box_size ({min_box_size}) must be less than max_box_size '
            f'({max_box_size})'
        )

    # Generate logarithmically-spaced box sizes
    box_sizes = np.logspace(
        np.log10(min_box_size),
        np.log10(max_box_size),
        num=num_scales,
        dtype=int
    )
    box_sizes = np.unique(box_sizes)  # Remove duplicates from rounding

    partition_sums = []

    # Calculate partition function for each box size
    for box_size in box_sizes:
        masses = []

        if dim == 2:
            # 2D case
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    # Extract box and sum the mass
                    box = field_norm[i:i+box_size, j:j+box_size]
                    mass = np.sum(box)
                    if mass > 0:  # Only include non-empty boxes
                        masses.append(mass)

        else:  # dim == 3
            # 3D case
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    for k in range(0, field.shape[2], box_size):
                        # Extract box and sum the mass
                        box = field_norm[i:i+box_size, j:j+box_size, k:k+box_size]
                        mass = np.sum(box)
                        if mass > 0:  # Only include non-empty boxes
                            masses.append(mass)

        masses = np.array(masses)

        # Calculate partition function: Z_q(ε) = Sum_i(m_i^q)
        if q == 1:
            # Special case: q=1 uses information/entropy formulation
            # For q=1, we use: Sum_i(m_i * log(m_i))
            masses_nonzero = masses[masses > 0]
            if len(masses_nonzero) > 0:
                # Information sum (will be negative, but we track the sum itself)
                info_sum = np.sum(masses_nonzero * np.log(masses_nonzero))
                partition_sum = info_sum  # Store as-is for linear fit
            else:
                partition_sum = 0
        else:
            # General case: Z_q = Sum_i(m_i^q)
            partition_sum = np.sum(masses ** q)

        partition_sums.append(partition_sum)

    partition_sums = np.array(partition_sums)

    # Handle different cases for q
    if q == 1:
        # For q=1, partition sums are negative (information content)
        # We fit the information sum directly vs log(ε)
        valid_mask = partition_sums != 0
        box_sizes_valid = box_sizes[valid_mask]
        partition_sums_valid = partition_sums[valid_mask]

        if len(box_sizes_valid) < 2:
            raise ValueError(
                'Not enough valid data points for dimension estimation. '
                'Try adjusting box size parameters.'
            )

        log_scales = np.log(box_sizes_valid)
        # For q=1, we fit I(ε) = Sum(p_i*log(p_i)) vs log(ε)
        # D_1 = -dI/d(log ε)
        log_partition = partition_sums_valid  # Don't take log, use values directly
    else:
        # Remove any zero or negative partition sums (would cause log issues)
        valid_mask = partition_sums > 0
        box_sizes_valid = box_sizes[valid_mask]
        partition_sums_valid = partition_sums[valid_mask]

        if len(box_sizes_valid) < 2:
            raise ValueError(
                'Not enough valid data points for dimension estimation. '
                'Try adjusting box size parameters.'
            )

        # Fit log(Z_q) vs log(ε) to get dimension
        # Z_q(ε) ~ ε^(τ(q)) where τ(q) = (q-1)*D_q
        # log(Z_q) = τ(q) * log(ε) + const
        log_scales = np.log(box_sizes_valid)
        log_partition = np.log(partition_sums_valid)

    # Linear regression
    coeffs = np.polyfit(log_scales, log_partition, 1)
    slope, intercept = coeffs

    # The dimension D_q is related to slope by: slope = (q-1)*D_q
    if q == 1:
        # For q=1 (information dimension), D_1 = -slope
        # since we fit I(ε) vs log(ε) where I = Sum(p_i*log(p_i))
        dimension = -slope
    else:
        dimension = slope / (q - 1)

    # Calculate R² for goodness of fit
    log_partition_pred = slope * log_scales + intercept
    ss_res = np.sum((log_partition - log_partition_pred) ** 2)
    ss_tot = np.sum((log_partition - np.mean(log_partition)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    print(f'Mass-based fractal dimension D_{q} = {dimension:.4f}')
    print(f'Fit quality (R²): {r_squared:.4f}')

    if return_diagnostics:
        diagnostics = {
            'box_sizes': box_sizes_valid,
            'partition_sums': partition_sums_valid,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'q': q
        }
        return dimension, diagnostics

    return dimension


def mock(density=None, boxsize=100, Npart=100):
    """
    Generate mock particle distributions from a density field.

    Creates a Poisson-sampled particle distribution from an input density field,
    along with a uniform random sample for comparison.

    Parameters
    ----------
    density : ndarray, optional
        Input density field (2D or 3D array). If None, must be provided.
    boxsize : float, optional
        Physical size of the simulation box. Default is 100.
    Npart : int, optional
        Target number of particles to generate. Default is 100.

    Returns
    -------
    points : ndarray
        Positions of particles sampled from the density field.
        Shape is (n_particles, dim) where dim is 2 or 3.
    randoms : ndarray
        Uniform random particle positions for comparison.
        Shape is (10*Npart, dim).

    Raises
    ------
    ValueError
        If density is None or has invalid shape

    Examples
    --------
    >>> density_field = np.ones((32, 32))
    >>> particles, randoms = mock(density_field, boxsize=100, Npart=1000)
    """
    # Input validation
    if density is None or len(density) == 0:
        raise ValueError('Density field must be provided')

    density = np.asarray(density)
    dim = len(density.shape)

    if dim not in [2, 3]:
        raise ValueError(f'Density field must be 2D or 3D. Got shape: {density.shape}')

    Nmesh = density.shape[0]
    cell_size = boxsize / Nmesh

    # Poisson sample the density field
    density_counts = poisson.rvs(Npart * density / np.sum(density))

    # Random sample multiplier (for generating comparison uniform random sample)
    RANDOM_SAMPLE_MULTIPLIER = 10

    # Generate particle positions
    points_list = []

    if dim == 3:
        for i in range(Nmesh):
            for j in range(Nmesh):
                for k in range(Nmesh):
                    n_particles = density_counts[i, j, k]
                    if n_particles > 0:
                        xpoints = np.random.uniform(
                            low=i*cell_size,
                            high=(i+1)*cell_size,
                            size=n_particles
                        )
                        ypoints = np.random.uniform(
                            low=j*cell_size,
                            high=(j+1)*cell_size,
                            size=n_particles
                        )
                        zpoints = np.random.uniform(
                            low=k*cell_size,
                            high=(k+1)*cell_size,
                            size=n_particles
                        )
                        cell_points = np.column_stack((xpoints, ypoints, zpoints))
                        points_list.append(cell_points)

        randoms = np.random.uniform(
            low=0.0,
            high=boxsize,
            size=(RANDOM_SAMPLE_MULTIPLIER * Npart, 3)
        )

    else:  # dim == 2
        for i in range(Nmesh):
            for j in range(Nmesh):
                n_particles = density_counts[i, j]
                if n_particles > 0:
                    xpoints = np.random.uniform(
                        low=i*cell_size,
                        high=(i+1)*cell_size,
                        size=n_particles
                    )
                    ypoints = np.random.uniform(
                        low=j*cell_size,
                        high=(j+1)*cell_size,
                        size=n_particles
                    )
                    cell_points = np.column_stack((xpoints, ypoints))
                    points_list.append(cell_points)

        randoms = np.random.uniform(
            low=0.0,
            high=boxsize,
            size=(RANDOM_SAMPLE_MULTIPLIER * Npart, 2)
        )

    # Combine all points
    if len(points_list) > 0:
        points = np.vstack(points_list)
    else:
        # If no particles were generated, return empty array with correct shape
        points = np.empty((0, dim))

    print(f'Number of point samples: {points.shape[0]}')

    return points, randoms
