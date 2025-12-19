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
                      num_scales=10, method='binary', q=2, weights=None,
                      mask=None, min_observed_fraction=0.5,
                      return_diagnostics=False):
    """
    Measure the fractal dimension of a 2D or 3D field using box-counting methods.

    This unified function supports both binary (threshold-based) and mass-based
    box-counting methods for estimating fractal dimensions. It also supports
    spatial weighting and masking for handling incomplete observations.

    Parameters
    ----------
    field : ndarray
        Input field (2D or 3D array) to analyze.
    threshold : float, optional
        Threshold value for determining occupied boxes (binary method only).
        Boxes with mean values above this threshold are counted as occupied.
        If None, uses the median of non-zero values. Default is None.
    min_box_size : int, optional
        Minimum box size to use (in pixels/voxels). Must be >= 2. Default is 2.
    max_box_size : int, optional
        Maximum box size to use. If None, uses half the smallest field dimension.
        Default is None.
    num_scales : int, optional
        Number of different box sizes to test. Default is 10.
    method : str, optional
        Method for dimension estimation:
        - 'binary': Threshold-based box counting. Counts boxes where mean > threshold.
        - 'mass': Mass-based method using partition functions. Accounts for field
          intensity rather than just spatial occupancy.
        Default is 'binary'.
    q : float, optional
        Order of the generalized dimension (mass method only). Common values:
        - q=0: Capacity dimension (counts non-empty boxes)
        - q=1: Information dimension (entropy-based)
        - q=2: Correlation dimension (default, recommended)
        Higher q values emphasize high-density regions. Default is 2.
    weights : ndarray, optional
        Spatial weight array with same shape as field. Each pixel/voxel's
        contribution is multiplied by its weight. Useful for emphasizing
        certain regions. If None, uniform weighting is used. Default is None.
    mask : ndarray, optional
        Boolean or binary array with same shape as field indicating observed
        regions. True/1 = observed, False/0 = masked/unobserved. The function
        corrects for partial coverage in each box. If None, all regions are
        considered observed. Default is None.
    min_observed_fraction : float, optional
        Minimum fraction of a box that must be observed (unmasked) to include
        it in the analysis. Boxes with less coverage are excluded. Must be
        between 0 and 1. Default is 0.5.
    return_diagnostics : bool, optional
        If True, returns additional diagnostic information. Default is False.

    Returns
    -------
    dimension : float
        The estimated fractal dimension.
    diagnostics : dict, optional
        Only returned if return_diagnostics=True. Contains:
        - 'box_sizes': array of box sizes used
        - 'box_counts' or 'partition_sums': array of counts/partition values
        - 'slope': fitted slope
        - 'intercept': y-intercept of the fit
        - 'r_squared': coefficient of determination for the fit
        - 'method': the method used
        - 'threshold_used': threshold value (binary method only)
        - 'q': the q value (mass method only)
        - 'mask_correction_applied': whether mask correction was used

    Raises
    ------
    ValueError
        If field is not 2D or 3D, or if parameters are invalid

    Examples
    --------
    >>> # Binary box-counting (default)
    >>> weights = np.array([0.1, 0.2, 0.3, 0.4])
    >>> field, _ = multifrac(weights, dim=2, size=2, levels=5)
    >>> dimension = fractal_dimension(field)

    >>> # Mass-based method
    >>> dimension = fractal_dimension(field, method='mass', q=2)

    >>> # With spatial weighting
    >>> weights = np.exp(-field)  # Example: inverse intensity weighting
    >>> dimension = fractal_dimension(field, weights=weights)

    >>> # With mask for unobserved regions
    >>> mask = np.random.random(field.shape) > 0.2  # 80% observed
    >>> dimension = fractal_dimension(field, mask=mask, min_observed_fraction=0.3)
    """
    # Input validation
    field = np.asarray(field, dtype=float)
    ndim = len(field.shape)

    if ndim not in [2, 3]:
        raise ValueError(f'Field must be 2D or 3D. Got shape: {field.shape}')

    if min_box_size < 2:
        raise ValueError(f'min_box_size must be >= 2. Got: {min_box_size}')

    if method not in ['binary', 'mass']:
        raise ValueError(f"method must be 'binary' or 'mass'. Got: {method}")

    if not 0 <= min_observed_fraction <= 1:
        raise ValueError(
            f'min_observed_fraction must be between 0 and 1. Got: {min_observed_fraction}'
        )

    # Validate and process weights
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != field.shape:
            raise ValueError(
                f'weights shape {weights.shape} must match field shape {field.shape}'
            )
        if np.any(weights < 0):
            raise ValueError('weights must be non-negative')

    # Validate and process mask
    mask_applied = False
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != field.shape:
            raise ValueError(
                f'mask shape {mask.shape} must match field shape {field.shape}'
            )
        mask_applied = True
        # Check that we have some observed data
        if not np.any(mask):
            raise ValueError('mask must have at least some True (observed) values')
    else:
        # No mask - all pixels observed
        mask = np.ones(field.shape, dtype=bool)

    # Apply weights to field if provided
    if weights is not None:
        weighted_field = field * weights
    else:
        weighted_field = field.copy()

    # For mass method, validate field values
    if method == 'mass':
        if np.any(weighted_field[mask] < 0):
            raise ValueError('Field must contain non-negative values for mass-based analysis')
        # Normalize to sum to 1 (treating as mass distribution), considering only observed
        total_mass = np.sum(weighted_field[mask])
        if total_mass == 0:
            raise ValueError('Field has zero total mass in observed regions')
        field_norm = weighted_field / total_mass
    else:
        field_norm = weighted_field

    # Set default threshold for binary method
    if method == 'binary' and threshold is None:
        observed_values = field_norm[mask]
        non_zero = observed_values[observed_values > 0]
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

    # Process boxes based on method
    if method == 'binary':
        result_values = _binary_box_counting(
            field_norm, mask, box_sizes, threshold, min_observed_fraction, ndim
        )
        result_key = 'box_counts'
    else:  # method == 'mass'
        result_values = _mass_box_counting(
            field_norm, mask, box_sizes, q, min_observed_fraction, ndim
        )
        result_key = 'partition_sums'

    result_values = np.array(result_values)

    # Handle fitting based on method
    if method == 'binary':
        # Remove any zero counts
        valid_idx = result_values > 0
        box_sizes_valid = box_sizes[valid_idx]
        values_valid = result_values[valid_idx]

        if len(box_sizes_valid) < 2:
            raise ValueError(
                'Not enough valid data points for fractal dimension estimation. '
                'Try adjusting threshold or box size parameters.'
            )

        log_scales = np.log(box_sizes_valid)
        log_values = np.log(values_valid)

        # Linear regression: log(N) = -D * log(ε) + const
        coeffs = np.polyfit(log_scales, log_values, 1)
        slope, intercept = coeffs
        dimension = -slope

    else:  # method == 'mass'
        if q == 1:
            # For q=1 (information dimension), values are already log-like
            valid_idx = result_values != 0
            box_sizes_valid = box_sizes[valid_idx]
            values_valid = result_values[valid_idx]

            if len(box_sizes_valid) < 2:
                raise ValueError(
                    'Not enough valid data points for dimension estimation. '
                    'Try adjusting box size parameters.'
                )

            log_scales = np.log(box_sizes_valid)
            log_values = values_valid  # Already in log-form for q=1
        else:
            # General case
            valid_idx = result_values > 0
            box_sizes_valid = box_sizes[valid_idx]
            values_valid = result_values[valid_idx]

            if len(box_sizes_valid) < 2:
                raise ValueError(
                    'Not enough valid data points for dimension estimation. '
                    'Try adjusting box size parameters.'
                )

            log_scales = np.log(box_sizes_valid)
            log_values = np.log(values_valid)

        # Linear regression
        coeffs = np.polyfit(log_scales, log_values, 1)
        slope, intercept = coeffs

        # Extract dimension based on q
        if q == 1:
            dimension = -slope
        else:
            dimension = slope / (q - 1)

    # Calculate R² for goodness of fit
    log_values_pred = slope * log_scales + intercept
    ss_res = np.sum((log_values - log_values_pred) ** 2)
    ss_tot = np.sum((log_values - np.mean(log_values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if method == 'binary':
        print(f'Fractal dimension (box-counting): {dimension:.4f}')
    else:
        print(f'Mass-based fractal dimension D_{q} = {dimension:.4f}')
    print(f'Fit quality (R²): {r_squared:.4f}')
    if mask_applied:
        print(f'Mask correction applied (min_observed_fraction={min_observed_fraction})')

    if return_diagnostics:
        diagnostics = {
            'box_sizes': box_sizes_valid,
            result_key: values_valid,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'method': method,
            'mask_correction_applied': mask_applied
        }
        if method == 'binary':
            diagnostics['threshold_used'] = threshold
        else:
            diagnostics['q'] = q

        return dimension, diagnostics

    return dimension


def _binary_box_counting(field, mask, box_sizes, threshold, min_observed_fraction, ndim):
    """
    Perform binary box counting with mask support.

    Parameters
    ----------
    field : ndarray
        The field to analyze.
    mask : ndarray
        Boolean mask where True = observed.
    box_sizes : array-like
        Box sizes to use.
    threshold : float
        Threshold for occupancy.
    min_observed_fraction : float
        Minimum fraction of box that must be observed.
    ndim : int
        Number of dimensions (2 or 3).

    Returns
    -------
    box_counts : list
        Count of occupied boxes for each box size.
    """
    box_counts = []

    for box_size in box_sizes:
        count = 0

        if ndim == 2:
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    box_field = field[i:i+box_size, j:j+box_size]
                    box_mask = mask[i:i+box_size, j:j+box_size]

                    # Calculate observed fraction
                    observed_fraction = np.mean(box_mask)
                    if observed_fraction < min_observed_fraction:
                        continue  # Skip boxes with insufficient coverage

                    # Calculate mean of observed pixels only
                    observed_pixels = box_field[box_mask]
                    if len(observed_pixels) > 0:
                        box_mean = np.mean(observed_pixels)
                        if box_mean > threshold:
                            count += 1
        else:  # ndim == 3
            for i in range(0, field.shape[0], box_size):
                for j in range(0, field.shape[1], box_size):
                    for k in range(0, field.shape[2], box_size):
                        box_field = field[i:i+box_size, j:j+box_size, k:k+box_size]
                        box_mask = mask[i:i+box_size, j:j+box_size, k:k+box_size]

                        # Calculate observed fraction
                        observed_fraction = np.mean(box_mask)
                        if observed_fraction < min_observed_fraction:
                            continue

                        # Calculate mean of observed pixels only
                        observed_pixels = box_field[box_mask]
                        if len(observed_pixels) > 0:
                            box_mean = np.mean(observed_pixels)
                            if box_mean > threshold:
                                count += 1

        box_counts.append(count)

    return box_counts


def _mass_box_counting(field_norm, mask, box_sizes, q, min_observed_fraction, ndim):
    """
    Perform mass-based box counting with mask support.

    The mask correction works by estimating the total mass in each box from
    the observed portion. If a box has observed fraction f, the observed mass
    m_obs is scaled by 1/f to estimate the total mass: m_est = m_obs / f.

    Parameters
    ----------
    field_norm : ndarray
        The normalized field (sums to 1 over observed region).
    mask : ndarray
        Boolean mask where True = observed.
    box_sizes : array-like
        Box sizes to use.
    q : float
        Order of generalized dimension.
    min_observed_fraction : float
        Minimum fraction of box that must be observed.
    ndim : int
        Number of dimensions (2 or 3).

    Returns
    -------
    partition_sums : list
        Partition function values for each box size.
    """
    partition_sums = []

    for box_size in box_sizes:
        masses = []

        if ndim == 2:
            for i in range(0, field_norm.shape[0], box_size):
                for j in range(0, field_norm.shape[1], box_size):
                    box_field = field_norm[i:i+box_size, j:j+box_size]
                    box_mask = mask[i:i+box_size, j:j+box_size]

                    # Calculate observed fraction
                    observed_fraction = np.mean(box_mask)
                    if observed_fraction < min_observed_fraction:
                        continue

                    # Sum observed mass and correct for coverage
                    observed_mass = np.sum(box_field[box_mask])
                    if observed_mass > 0:
                        # Correct mass for partial coverage
                        corrected_mass = observed_mass / observed_fraction
                        masses.append(corrected_mass)
        else:  # ndim == 3
            for i in range(0, field_norm.shape[0], box_size):
                for j in range(0, field_norm.shape[1], box_size):
                    for k in range(0, field_norm.shape[2], box_size):
                        box_field = field_norm[i:i+box_size, j:j+box_size, k:k+box_size]
                        box_mask = mask[i:i+box_size, j:j+box_size, k:k+box_size]

                        # Calculate observed fraction
                        observed_fraction = np.mean(box_mask)
                        if observed_fraction < min_observed_fraction:
                            continue

                        # Sum observed mass and correct for coverage
                        observed_mass = np.sum(box_field[box_mask])
                        if observed_mass > 0:
                            corrected_mass = observed_mass / observed_fraction
                            masses.append(corrected_mass)

        masses = np.array(masses)

        # Calculate partition function: Z_q(ε) = Sum_i(m_i^q)
        if len(masses) == 0:
            partition_sums.append(0)
        elif q == 1:
            # Special case: q=1 uses information/entropy formulation
            masses_nonzero = masses[masses > 0]
            if len(masses_nonzero) > 0:
                # Renormalize masses for entropy calculation
                masses_renorm = masses_nonzero / np.sum(masses_nonzero)
                info_sum = np.sum(masses_renorm * np.log(masses_renorm))
                partition_sums.append(info_sum)
            else:
                partition_sums.append(0)
        else:
            # General case: Z_q = Sum_i(m_i^q)
            # Renormalize masses before computing partition function
            if np.sum(masses) > 0:
                masses_renorm = masses / np.sum(masses)
                partition_sum = np.sum(masses_renorm ** q)
                partition_sums.append(partition_sum)
            else:
                partition_sums.append(0)

    return partition_sums


def fractal_dimension_mass(field, min_box_size=2, max_box_size=None,
                           num_scales=10, q=2, return_diagnostics=False):
    """
    Measure the mass-based fractal dimension of a 2D or 3D field.

    .. deprecated::
        This function is deprecated. Use `fractal_dimension(field, method='mass')`
        instead, which provides additional features including spatial weighting
        and masking support.

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

    See Also
    --------
    fractal_dimension : Unified function with weighting and masking support.
    """
    import warnings
    warnings.warn(
        "fractal_dimension_mass is deprecated. Use fractal_dimension(field, method='mass') instead.",
        DeprecationWarning,
        stacklevel=2
    )

    return fractal_dimension(
        field,
        min_box_size=min_box_size,
        max_box_size=max_box_size,
        num_scales=num_scales,
        method='mass',
        q=q,
        return_diagnostics=return_diagnostics
    )


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
