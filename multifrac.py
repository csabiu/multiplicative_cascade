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
