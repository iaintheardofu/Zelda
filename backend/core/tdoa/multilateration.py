"""
Multilateration algorithms for TDOA geolocation.

Given TDOA measurements from multiple receivers at known positions,
these algorithms estimate the source position.
"""

import numpy as np
from scipy.optimize import least_squares, minimize
from dataclasses import dataclass
from typing import List, Tuple, Optional
from loguru import logger

# Optional: Genetic algorithm optimization
try:
    from deap import base, creator, tools, algorithms
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    logger.warning("DEAP not available - genetic algorithm optimization disabled")


# Speed of light (m/s)
SPEED_OF_LIGHT = 299792458.0


@dataclass
class TDOAMeasurement:
    """A single TDOA measurement between two receivers"""

    # Receiver positions (latitude, longitude, altitude in meters)
    receiver1_pos: Tuple[float, float, float]
    receiver2_pos: Tuple[float, float, float]

    # Time difference of arrival (seconds)
    tdoa: float

    # Confidence/weight (0-1)
    confidence: float = 1.0

    @property
    def range_difference(self) -> float:
        """Convert TDOA to range difference in meters"""
        return self.tdoa * SPEED_OF_LIGHT


def multilaterate_taylor_series(
    measurements: List[TDOAMeasurement],
    initial_guess: Optional[Tuple[float, float, float]] = None,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Multilateration using Taylor Series Least Squares method.

    This is a fast, iterative method that linearizes the nonlinear TDOA
    equations around an initial guess and iteratively refines the position.

    Args:
        measurements: List of TDOA measurements
        initial_guess: Initial position guess (lat, lon, alt). If None, uses centroid.
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Tuple of ((lat, lon, alt), residual_error)

    References:
        Fang, B. T. (1990). Simple solutions for hyperbolic and related
        position fixes. IEEE Trans. Aerospace and Electronic Systems, 26(5).
    """

    if len(measurements) < 3:
        raise ValueError("Need at least 3 TDOA measurements")

    # Convert to Cartesian coordinates (ECEF - Earth-Centered Earth-Fixed)
    # For simplicity, we'll use a local tangent plane approximation
    # In production, use proper geodetic transformations (pyproj)

    receiver_positions = []
    tdoa_values = []
    weights = []

    for meas in measurements:
        # Use receiver1 as reference, calculate relative positions
        pos1 = np.array(meas.receiver1_pos)
        pos2 = np.array(meas.receiver2_pos)

        receiver_positions.append((pos1, pos2))
        tdoa_values.append(meas.tdoa)
        weights.append(meas.confidence)

    # Initial guess: centroid of receivers if not provided
    if initial_guess is None:
        all_receivers = [pos for pair in receiver_positions for pos in pair]
        initial_guess = tuple(np.mean(all_receivers, axis=0))

    # Convert initial guess to array
    x = np.array(initial_guess)

    # Taylor series iteration
    for iteration in range(max_iterations):
        # Build Jacobian matrix and residual vector
        jacobian = []
        residuals = []

        for (pos1, pos2), tdoa, weight in zip(receiver_positions, tdoa_values, weights):
            # Predicted range difference
            r1 = np.linalg.norm(x - pos1)
            r2 = np.linalg.norm(x - pos2)
            predicted_range_diff = r2 - r1

            # Measured range difference
            measured_range_diff = tdoa * SPEED_OF_LIGHT

            # Residual (error)
            residual = measured_range_diff - predicted_range_diff
            residuals.append(weight * residual)

            # Partial derivatives for Jacobian
            if r1 > 1e-6 and r2 > 1e-6:
                dr1_dx = -(x - pos1) / r1
                dr2_dx = (x - pos2) / r2
                jacobian_row = weight * (dr2_dx - dr1_dx)
            else:
                jacobian_row = np.zeros(3)

            jacobian.append(jacobian_row)

        J = np.array(jacobian)
        r = np.array(residuals)

        # Solve normal equations: J^T * J * delta = J^T * r
        try:
            JTJ = J.T @ J
            JTr = J.T @ r
            delta = np.linalg.solve(JTJ, JTr)
        except np.linalg.LinAlgError:
            logger.warning("Singular matrix in Taylor series, using pseudoinverse")
            delta = np.linalg.lstsq(J, r, rcond=None)[0]

        # Update position
        x = x + delta

        # Check convergence
        if np.linalg.norm(delta) < tolerance:
            logger.debug(f"Taylor series converged in {iteration + 1} iterations")
            break

    # Calculate final residual error
    residual_error = np.sqrt(np.mean(r**2))

    return tuple(x), residual_error


def multilaterate_least_squares(
    measurements: List[TDOAMeasurement],
    initial_guess: Optional[Tuple[float, float, float]] = None,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Multilateration using nonlinear least squares optimization.

    Uses scipy's least_squares optimizer which is robust to outliers
    and handles bounds well.

    Args:
        measurements: List of TDOA measurements
        initial_guess: Initial position guess (lat, lon, alt)

    Returns:
        Tuple of ((lat, lon, alt), residual_error)
    """

    if len(measurements) < 3:
        raise ValueError("Need at least 3 TDOA measurements")

    # Extract data
    receiver_positions = []
    tdoa_values = []
    weights = []

    for meas in measurements:
        pos1 = np.array(meas.receiver1_pos)
        pos2 = np.array(meas.receiver2_pos)
        receiver_positions.append((pos1, pos2))
        tdoa_values.append(meas.tdoa)
        weights.append(meas.confidence)

    # Initial guess
    if initial_guess is None:
        all_receivers = [pos for pair in receiver_positions for pos in pair]
        initial_guess = tuple(np.mean(all_receivers, axis=0))

    # Residual function
    def residuals(x):
        res = []
        for (pos1, pos2), tdoa, weight in zip(receiver_positions, tdoa_values, weights):
            r1 = np.linalg.norm(x - pos1)
            r2 = np.linalg.norm(x - pos2)
            predicted_range_diff = r2 - r1
            measured_range_diff = tdoa * SPEED_OF_LIGHT
            res.append(weight * (measured_range_diff - predicted_range_diff))
        return np.array(res)

    # Optimize
    result = least_squares(residuals, initial_guess, method='lm')

    if not result.success:
        logger.warning(f"Least squares optimization did not converge: {result.message}")

    position = tuple(result.x)
    residual_error = np.sqrt(np.mean(result.fun**2))

    return position, residual_error


def multilaterate_genetic(
    measurements: List[TDOAMeasurement],
    bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
    population_size: int = 100,
    generations: int = 50,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Multilateration using Genetic Algorithm optimization.

    Genetic algorithms are good for:
    - Global optimization (avoids local minima)
    - Difficult geometries (e.g., receivers on same side of target)
    - Cases where gradient-based methods fail

    Args:
        measurements: List of TDOA measurements
        bounds: Search bounds ((lat_min, lat_max), (lon_min, lon_max), (alt_min, alt_max))
        population_size: GA population size
        generations: Number of generations to evolve

    Returns:
        Tuple of ((lat, lon, alt), residual_error)
    """

    if not DEAP_AVAILABLE:
        raise RuntimeError("DEAP library required for genetic algorithm. Install: pip install deap")

    if len(measurements) < 3:
        raise ValueError("Need at least 3 TDOA measurements")

    # Extract data
    receiver_positions = []
    tdoa_values = []
    weights = []

    for meas in measurements:
        pos1 = np.array(meas.receiver1_pos)
        pos2 = np.array(meas.receiver2_pos)
        receiver_positions.append((pos1, pos2))
        tdoa_values.append(meas.tdoa)
        weights.append(meas.confidence)

    # Determine bounds if not provided
    if bounds is None:
        all_receivers = np.array([pos for pair in receiver_positions for pos in pair])
        min_pos = np.min(all_receivers, axis=0)
        max_pos = np.max(all_receivers, axis=0)

        # Expand bounds by 50%
        center = (min_pos + max_pos) / 2
        span = max_pos - min_pos
        min_pos = center - span
        max_pos = center + span

        bounds = tuple((float(min_pos[i]), float(max_pos[i])) for i in range(3))

    # Fitness function (minimize TDOA residual error)
    def evaluate(individual):
        x = np.array(individual)
        total_error = 0.0

        for (pos1, pos2), tdoa, weight in zip(receiver_positions, tdoa_values, weights):
            r1 = np.linalg.norm(x - pos1)
            r2 = np.linalg.norm(x - pos2)
            predicted_range_diff = r2 - r1
            measured_range_diff = tdoa * SPEED_OF_LIGHT
            error = (measured_range_diff - predicted_range_diff) ** 2
            total_error += weight * error

        return (total_error,)  # DEAP requires tuple return

    # Setup DEAP
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Attribute generators
    for i in range(3):
        toolbox.register(f"attr_{i}", np.random.uniform, bounds[i][0], bounds[i][1])

    # Individual and population
    toolbox.register(
        "individual",
        tools.initCycle,
        creator.Individual,
        (getattr(toolbox, f"attr_{i}") for i in range(3)),
        n=1
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operators
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create population and evolve
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    # Run evolution
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,  # Crossover probability
        mutpb=0.2,  # Mutation probability
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=False
    )

    # Best individual
    best = hof[0]
    position = tuple(best)
    residual_error = np.sqrt(best.fitness.values[0] / len(measurements))

    logger.debug(f"GA converged with residual error: {residual_error:.2f}m")

    return position, residual_error


def compute_gdop(
    receiver_positions: List[Tuple[float, float, float]],
    target_position: Tuple[float, float, float]
) -> float:
    """
    Compute Geometric Dilution of Precision (GDOP) for a given geometry.

    GDOP indicates how receiver geometry affects position accuracy.
    Lower GDOP is better (< 4 is good, > 10 is poor).

    Args:
        receiver_positions: List of receiver positions
        target_position: Target position

    Returns:
        GDOP value
    """

    target = np.array(target_position)
    n_receivers = len(receiver_positions)

    if n_receivers < 4:
        return float('inf')

    # Build geometry matrix
    G = []
    for pos in receiver_positions:
        receiver = np.array(pos)
        diff = target - receiver
        range_val = np.linalg.norm(diff)

        if range_val > 1e-6:
            # Unit vector from target to receiver
            unit_vector = diff / range_val
            # Add time component (for 4D: x, y, z, t)
            row = np.append(unit_vector, 1.0)
            G.append(row)

    G = np.array(G)

    # GDOP = sqrt(trace((G^T * G)^-1))
    try:
        GTG_inv = np.linalg.inv(G.T @ G)
        gdop = np.sqrt(np.trace(GTG_inv))
    except np.linalg.LinAlgError:
        gdop = float('inf')

    return gdop
