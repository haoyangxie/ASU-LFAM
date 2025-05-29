import sympy as sp

# unit is mm/s
MAX_SPEED = 100
MIN_SPEED = 5
# unit is mm/s^2
MAX_ACCELERATION = 136
MIN_ACCELERATION = -136
TEMPERATURE_UPPER_BOUND = 150
IDEAL_TEMPERATURE = 120
TEMPERATURE_LOWER_BOUND = 90


def get_normalized_speed(speed):
    normalized_speed = (speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
    return normalized_speed


def get_original_speed(normalized_speed):
    original_speed = normalized_speed * (MAX_SPEED - MIN_SPEED) + MIN_SPEED
    return original_speed


def get_normalized_acceleration(acceleration):
    normalized_acceleration = -1 + (acceleration - MIN_ACCELERATION) * (1 - (-1)) / (MAX_ACCELERATION - MIN_ACCELERATION)
    return normalized_acceleration


def get_original_acceleration(normalized_acceleration):
    original_acceleration = MIN_ACCELERATION + (normalized_acceleration - (-1)) * (MAX_ACCELERATION - MIN_ACCELERATION) / (1 - (-1))
    return original_acceleration


def solve_for_t(u, a, s):
    '''
    Solve for t in the equation s = ut + 0.5 * a * t^2
    '''
    t = sp.Symbol('t')
    equation = sp.Eq(0.5 * a * t ** 2 + u * t - s, 0)
    solutions = sp.solve(equation, t)
    positive_solutions = [sol for sol in solutions if sol.is_real and sol.is_nonnegative]

    # If we found a valid solution, return the smallest positive solution
    if positive_solutions:
        return min(positive_solutions)
    # Case 2: If no valid solution, decelerate to MIN_SPEED and continue
    t_decelerate = (u - MIN_SPEED) / a
    s_decelerate = 0.5 * a * t_decelerate ** 2 + u * t_decelerate

    # Check if distance covered during deceleration exceeds the target distance
    if s_decelerate >= s:
        equation_decelerate = sp.Eq(0.5 * a * t ** 2 + u * t - s, 0)
        solutions_decelerate = sp.solve(equation_decelerate, t)
        positive_solutions_decelerate = [sol for sol in solutions_decelerate if sol.is_real and sol.is_nonnegative]
        return min(positive_solutions_decelerate) if positive_solutions_decelerate else None

    # If not, calculate the remaining distance and time at constant speed
    remaining_distance = s - s_decelerate
    t_constant_speed = remaining_distance / MIN_SPEED
    total_time = t_decelerate + t_constant_speed

    return total_time
