import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(x, y, mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).

    Use the formula:

    f_XY(x,y) =
    1 / (2*pi*sigma_x*sigma_y*sqrt(1-rho^2))
    *
    exp( -Q / (2*(1-rho^2)) )
    """
    # Calculate Q(x,y)
    Q = (
        ((x - mu_x) ** 2) / sigma_x**2
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / sigma_y**2
    )
    
    # Calculate the joint Gaussian PDF
    normalizer = 1 / (2 * np.pi * sigma_x * sigma_y * np.sqrt(1 - rho**2))
    exponent = np.exp(-Q / (2 * (1 - rho**2)))
    
    return normalizer * exponent


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma_x)) * np.exp(-((x - mu_x) ** 2) / (2 * sigma_x**2))


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """
    return (1 / (np.sqrt(2 * np.pi) * sigma_y)) * np.exp(-((y - mu_y) ** 2) / (2 * sigma_y**2))


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix:

    [[sigma_x^2, rho*sigma_x*sigma_y],
     [rho*sigma_x*sigma_y, sigma_y^2]]
    """
    return np.array([
        [sigma_x**2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y**2]
    ])


def joint_pdf_grid_integral(mu_x=1, mu_y=-2, sigma_x=2, sigma_y=3, rho=0.6, n=250):
    """
    Numerically approximate integral of joint Gaussian PDF
    over the rectangle:

    [mu_x - 4*sigma_x, mu_x + 4*sigma_x]
    x
    [mu_y - 4*sigma_y, mu_y + 4*sigma_y]

    Use a rectangular grid or trapezoidal numerical integration.
    """
    # Define the integration bounds
    x_min = mu_x - 4 * sigma_x
    x_max = mu_x + 4 * sigma_x
    y_min = mu_y - 4 * sigma_y
    y_max = mu_y + 4 * sigma_y
    
    # Create grid
    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)
    
    dx = (x_max - x_min) / (n - 1)
    dy = (y_max - y_min) / (n - 1)
    
    # Use trapezoidal rule: sum of all grid points with appropriate weights
    integral = 0.0
    for i in range(n):
        for j in range(n):
            x = x_vals[i]
            y = y_vals[j]
            f = joint_gaussian_pdf(x, y, mu_x, mu_y, sigma_x, sigma_y, rho)
            
            # Weight for trapezoidal rule
            weight = 1.0
            if i == 0 or i == n - 1:
                weight *= 0.5
            if j == 0 or j == n - 1:
                weight *= 0.5
            
            integral += weight * f
    
    # Multiply by grid spacing
    integral *= dx * dy
    
    return integral


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate n samples from a jointly Gaussian distribution.

    Return two arrays:
    x_samples, y_samples

    Hint:
    Use np.random.multivariate_normal.
    """
    np.random.seed(seed)
    
    # Mean vector
    mean = [mu_x, mu_y]
    
    # Covariance matrix
    cov = covariance_matrix(sigma_x, sigma_y, rho)
    
    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, n)
    
    x_samples = samples[:, 0]
    y_samples = samples[:, 1]
    
    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """
    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return 2 by 2 sample covariance matrix.

    Use denominator n-1.
    """
    # Stack samples into a 2D array where each row is an observation
    samples = np.column_stack([x_samples, y_samples])
    
    # Use np.cov with ddof=1 for n-1 denominator
    return np.cov(samples, rowvar=False, ddof=1)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """
    # Get sample covariance matrix
    cov_matrix = sample_covariance_matrix(x_samples, y_samples)
    
    # Standard deviations
    std_x = np.sqrt(cov_matrix[0, 0])
    std_y = np.sqrt(cov_matrix[1, 1])
    
    # Correlation coefficient = covariance / (std_x * std_y)
    return cov_matrix[0, 1] / (std_x * std_y)


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:
    return True if rho is zero, otherwise False.
    """
    return bool(rho == 0)


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and check that
    sample covariance is approximately zero.
    Return True or False.
    """
    # Generate samples with rho=0 (independent)
    x, y = generate_joint_gaussian_samples(n=n, rho=0)
    
    # Get sample covariance matrix
    cov_matrix = sample_covariance_matrix(x, y)
    
    # Check if off-diagonal covariance is close to zero
    # For independent variables, covariance should be approximately zero
    return bool(abs(cov_matrix[0, 1]) < 0.2)  # Allow some tolerance


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and check that
    sample covariance is close to rho*sigma_x*sigma_y.
    Return True or False.
    """
    # Generate samples with rho=0.6
    x, y = generate_joint_gaussian_samples(n=n, rho=0.6)
    
    # Get sample covariance matrix
    cov_matrix = sample_covariance_matrix(x, y)
    
    # Expected covariance: rho * sigma_x * sigma_y = 0.6 * 2 * 3 = 3.6
    expected_cov = 0.6 * 2 * 3
    
    # Check if sample covariance is close to expected
    return bool(abs(cov_matrix[0, 1] - expected_cov) < 0.3)  # Allow some tolerance
