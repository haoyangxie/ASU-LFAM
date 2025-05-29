import numpy as np
from scipy.stats import multivariate_normal


mean_vector = np.array([-0.02509307, 1.47657237])
covariance_matrix = np.array([[5.27384370e-05, -3.10535987e-03], [-3.10535987e-03,  2.15270209e-01]])
distribution = multivariate_normal(mean=mean_vector, cov=covariance_matrix)


def generate_profiles(num_samples=100):
    random_samples = distribution.rvs(size=num_samples)
    generated_slopes = random_samples[:, 0]
    generated_intercepts = random_samples[:, 1]
    profiles_list = []
    for slope, intercept in zip(generated_slopes, generated_intercepts):
        profile = []
        for t in range(120):
            T = 24 - (intercept / slope) + (200 - 24 + (intercept / slope)) * np.exp(slope * (t))
            profile.append(T)
        profiles_list.append(profile)
    my_dict = {}
    for i in range(100):
        my_dict[i] = profiles_list[i]
    return my_dict
