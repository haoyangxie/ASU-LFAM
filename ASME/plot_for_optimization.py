import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import Bounds
from scipy.optimize import minimize


def exponential_model(x, a, b, c):
    """curve fit model"""
    return a*np.exp(-b*x) + c


def prediction_regression(data_list):
    """regression for the transformer's output, to avoid overflow, divide input by 100"""
    x = range(len(data_list))
    params, covariance = curve_fit(exponential_model, x, data_list/100.0)
    return params


def get_all_profile(num_points, data,):



def objective(t, T_b, num_points, data, w_0=700, w_1=1):
    func = 0
    for i in range(num_points):
        a, b, c = prediction_regression(data[i])
        func += w_1 * (exponential_model(t, a, b, c)*100 - T_b) ** 2
    func += w_0 * t
    return func


def find_lb_ub(num_points, data, T_l=80, T_u=140):
    t_l_list = []
    t_u_list = []
    for i in range(num_points):
        a, b, c = prediction_regression(data[i])
        t_list = []
        for t in range(len(data[i])):
            if T_l <= exponential_model(t, a, b, c) * 100 <= T_u:
                t_list.append(t)
        t_l_list.append(min(t_list))
        t_u_list.append(max(t_list))
        return max(t_l_list), min(t_u_list)


def run_optimization(data_file):
    data = np.loadtxt(data_file)
    data = data[2:]
    num_points = len(data)
    lb, ub = find_lb_ub(num_points, data)
    bounds = Bounds(lb, ub)
    T_b = 100
    x0 = np.array([np.random.uniform(lb, ub)])
    res = minimize(objective, x0, method='SLSQP', args=(T_b, num_points, data), options={'ftol': 1e-9, 'disp': True},
                   bounds=bounds)
    return res.x

