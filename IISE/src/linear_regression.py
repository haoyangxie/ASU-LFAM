import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture


def temp(t, b0, b1, T_env, T0):
    T = T_env - (b0/b1) + (T0 - T_env + (b0/b1))*np.exp(b1*(t))
    return T


def regression_for_one_layer(layer_profile_file):
    reg = LinearRegression()
    model = {
        'beta0': [],
        'beta1': [],
        'score': []
    }
    T_env = 24
    df = pd.read_csv(layer_profile_file).to_numpy()
    profile_list = np.transpose(df)

    for profile in profile_list:
        X = [i - T_env for i in profile]
        X = X[:len(profile) - 1]
        Y = [profile[i + 1] - profile[i] for i in range(len(profile) - 1)]
        X = np.array(X).reshape(-1, 1)
        Y = np.array(Y).reshape(-1, 1)
        reg.fit(X, Y)
        model['beta0'].append(reg.intercept_[0])
        model['beta1'].append(reg.coef_[0][0])
        model['score'].append(reg.score(X, Y))

    return model


def plot_for_all_layers(case_path, case_name):
    b0_list = []
    b1_list = []

    for file in os.listdir(case_path):
        if file.endswith('.csv'):
            file_name = os.path.join(case_path, file)
            model = regression_for_one_layer(file_name)
            b0_list.extend(model['beta0'])
            b1_list.extend(model['beta1'])
    plt.scatter(b0_list, b1_list)
    plt.xlabel('intercept beta0')
    plt.ylabel('slope beta1')
    plt.title(f'{case_name}')
    plt.savefig(f'../output/image/cluster/{case_name}.png')
    plt.clf()


def save_slope_intercept(case_path):
    b0_list = []
    b1_list = []

    for file in os.listdir(case_path):
        if file.endswith('.csv'):
            file_name = os.path.join(case_path, file)
            model = regression_for_one_layer(file_name)
            b0_list.extend(model['beta0'])
            b1_list.extend(model['beta1'])
    return b0_list, b1_list


if __name__ == '__main__':
    case_list = ['planters', 'table', 'totems']
    slopes_all_cases = []
    intercept_all_cases = []

    for case in case_list:
        case_path = f'../input/{case}/transormer_input_no_rebound'
        b0_list, b1_list = save_slope_intercept(case_path)
        slopes_all_cases.extend(b1_list)
        intercept_all_cases.extend(b0_list)
        # plot_for_all_layers(case_path, case)
        slopes_all_cases_array = np.array(slopes_all_cases)
        intercept_all_cases_array = np.array(intercept_all_cases)
        np.save('slopes.npy', slopes_all_cases_array)
        np.save('intercept.npy', intercept_all_cases_array)