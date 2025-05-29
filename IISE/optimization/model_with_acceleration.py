import gurobipy as gp
from gurobipy import GRB
import pickle

a = 50
M = 1000
T_u, T_b, T_l = 150, 120, 90
initial_speed = 20
with open('../pkl_file/true_data_params.pkl', 'rb') as f:
    params = pickle.load(f)
b0_list = []
b1_list = []
T0_list = []
for param in params:
    b0, b1, T0 = param
    b0_list.append(b0)
    b1_list.append(b1)
    T0_list.append(T0)

mdl = gp.Model('speed_control_without_acceleration')
# speed, we calculate acceleration based on speed
v = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=100, name=f'v{i}') for i in range(54)]
# square of speed
v_square = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=25, ub=10000, name=f'v_square{i}') for i in range(54)]
# 1 / speed
v_inverse = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.01, ub=0.2, name=f'y{i}') for i in range(54)]
# acceleration distance for each partition
d = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=66, name=f'distance{i}') for i in range(54)]
v_inside_partition = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=100, name=f'v_inside_partition{partition}{i}') for i in range(660)] for partition in range(54)]
v_inside_partition_square = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=100, name=f'v_inside_partition_square{partition}{i}') for i in range(660)] for partition in range(54)]
t = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=500, name=f't_{partition}_{i}')
      for i in range(660)] for partition in range(54)]
inside_acceleration = [[mdl.addVar(vtype=GRB.BINARY, name=f'inside_acceleration_{partition}_{i}')
                        for i in range(660)] for partition in range(54)]
v_inside_average = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=100, name=f'v_inside_avg{i}') for i in range(660)] for partition in range(54)]
# sign to calculate time
delta_v_positive = [mdl.addVar(vtype=GRB.BINARY, name=f'delta_v_positive{i}') for i in range(54)]
mdl.update()
for i in range(54):
    mdl.addConstr(v_inverse[i] * v[i] == 1, name=f'inv_constr_{i}')
    mdl.addConstr(v_square[i] == v[i]*v[i])

for i in range(54):
    if i == 0:
        mdl.addConstr(v[i] >= 0.95 * initial_speed, name=f'constr_v{i}_lower')
        mdl.addConstr(v[i] <= 1.05 * initial_speed, name=f'constr_v{i}_upper')
    else:
        mdl.addConstr(v[i] >= 0.95 * v[i-1], name=f'constr_v{i}_lower')
        mdl.addConstr(v[i] <= 1.05 * v[i-1], name=f'constr_v{i}_upper')

# get acceleration distance for each partition
for i in range(54):
    if i == 0:
        mdl.addConstr(v[i] - initial_speed <= M * delta_v_positive[i])
        mdl.addConstr(v[i] - initial_speed >= (1 - delta_v_positive[i]) * (-M))

        mdl.addConstr(
            d[i] == delta_v_positive[i] * (v_square[i] - initial_speed ** 2) / (2 * a) + (1 - delta_v_positive[i]) * (
                        initial_speed ** 2 - v_square[i]) / (2 * a))
    else:
        mdl.addConstr(v[i] - v[i - 1] <= M * delta_v_positive[i])
        mdl.addConstr(v[i] - v[i - 1] >= (1 - delta_v_positive[i]) * (-M))

        mdl.addConstr(
            d[i] == delta_v_positive[i] * (v_square[i] - v_square[i - 1]) / (2 * a) + (1 - delta_v_positive[i]) * (
                        v_square[i - 1] - v_square[i]) / (2 * a))

for partition in range(54):
    for i in range(660):
        mdl.addConstr(0.1 * (i+1) - d[partition] <= M * (1 - inside_acceleration[partition][i]))
        mdl.addConstr(d[partition] - 0.1 * (i+1) <= M * inside_acceleration[partition][i])

for partition in range(54):
    for i in range(660):
        v_inside_partition_square[partition][i] = v_inside_partition[partition][i] * v_inside_partition[partition][i]

mdl.addConstr(v_inside_partition[0][0] == initial_speed + 2*a*0.1*inside_acceleration[0][0])
for partition in range(1, 54):
    mdl.addConstr(v_inside_partition[partition][0] == v[partition-1] + 2*a*0.1*inside_acceleration[partition][0])

for partition in range(54):
    for i in range(1, 660):
        mdl.addConstr(v_inside_partition_square[partition][i] == v_inside_partition_square[partition][i-1] + 2*a*0.1*inside_acceleration[partition][i])

mdl.addConstr(v_inside_average[0][0] * (initial_speed + v_inside_partition[0][0]) == 2)
for partition in range(1, 54):
    mdl.addConstr(v_inside_average[partition][0] *(v[partition-1] + v_inside_partition[partition][0]) == 2)
for partition in range(54):
    for i in range(1, 660):
        mdl.addConstr(v_inside_average[partition][i] * (v_inside_partition[partition][i] + v_inside_partition[partition][i-1]) == 2)

for partition in range(54):
    for i in range(660):
        mdl.addConstr(t[partition][i] == 0.1 * v_inside_average[partition][i])

prefix_sum = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f'prefix_sum_{partition}_{i}')
               for i in range(660)] for partition in range(54)]

for partition in range(54):
    for i in range(660):
        if partition == 0 and i == 0:
            mdl.addConstr(prefix_sum[partition][i] == t[partition][i])
        elif partition == 0:
            mdl.addConstr(prefix_sum[partition][i] == prefix_sum[partition][i-1] + t[partition][i])
        elif i == 0:
            mdl.addConstr(prefix_sum[partition][i] == prefix_sum[partition-1][660-1] + t[partition][i])
        else:
            mdl.addConstr(prefix_sum[partition][i] == prefix_sum[partition][i-1] + t[partition][i])

t_profile = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=GRB.INFINITY, name=f't_profile_{partition}_{i}')
      for i in range(660)] for partition in range(54)]

for partition in range(54):
    for i in range(660):
        sum_term = prefix_sum[partition][i]
        mdl.addConstr(t_profile[partition][i] == (53 - partition) * (66/20) + sum_term)

exp_var = [[mdl.addVar(name=f'exp_term{partition}_{i}') for i in range(660)] for partition in range(54)]
mul_var = [[mdl.addVar(vtype=GRB.CONTINUOUS, lb=-10, ub=0, name=f'mul_var{partition}_{i}') for i in range(660)] for partition in range(54)]
temperature_var = [[mdl.addVar(name=f'temperature{partition}{i}') for i in range(660)] for partition in range(54)]


for partition in range(54):
    for i in range(660):
        mdl.addConstr(mul_var[partition][i] == b1_list[partition] * t_profile[partition][i])
        mdl.addGenConstrExp(mul_var[partition][i], exp_var[partition][i])
        mdl.addConstr(temperature_var[partition][i] == 24 - (b0_list[partition] / b1_list[partition]) + (T0_list[partition] - 24 + (b0_list[partition] / b1_list[partition])) * exp_var[partition][i])
        mdl.addConstr(temperature_var[partition][i] >= T_l)
        mdl.addConstr(temperature_var[partition][i] <= T_u)

objective = gp.quicksum((temperature_var[partition][i] - T_b) * (temperature_var[partition][i] - T_b)
                        for partition in range(54)
                        for i in range(660))

mdl.setObjective(objective, GRB.MINIMIZE)
mdl.setParam('SolutionLimit', 1)
mdl.optimize()
mdl.write("feasible_solution.sol")
