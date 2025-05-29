import gurobipy as gp
from gurobipy import GRB
import pickle
import matplotlib.pyplot as plt

T_u, T_b, T_l = 150, 120, 90
initial_speed = 20
a = 50
hashmap = {0: (66, 0), 1: (132, 2), 2: (264, 6), 3: (132, 8), 4: (264, 12), 5: (132, 14), 6: (198, 17), 7: (66, 18), 8: (264, 22),
           9: (198, 25), 10: (66, 26), 11: (330, 31), 12: (66, 32), 13: (132, 34), 14: (66, 35), 15: (66, 36), 16: (132, 38), 17: (66, 39),
           18: (66, 40), 19: (132, 42), 20: (132, 44), 21: (264, 48), 22: (66, 49), 23: (132, 51), 24: (132, 53)
           }

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

mdl = gp.Model('speed_control')
v = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=100, name=f'v{i}') for i in range(25)]
t = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=500, name=f't{j}') for j in range(25)]
v_inverse = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.01, ub=0.2, name=f'y{i}') for i in range(25)]
mdl.update()
for i in range(25):
    mdl.addConstr(v_inverse[i] * v[i] == 1, name=f'inv_constr_{i}')

for i in range(25):
    if i == 0:
        mdl.addConstr(v[i] >= 0.95 * initial_speed, name=f'constr_v{i}_lower')
        mdl.addConstr(v[i] <= 1.05 * initial_speed, name=f'constr_v{i}_upper')
    else:
        mdl.addConstr(v[i] >= 0.95 * v[i - 1], name=f'constr_v{i}_lower')
        mdl.addConstr(v[i] <= 1.05 * v[i - 1], name=f'constr_v{i}_upper')

for j in range(25):
    sum_term_this_layer = gp.quicksum(hashmap[j][0] * v_inverse[i] for i in range(j + 1))
    sum_term_previous_layer = gp.quicksum(hashmap[j][0] / 20 for i in range(j+1, 25))
    mdl.addConstr(t[j] == sum_term_this_layer + sum_term_previous_layer)

exp_var = [mdl.addVar(name=f'exp_term{j}') for j in range(25)]
mul_var = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=-10, ub=0, name=f'mul_var{j}') for j in range(25)]
temperature_var = [mdl.addVar(name=f'temperature{j}') for j in range(25)]
for j in range(25):
    mdl.addConstr(mul_var[j] == b1_list[hashmap[j][1]] * t[j], name=f'mul_constr{j}')
    mdl.addGenConstrExp(mul_var[j], exp_var[j], name=f'exp_constr{j}')
    mdl.addConstr(
        temperature_var[j] == 24 - (b0_list[hashmap[j][1]] / b1_list[hashmap[j][1]]) + (T0_list[hashmap[j][1]] - 24 + (b0_list[hashmap[j][1]] / b1_list[hashmap[j][1]])) * exp_var[
            j],
        name=f'temperature_constr{j}')

    mdl.addConstr(temperature_var[j] >= T_l, name=f'temp_lower_{j}')
    mdl.addConstr(temperature_var[j] <= T_u, name=f'temp_upper_{j}')

objective = gp.quicksum((temperature_var[j] - T_b) ** 2 for j in range(25))
mdl.setObjective(objective, GRB.MINIMIZE)
mdl.optimize()


temp_diff = []
for i in range(25):
    print(v[i].X)
    temp_diff.append(temperature_var[i].X - T_b)
plt.plot(temp_diff)
plt.show()

