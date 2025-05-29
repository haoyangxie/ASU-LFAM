import gurobipy as gp
from gurobipy import GRB
import pickle

T_u, T_b, T_l = 150, 120, 90
initial_speed = 20
a = 50

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

import gurobipy as gp
from gurobipy import GRB

mdl = gp.Model('speed_control')

# 定义变量
v = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=5, ub=100, name=f'v{i}') for i in range(54)]
t = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=500, name=f't{j}') for j in range(54)]
v_inverse = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=0.01, ub=0.2, name=f'y{i}') for i in range(54)]

z = [mdl.addVar(vtype=GRB.INTEGER, lb=95, ub=105, name=f'z{i}') for i in range(54)]

mdl.update()

# 速度倒数约束
for i in range(54):
    mdl.addConstr(v_inverse[i] * v[i] == 1, name=f'inv_constr_{i}')

# 定义 v[i] 和 z[i] 的关系
for i in range(54):
    if i == 0:
        mdl.addConstr(v[i] >= (z[i] / 100) * initial_speed, name=f'constr_v{i}_lower')
        mdl.addConstr(v[i] <= (z[i] / 100) * initial_speed, name=f'constr_v{i}_upper')
    else:
        mdl.addConstr(v[i] == v[i - 1] * (z[i] / 100), name=f'v_z_relation_{i}')

# 时间约束
for j in range(54):
    sum_term = gp.quicksum(66 * v_inverse[i] for i in range(j + 1))
    mdl.addConstr(t[j] == (53 - j) * (66 / 20) + sum_term)

# 温度相关约束
exp_var = [mdl.addVar(name=f'exp_term{j}') for j in range(54)]
mul_var = [mdl.addVar(vtype=GRB.CONTINUOUS, lb=-10, ub=0, name=f'mul_var{j}') for j in range(54)]
temperature_var = [mdl.addVar(name=f'temperature{j}') for j in range(54)]

for j in range(54):
    mdl.addConstr(mul_var[j] == b1_list[j] * t[j], name=f'mul_constr{j}')
    mdl.addGenConstrExp(mul_var[j], exp_var[j], name=f'exp_constr{j}')
    mdl.addConstr(
        temperature_var[j] == 24 - (b0_list[j] / b1_list[j]) + (T0_list[j] - 24 + (b0_list[j] / b1_list[j])) * exp_var[
            j],
        name=f'temperature_constr{j}')

    mdl.addConstr(temperature_var[j] >= T_l, name=f'temp_lower_{j}')
    mdl.addConstr(temperature_var[j] <= T_u, name=f'temp_upper_{j}')

# 定义目标函数
objective = gp.quicksum((temperature_var[j] - T_b) ** 2 for j in range(54))
mdl.setObjective(objective, GRB.MINIMIZE)

# 求解模型
mdl.setParam('TimeLimit', 300)
mdl.optimize()

if mdl.status == GRB.OPTIMAL:
    print("Optimal solution found.")
elif mdl.status == GRB.TIME_LIMIT:
    print("Time limit reached. Current solution is:", mdl.objVal)
elif mdl.status == GRB.INTERRUPTED:
    print("Interrupted. Current solution is:", mdl.objVal)
elif mdl.status == GRB.INFEASIBLE:
    print("Model is infeasible.")
else:
    print("Other status:", mdl.status)

# 保存模型
mdl.write("mmodel.lp")  # 保存为 LP 格式
mdl.write("mmodel.mps") # 保存为 MPS 格式

# 如果有当前解并且是可行的，可以保存解
if mdl.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
    mdl.write("solution.sol")  # 保存当前解