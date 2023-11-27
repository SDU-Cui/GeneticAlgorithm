import numpy as np
from pyomo.environ import *
import tomli

# Define row and col number
row = 200
col = 96

tasks = np.genfromtxt('data/vehicle_data_200.csv', delimiter=',', names=True, dtype=None, encoding='ANSI')
phase_base_load = np.genfromtxt('data/phase_base_load.csv', delimiter=',', dtype=None, encoding='UTF-8')

''' power--每辆车的充电功率;
    efficiency--每辆车的充电效率
    capacity--每辆车的电池容量
    ρ--电价
    R--惩罚系数
    n--种群数量
    probability--变异概率
    α--目标值cost + α * imbalance系数'''

arrival_time_step = tasks['ta']
departure_time_step = tasks['td']
power = np.expand_dims(tasks['P'], axis=1)
efficiency = np.expand_dims(tasks['η'], axis=1)
capacity = np.expand_dims(tasks['E'], axis=1)
sa = tasks['sa']
sd = tasks['sd']
Φ = tasks['Φ']
ρ1 = np.full((30), 1)
ρ2 = np.full((col - 30), 0.1)
ρ = np.hstack((ρ1 , ρ2))
α = 0.001

#获得基础负载和电路限制get_restriction_in_power
base_load = np.sum(phase_base_load, axis=0)
restriction_in_power = 2200 - base_load

# 最大三相不平衡度
max_imbalance_limit = 0.04

# 为电池SOC约束做准备，计算每辆车15min的充电增量
unit_increment = power * efficiency * 0.25 / capacity #每辆车15min的单位增量

def Distinguish_phase():
    # Get a array to distinguish phase A,B,C
    phase_list = np.empty((row), dtype = str)
    for vehicle in range(row):
        # Get three phaseA,B,C load
        if Φ[vehicle] == 'A':
            phase_list[vehicle] = 'A'
        elif Φ[vehicle] == 'B':
            phase_list[vehicle] = 'B'
        else:
            phase_list[vehicle] = 'C'
    return phase_list

def Calculate_phase(x_values, phase_list):
    # Get phase A,B,C charging power
    power_list = x_values * power
    power_phaseA = np.array(power_list[np.where(phase_list == 'A')])
    power_phaseB = np.array(power_list[np.where(phase_list == 'B')])
    power_phaseC = np.array(power_list[np.where(phase_list == 'C')])
    # PhaseA charging load
    phaseA = np.sum(power_phaseA, axis = 0)
    # PhaseB charging load
    phaseB = np.sum(power_phaseB, axis = 0)
    # PhaseC charging load
    phaseC = np.sum(power_phaseC, axis = 0)
    # Base load for each of the three items
    three_phase = np.vstack((phaseA, phaseB, phaseC)) + phase_base_load

    return three_phase

def Calculate_imbalance(three_phase):
    avg = np.mean(three_phase, axis=0)
    minus = three_phase - avg

    return np.sum(np.square(minus), axis=0)

# 创建目标函数
def objective_rule(model):
    x_values = np.array([[model.x[j, k] for k in model.cols] for j in model.rows]) #获得model.x的numpy矩阵
    cost = x_values * ρ * power * 0.25
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x_values, phase_list)
    result_phase = Calculate_imbalance(three_phase)

    return np.sum(cost) + α * np.sum(result_phase)

# 添加充电时间约束条件
def zero_constraint_rule(model, i):
    lower_bound = arrival_time_step[i - 1]
    upper_bound = departure_time_step[i - 1]
    for j in model.cols:
        if lower_bound <= j - 1 <= upper_bound:
            return Constraint.Skip  # 不对在区间内的变量添加约束
        else:
            return model.x[i, j] == 0  # 将不在区间内的变量置零

def fix_model_value(model):
    for i in model.rows:
        lower_bound = arrival_time_step[i - 1]
        upper_bound = departure_time_step[i - 1]
        for j in model.cols:
            if lower_bound <= j - 1 <= upper_bound:
                continue
            else:
                model.x[i, j].fix(0)

# 添加电路负荷三相不平衡约束条件
def power_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    power_constraint_rule = (used_power - restriction_in_power[i - 1] <= 0)
    return power_constraint_rule
 
def power_phase_AB_lower_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_A = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'A')
    phase_B = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'B')
    rule_AB_lower = (-unbalance - 3 * (phase_A - phase_B) <= 0)
    return rule_AB_lower

def power_phase_AB_upper_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_A = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'A')
    phase_B = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'B')
    rule_AB_upper = (3 * (phase_A - phase_B) - unbalance <= 0)
    return rule_AB_upper

def power_phase_AC_lower_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_A = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'A')
    phase_C = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'C')
    rule_AC_lower = (-unbalance - 3 * (phase_A - phase_C) <= 0)
    return rule_AC_lower

def power_phase_AC_upper_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_A = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'A')
    phase_C = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'C')
    rule_AC_upper = (3 * (phase_A - phase_C) - unbalance <= 0)
    return rule_AC_upper

def power_phase_BC_lower_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_B = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'B')
    phase_C = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'C')
    rule_BC_lower = (-unbalance - 3 * (phase_B - phase_C) <= 0)
    return rule_BC_lower

def power_phase_BC_upper_constraint_rule(model, i):
    power_col = (model.x[j, i] * power[j - 1] for j in model.rows)
    used_power = sum(power_col)
    total_power = base_load[i - 1] + used_power
    unbalance = total_power * max_imbalance_limit
    phase_B = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'B')
    phase_C = sum(model.x[k + 1, i] * power[k] for k in range(len(tasks)) if Φ[k] == 'C')
    rule_BC_upper = (3 * (phase_B - phase_C) - unbalance <= 0)
    return rule_BC_upper

# 添加电池SOC约束
def energy_constraint_rule(model, i):
    #unit_increment = power * efficiency * 0.25 / capacity #每辆车15min的单位增量
    increment_list = np.multiply(unit_increment, np.array([[model.x[j, k] for k in model.cols] for j in model.rows]))
    max_SOC = np.floor((sd - sa) / unit_increment) * unit_increment #可由开关控制实现的最大 SOC 值
    max_SOC = max_SOC.reshape(-1)
    total_increment = np.sum(increment_list, axis = 1) #计算每辆车所有时间步的增量
    return total_increment[i - 1] >= max_SOC[i - 1]

# 创建一个具体的模型
model = ConcreteModel()

# 决策变量
# 创建一个矩阵变量
model.rows = RangeSet(1, row)  # 行数
model.cols = RangeSet(1, col)  # 列数
model.x = Var(model.rows, model.cols, within=Binary)

# 添加约束条件
# 添加充电时间约束条件
#model.zero_constraint = Constraint(range(1, 101), rule = zero_constraint_rule)
fix_model_value(model)

# 添加电池SOC约束
model.energy_constraint = Constraint(range(1, row + 1), rule = energy_constraint_rule)

# 添加电路负荷三相不平衡约束条件
model.power_constraint = Constraint(range(1, col + 1), rule = power_constraint_rule)
# model.power_AB_lower_constraint = Constraint(range(1, 97), rule = power_phase_AB_lower_constraint_rule)
# model.power_AB_upper_constraint = Constraint(range(1, 97), rule = power_phase_AB_upper_constraint_rule)
# model.power_AC_lower_constraint = Constraint(range(1, 97), rule = power_phase_AC_lower_constraint_rule)
# model.power_AC_upper_constraint = Constraint(range(1, 97), rule = power_phase_AC_upper_constraint_rule)
# model.power_BC_lower_constraint = Constraint(range(1, 97), rule = power_phase_BC_lower_constraint_rule)
# model.power_BC_upper_constraint = Constraint(range(1, 97), rule = power_phase_BC_upper_constraint_rule)

# 创建目标函数
model.objective = Objective(rule=objective_rule, sense=minimize)

# Read param from param.toml
with open("param.toml", "rb") as toml_file:
    config_data = tomli.load(toml_file)
# 求解线性规划问题
solver = SolverFactory('gurobi')
# 从配置文件中读取Gurobi参数
gurobi_params = config_data["Gurobi"]
# 设置Gurobi求解器参数
solver.options.update(gurobi_params)

results = solver.solve(model, tee=True)

# 打印结果
print(results.solver.status)
print('优化结果：', model.objective())

# Save model.x as csv file
with open('./data/optimization5.csv', 'w') as cvs_file:
    model.x.pprint(cvs_file)