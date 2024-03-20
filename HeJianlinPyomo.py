from pyomo.environ import *
import pandas as pd
import matplotlib.pyplot as plt

# 初始化模型
model = ConcreteModel()
# 定义集合
model.Vehicles = RangeSet(0, 399)  # 400辆车的集合
model.TimePeriods = RangeSet(0, 95)  # 96个时间段的集合
model.MaxLoad = Param(initialize=4400, doc='三相最大负载')
model.MaxImbalance = Param(initialize=0.04, doc='最大不平衡度')
# 定义电价参数
def electricity_price_init(model, t):
    if 0 <= t <= 29:
        return 1
    elif 30 <= t <= 95:
        return 0.1
    else:
        # 对于超出0-95范围的时间段，这里返回0只是为了完整性，实际上不会用到
        return 0

model.ElectricityPrice = Param(model.TimePeriods, initialize=electricity_price_init, doc='电价')
# 定义决策变量
model.ChargingStatus = Var(model.Vehicles, model.TimePeriods, within=Binary, doc='充电状态')

# 读取CSV文件
vehicle_data_path = './data/vehicle_data_400.csv'
vehicle_data_df = pd.read_csv(vehicle_data_path, encoding='ANSI')

# 将到达时间和离开时间转换为字典
ta = vehicle_data_df.set_index('v')['ta'].to_dict()
td = vehicle_data_df.set_index('v')['td'].to_dict()
sa = vehicle_data_df.set_index('v')['sa'].to_dict()
sd = vehicle_data_df.set_index('v')['sd'].to_dict()
E = vehicle_data_df.set_index('v')['E'].to_dict()  # 每辆车的电池容量
P = vehicle_data_df.set_index('v')['P'].to_dict()  # 每辆车的充电功率
eta = vehicle_data_df.set_index('v')['η'].to_dict()  # 每辆车的充电效率
Delta = vehicle_data_df.set_index('v')['Δ'].to_dict()  # 每辆车单位时间内能够充的电量
k = vehicle_data_df.set_index('v')['k'].to_dict()  # 每辆车能够充电的总时间段数
phi = vehicle_data_df.set_index('v')['Φ'].to_dict()  # 每辆车充电使用的一相

phase_base_load_path = './data/phase_base_load.csv'

# 使用pandas读取CSV文件
phase_base_load_df = pd.read_csv(phase_base_load_path, header=None)

# 将数据转换为字典，其中键为相（A, B, C），值为96个时间段的基础负荷列表
phase_base_load = {
    'A': phase_base_load_df.iloc[0].tolist(),
    'B': phase_base_load_df.iloc[1].tolist(),
    'C': phase_base_load_df.iloc[2].tolist()
}


# 定义充电时间约束
def charging_time_rule(model, v, t):
    if t < ta[v] or t > td[v]:
        return model.ChargingStatus[v, t] == 0
    else:
        return Constraint.Skip

model.ChargingTimeConstraint = Constraint(model.Vehicles, model.TimePeriods, rule=charging_time_rule)

# 定义SOC约束
def SOC_constraint_rule(model, v):
    return sum(model.ChargingStatus[v, t] for t in model.TimePeriods if ta[v] <= t <= td[v]) >= k[v]

# 将SOC约束添加到模型中
model.SOCConstraint = Constraint(model.Vehicles, rule=SOC_constraint_rule)

# 定义基础负载约束函数
def base_load_constraint_rule(model, t):
    # 计算所有车辆在时间t的总充电功率
    total_charging_power = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles)
    # 计算三相基础负荷之和
    total_base_load = phase_base_load['A'][t] + phase_base_load['B'][t] + phase_base_load['C'][t]
    # 约束：基础负荷 + 充电负荷 <= 最大负载
    return total_charging_power + total_base_load <= model.MaxLoad

# 将基础负载约束添加到模型
model.BaseLoadConstraint = Constraint(model.TimePeriods, rule=base_load_constraint_rule)

# 定义三相不平衡度约束
model.ImbalanceConstraint = ConstraintList()

# 在约束规则中将约束表达式添加到具体的约束对象中
# 定义六个不平衡度约束的规则函数
def imbalance_AB_rule(model, t):
    load_A = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][t]
    load_B = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') + phase_base_load['B'][t]
    avg_load = (load_A + load_B + sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') + phase_base_load['C'][t]) / 3.0
    return (load_A - load_B) <= model.MaxImbalance*avg_load

def imbalance_BA_rule(model, t):
    load_A = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][t]
    load_B = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') + phase_base_load['B'][t]
    avg_load = (load_A + load_B + sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') +
                phase_base_load['C'][t]) / 3.0
    return (load_B - load_A) <= model.MaxImbalance*avg_load

def imbalance_AC_rule(model, t):
    load_A = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][t]
    load_C = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') + phase_base_load['C'][t]
    avg_load = (load_A + sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') + phase_base_load['B'][t] + load_C) / 3.0
    return (load_A - load_C) <= model.MaxImbalance*avg_load

def imbalance_CA_rule(model, t):
    load_A = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][t]
    load_C = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') + phase_base_load['C'][t]
    avg_load = (load_A + sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') +
                phase_base_load['B'][t] + load_C) / 3.0
    return (load_C - load_A) <= model.MaxImbalance*avg_load

def imbalance_BC_rule(model, t):
    load_B = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') + phase_base_load['B'][t]
    load_C = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') + phase_base_load['C'][t]
    avg_load = (sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][t] + load_B + load_C) / 3.0
    return (load_B - load_C) <= model.MaxImbalance*avg_load

def imbalance_CB_rule(model, t):
    load_B = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'B') + phase_base_load['B'][t]
    load_C = sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'C') + phase_base_load['C'][t]
    avg_load = (sum(model.ChargingStatus[v, t] * P[v] for v in model.Vehicles if phi[v] == 'A') + phase_base_load['A'][
        t] + load_B + load_C) / 3.0
    return (load_C - load_B)  <= model.MaxImbalance*avg_load

model.ImbalanceAB = Constraint(model.TimePeriods, rule=imbalance_AB_rule)
model.ImbalanceBA = Constraint(model.TimePeriods, rule=imbalance_BA_rule)
model.ImbalanceAC = Constraint(model.TimePeriods, rule=imbalance_AC_rule)
model.ImbalanceCA = Constraint(model.TimePeriods, rule=imbalance_CA_rule)
model.ImbalanceBC = Constraint(model.TimePeriods, rule=imbalance_BC_rule)
model.ImbalanceCB = Constraint(model.TimePeriods, rule=imbalance_CB_rule)


# 在模型中应用约束规则

# 初始化三相不平衡度约束的容器


def total_charging_cost(model):
    return sum(model.ElectricityPrice[t] * model.ChargingStatus[v, t] * P[v]*0.25 for v in model.Vehicles for t in model.TimePeriods)

model.TotalCost = Objective(rule=total_charging_cost, sense=minimize, doc='Minimize total charging cost')

# 创建一个求解器实例
solver = SolverFactory('gurobi')

# 求解模型
solution = solver.solve(model)

# 输出求解结果
# model.display()
# 输出最终的目标函数值
print("Total Charging Cost:", model.TotalCost())

# 初始化一个空的DataFrame，索引为时间段，列为车辆编号
df = pd.DataFrame(index=range(0, 96), columns=range(0, 400))

# 填充DataFrame
for v in model.Vehicles:
    for t in model.TimePeriods:
        df.at[t, v] = model.ChargingStatus[v, t].value

# 保存DataFrame到CSV文件
df.to_csv('./data/charging_status_by_vehicle_and_time.csv')

'''
# 计算基础负荷、充电负荷和总负荷
base_loads = [phase_base_load['A'][t] + phase_base_load['B'][t] + phase_base_load['C'][t] for t in range(96)]
charging_loads = [0] * 96
for v in model.Vehicles:
    for t in model.TimePeriods:
        charging_loads[t] += model.ChargingStatus[v, t].value * P[v]
total_loads = [base_loads[t] + charging_loads[t] for t in range(96)]
max_loads = [model.MaxLoad() for _ in range(96)]

# 绘制图形
plt.figure(figsize=(12, 8))
plt.plot(base_loads, label='Base Load', linestyle='--')
plt.plot(charging_loads, label='Charging Load', linestyle='-.')
plt.plot(total_loads, label='Total Load', linewidth=2)
plt.plot(max_loads, label='Max Load Limit', linestyle=':')
plt.xlabel('Time Period')
plt.ylabel('Load')
plt.title('Load Comparison')
plt.legend()
plt.grid(True)
plt.savefig('E:\\EE5003\\pyomo1\\load_comparison.png')
plt.show()

# 保存数据到CSV
df = pd.DataFrame({'Time Period': range(96), 'Base Load': base_loads, 'Charging Load': charging_loads, 'Total Load': total_loads, 'Max Load': max_loads})
df.to_csv('E:\\EE5003\\pyomo1\\loads_data.csv', index=False)

# 初始化每个时间段三相的充电负荷列表
load_A, load_B, load_C = [0]*96, [0]*96, [0]*96

# 累加计算每个时间段的充电负荷
for v in model.Vehicles:
    for t in model.TimePeriods:
        if phi[v] == 'A':
            load_A[t] += model.ChargingStatus[v, t].value * P[v]
        elif phi[v] == 'B':
            load_B[t] += model.ChargingStatus[v, t].value * P[v]
        elif phi[v] == 'C':
            load_C[t] += model.ChargingStatus[v, t].value * P[v]

# 转换为三相总负荷
total_load_A = [phase_base_load['A'][t] + load_A[t] for t in range(96)]
total_load_B = [phase_base_load['B'][t] + load_B[t] for t in range(96)]
total_load_C = [phase_base_load['C'][t] + load_C[t] for t in range(96)]

# 初始化每个时间段的不平衡度列表
imbalance = []

for t in range(96):  # 假设96个时间段
    # 计算每个时间段的三相总负荷
    load_A = total_load_A[t]
    load_B = total_load_B[t]
    load_C = total_load_C[t]
    total_load = load_A + load_B + load_C

    # 计算两相之间的最大功率差异
    max_diff = max(abs(load_A - load_B), abs(load_A - load_C), abs(load_B - load_C))

    # 计算不平衡度：最大功率差异与三相总功率的比值
    if total_load > 0:  # 防止分母为0
        imbalance_value = 3*max_diff / total_load
    else:
        imbalance_value = 0
    imbalance.append(imbalance_value)

# 使用matplotlib绘制不平衡度图像
plt.figure(figsize=(10, 6))
plt.plot(range(96), imbalance, label='Three-Phase Imbalance', marker='o')
plt.xlabel('Time Period')
plt.ylabel('Imbalance Ratio')
plt.title('Three-Phase Imbalance Over Time')
plt.legend()
plt.grid(True)
plt.show()
'''