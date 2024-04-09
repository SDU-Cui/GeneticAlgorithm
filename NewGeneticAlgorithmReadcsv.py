import numpy as np
import NewGeneticAlgorithmV7Function as ga

'''
读取csv数据，定义变量
'''

# Define row and col number
row = 200
col = 96
# 功率放大倍数
times = row/200

tasks = np.genfromtxt('data/vehicle_data_200(2).csv', delimiter=',', names=True, dtype=None, encoding='ANSI')
phase_base_load = np.genfromtxt('data/phase_base_load.csv', delimiter=',', dtype=None, encoding='UTF-8')
phase_base_load *= times

''' power--每辆车的充电功率;
    efficiency--每辆车的充电效率
    capacity--每辆车的电池容量
    ρ--电价
    R--惩罚系数
    n--种群数量
    probability--变异概率'''

ta = tasks['ta']
td = tasks['td']
power = tasks['P']
efficiency = tasks['η']
capacity = tasks['E']
sa = tasks['sa']
sd = tasks['sd']
Φ = tasks['Φ']
Δ = tasks['Δ']
# 电价
ρ1 = np.full((30), 1)
ρ2 = np.full((col - 30), 0.1)
ρ = np.hstack((ρ1 , ρ2))
# 惩罚系数
R = row * col * max(ρ)
# 种群数量
n = 20
# 变异率
probability = 0.4
# 价格映射
map_ρ = ga.Price_Probability(ρ, 0.1, 0.9)
# 三相不平衡系数
α = 10

# 最大三相不平衡度
max_imbalance_limit = 0.04

# 最大功率上限
max_power_limit = 2200 * times

# 获得基础负载和电路限制restriction_power
base_load = np.sum(phase_base_load, axis=0)
restriction_power = max_power_limit - base_load