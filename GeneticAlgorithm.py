import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from juliacall import Main as jl
import juliapkg
import csv
import matplotlib.pyplot as plt
jl.seval("import JLD2")

'''
这是最始的遗传算法版本
'''

# Define row and col number
row = 5
col = 96

class three_phase_balance():
    def __init__(self, x, j, phase, phase_list):
        self.x = x
        self.j = j
        self.phase = phase
        self.phase_list = phase_list
    
    def max_phase(self):
        phase_X = np.where(self.phase_list == self.phase)
        phase_1 = np.where(self.x[:, self.j] == 1)
        time_list = []
        for i in range(row):
            if arrival_time_step[i] <= self.j <= departure_time_step[i]:
                time_list.append(i)
        time_constraint = np.array(time_list)
        phaseX_row = np.intersect1d(phase_X, phase_1)
        phaseX_row = np.intersect1d(phaseX_row, time_constraint)
        rowX = phaseX_row.shape
        # Get the row index randomly
        max_attempt = rowX[0]
        while max_attempt:
            # Set 0
            indexX = np.random.randint(0, rowX[0])
            indexX_row = phaseX_row[indexX]
            
            # Set 1
            X_col_time = np.arange(self.j, departure_time_step[indexX_row] + 1)
            X_col_0 = np.where(self.x[indexX_row, :] == 0)
            phaseX_col = np.intersect1d(X_col_time, X_col_0)
            phaseX_col = np.intersect1d(phaseX_col, t2)
            if np.size(phaseX_col) == 0:
                # Get indexA again. IndexA_col is empty. This indexA_col can't meet SOC constraint.
                max_attempt -= 1
            else:
                self.x[indexX_row][self.j] = 0
                colx = phaseX_col.shape
                indexX = np.random.randint(0, colx)
                indexX_col = phaseX_col[indexX]
                self.x[indexX_row][indexX_col] = 1
                break
        if max_attempt == 0:
            self.x[indexX_row][self.j] = 0
    
    def min_phase(self):
        phase_X = np.where(self.phase_list == self.phase)
        phase_0 = np.where(self.x[:, self.j] == 0)
        time_list = []
        for i in range(row):
            if arrival_time_step[i] <= self.j <= departure_time_step[i]:
                time_list.append(i)
        time_constraint = np.array(time_list)
        phaseX_row = np.intersect1d(phase_X, phase_0)
        phaseX_row = np.intersect1d(phaseX_row, time_constraint)
        rowX = phaseX_row.shape
        # Get the row index randomly
        max_attempt = rowX[0]
        while max_attempt:
            # Set 1
            indexX = np.random.randint(0, rowX[0])
            indexX_row = phaseX_row[indexX]
            
            # Set 0
            X_col_time = np.arange(self.j, departure_time_step[indexX_row] + 1)
            X_col_0 = np.where(self.x[indexX_row, :] == 1)
            phaseX_col = np.intersect1d(X_col_time, X_col_0)
            phaseX_col = np.intersect1d(phaseX_col, t2)
            if np.size(phaseX_col) == 0:
                # Get indexA again. IndexA_col is empty. This indexA_col can't meet SOC constraint.
                max_attempt -= 1
            else:
                self.x[indexX_row][self.j] = 1
                colx = phaseX_col.shape
                indexX = np.random.randint(0, colx)
                indexX_col = phaseX_col[indexX]
                self.x[indexX_row][indexX_col] = 0
                break
        if max_attempt == 0:
            self.x[indexX_row][self.j] = 1

def PILP_algorithm(n):
    #H = Estimate_Heats(w, r, W, η)
    P = Custom_Initialization(n)
    #B = Update_Best(P)
    k = Updata_Best(P)
    B = P[k][0]
    f_T = P[k][1]
    f_log = []
    t = 0
    err = 1
    
    while err >= 1e-16:
        Q = []
        while len(Q) < n:
            ID1, ID2, ID3, ID4 = random.sample(range(n), 4)
            parent1 = tournament_Selection(P[ID1], P[ID2])
            parent2 = tournament_Selection(P[ID3], P[ID4])
            offspring = Custom_Recombination(parent1[0], parent2[0])
            repair0 = Custom_Repair0(offspring)
            repair1 = Custom_Repair1(repair0)
            repair2 = Custom_Repair2(repair1)
            repair3 = Custom_Repair3(repair2)
            fitness = F(repair3)
            Q.append((repair2, fitness))
        
        k = Updata_Best(Q)
        B = Q[k][0]
        err = abs(f_T - Q[k][1])
        f_T = Q[k][1]
        f_log.append(f_T)
        P = Q
        t += 1
        print(t, f_T, Calculate_punishment(B))
    
    f_T = F(B)
    
    return B, f_T, f_log

def Custom_Initialization(n):
    # Initialize n solutions,种群数量
    P = []
    for k in range(n):
        x = np.random.randint(0, 2, size=(row, col))
        x = Custom_Repair0(x)
        x = Custom_Repair1(x)
        x = Custom_Repair2(x)
        x = Custom_Repair3(x)
        fitness = F(x)

        P.append((x, fitness))

    return P

def Updata_Best(P):
    # Find the best feasible solution in P
    best_fitness = float('+inf')
    # best_solution = None

    for k in range(len(P)):
        solution, fitness = P[k]
        if fitness < best_fitness:
            best_k = k
            best_fitness = fitness

    return best_k
    #return best_solution

def tournament_Selection(x1, x2):
    # Perform tournament selection and return the better solution
    return x1 if x1[1] < x2[1] else x2

def Custom_Recombination(parent1, parent2):
    # Perform custom recombination to create a new solution
    charging_col_parent1 = np.sum(power * parent1 * ρ, axis=1)
    charging_col_parent2 = np.sum(power * parent2 * ρ, axis=1)

    if charging_col_parent1[0] > charging_col_parent2[0]:
        offspring = parent2[0, :]
    else:
        offspring = parent1[0, :]

    for i in range(1, row):
        if charging_col_parent1[i] > charging_col_parent2[i]:
            # If the first parent has better charging
            offspring  = np.vstack((offspring, parent2[i, :]))
        else:
            # Else, child solution inherits assignment from second parent
            offspring  = np.vstack((offspring, parent1[i, :]))

    return offspring

def Calculate_used_power(x):
    power_col = x * power
    used_power = np.sum(power_col, axis = 0)

    return used_power

def Calculate_SOC_increment(x):
    # 为电池SOC约束做准备，计算每辆车15min的充电增量
    # 每辆车15min的单位增量
    unit_increment = power * efficiency * 0.25 / capacity 
    increment_list = np.multiply(unit_increment, x)
    # 计算每辆车所有时间步的增量
    total_increment = np.sum(increment_list, axis = 1) 

    return total_increment

def Calculate_max_SOC(x):
    # 计算可由开关控制实现的最大 SOC 值
    # 每辆车15min的单位增量
    unit_increment = power * efficiency * 0.25 / capacity 
    unit_increment = unit_increment.reshape((row))
    # 可由开关控制实现的最大 SOC 值
    max_SOC = np.floor((sd - sa) / unit_increment) * unit_increment 

    return max_SOC

def Distinguish_phase():
    # Get a array to distinguish phase A,B,C
    phase_list = np.empty((row), dtype = str)
    for vehicle in range(row):
        # Get three phaseA,B,C load
        if list2[vehicle].φ == 'A':
            phase_list[vehicle] = 'A'
        elif list2[vehicle].φ == 'B':
            phase_list[vehicle] = 'B'
        else:
            phase_list[vehicle] = 'C'
    return phase_list

def Calculate_phase(x, phase_list):
    # Get phase A,B,C charging power
    power_list = x * power
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

def Calculate_result_phase(three_phase):
    # Get max(phase A,B,C) and min(phase A,B,C)
    max_phase = np.max(three_phase, axis = 0)
    min_phase = np.min(three_phase, axis = 0)
    # Get max imbalance
    result_phase = 3 * (max_phase - min_phase)

    return result_phase

def Calculate_max_imbalance(three_phase):
    total_power = np.sum(three_phase, axis = 0)
    max_imbalance = max_imbalance_limit * total_power

    return max_imbalance

def Custom_Repair0(x):
    # x[:, len(ρ1)] = 0
    # x[:, len(ρ1):len(ρ2)] = 1
    variation = np.random.random((row, col))
    high_price = len(ρ1)
    low_price = len(ρ)
    # high price time set 0
    x1 = np.where(variation[:, :high_price] > probability, 1, 0)
    # low price time set 1
    x2 = np.where(variation[:, high_price:low_price] > probability, 0, 1)
    x = np.hstack((x1, x2))

    return x

def Custom_Repair1(x):
    # Repair offspring for time constraint satisfaction
    for i, row in enumerate(x):
        lower_bound = arrival_time_step[i]
        upper_bound = departure_time_step[i]
        row[:lower_bound] = 0
        row[upper_bound:] = 0

    return x

def Custom_Repair2(x):
    # Repair offspring for SOC capacity constraint satisfaction
    total_increment = Calculate_SOC_increment(x)
    max_SOC = Calculate_max_SOC(x)
    for i, row in enumerate(x):
        lower_bound = arrival_time_step[i]
        upper_bound = departure_time_step[i]
        while total_increment[i] != max_SOC[i]:
            if total_increment[i] < max_SOC[i]:
                while True:
                    random_index = np.random.choice(t2)
                    if lower_bound <= random_index <= upper_bound:
                        row[random_index] = 1
                        break
            else:
                while True:
                    random_index = np.random.randint(lower_bound, upper_bound)
                    if lower_bound <= random_index <= upper_bound:
                        row[random_index] = 0
                        break
            total_increment = Calculate_SOC_increment(x)
            max_SOC = Calculate_max_SOC(x)

    return x

def Custom_Repair3(x):
    # Repair offspring for circuit load constraint satisfaction
    used_power = Calculate_used_power(x)
    for j in range(len(x[0])):
        while (used_power[j] - restriction_in_power[j] > 0):
            total_increment = Calculate_SOC_increment(x)
            max_row = np.argmax(total_increment)
            x[max_row][j] = 0
    # Repair offspring for three phase imbalance constraint satisfaction
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    result_phase = Calculate_result_phase(three_phase)
    max_imbalance = Calculate_max_imbalance(three_phase)
    for j in range(len(x[0])):
        while result_phase[j] > max_imbalance[j]:
            # Get max phase to set 0
            max_phase = np.argmax(three_phase[:, j])
            if max_phase == 0:
                # A is max phase
                repairA = three_phase_balance(x, j, 'A', phase_list)
                repairA.max_phase()
                
            elif max_phase == 1:
                # B is max phase
                repairB = three_phase_balance(x, j, 'B', phase_list)
                repairB.max_phase()

            else:
                # C is max phase
                repairC = three_phase_balance(x, j, 'C', phase_list)

            # Get min phase to set 1
            min_phase = np.argmin(three_phase[:, j])
            if min_phase == 0:
                # A is min phase
                repairA = three_phase_balance(x, j, 'A', phase_list)
                repairA.min_phase()

            elif min_phase == 1:
                # B is min phase
                repairB = three_phase_balance(x, j, 'B', phase_list)
                repairB.min_phase()

            else:
                # C is min phase
                repairC = three_phase_balance(x, j, 'C', phase_list)
                repairC.min_phase()
            # Recalculate three imbalance
            three_phase = Calculate_phase(x, phase_list)
            result_phase = Calculate_result_phase(three_phase)
            max_imbalance = Calculate_max_imbalance(three_phase)
    
    return x

def Calculate_R():
    Δ = np.min(power * efficiency * 0.25 / capacity)
    return np.sum(ρ * np.full((col), power.max()) * 0.25) / Δ

def Calculate_punishment(x):
    term1 = np.sum(np.abs(Calculate_SOC_increment(x) - Calculate_max_SOC(x)))
    load_overrun = Calculate_used_power(x) - restriction_in_power
    term2 = np.sum(load_overrun[load_overrun > 0])
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    result_phase = Calculate_result_phase(three_phase)
    max_imbalance = Calculate_max_imbalance(three_phase)
    imbalance_overrun = result_phase - max_imbalance
    term3 = np.sum(imbalance_overrun[imbalance_overrun > 0])
    return R * (term1 + term2 + term3)

def F(solution):
    # Calculate the objective value of the solution
    x = solution
    used_power = Calculate_used_power(x)
    charging_cost = used_power * ρ * 0.25
    punishment = Calculate_punishment(x)
    return np.sum(charging_cost) + punishment

def Plot_circuit_load(x):
    step = np.arange(1, col + 1)
    limit_power = np.full((col), 2000)
    used_power = Calculate_used_power(x)
    total_power = base_load + used_power
    plt.figure(1, figsize=(12, 8))
    plt.ylim((0,2600))
    plt.plot(step, used_power, label = 'Charging load')
    plt.plot(step, total_power, label = 'Total load')
    plt.plot(step, base_load, label = 'Basic load')
    plt.plot(step, limit_power, label = 'Power limitation')
    plt.legend(["Charging load","Total load","Basic load","Power limitation"],loc='upper right',fontsize='x-small')
    plt.xlabel("step")
    plt.ylabel("power/kw")
    plt.show()

def Plot_phase_imbalance(x):
    step = np.arange(1, col + 1)
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    total_power = np.sum(three_phase, axis = 0)
    result_phase = Calculate_result_phase(three_phase)
    imbalance = result_phase / total_power
    limit_imbalance = np.full((col), max_imbalance_limit)
    plt.figure(2, figsize=(12, 8))
    plt.ylim((0,max_imbalance_limit))
    plt.plot(step, limit_imbalance, color = 'r', label = 'limit imbalance')
    plt.plot(step, imbalance, marker = '.', color = 'b', label = 'imbalance')
    plt.show()

def Plot_time_constraint(x):
    # 获取矩阵中数值为1的元素的坐标
    rows, cols = np.where(x.T == 1)
    cols += 1
    # 绘制散点图
    plt.figure(3, figsize=(12, 8))
    plt.xlim((0, row + 1))
    plt.ylim((0, col))
    plt.scatter(cols, rows, c='red', marker='o', s = 3)

    # 画出时间上下限
    vehicle = np.arange(1, row + 1)
    plt.plot(vehicle, arrival_time_step, color = 'b', marker = '.')
    plt.plot(vehicle, departure_time_step, color = 'b', marker = '.')
    plt.show()

class CTask:
    def __init__(self, v, τa, ta, τd, td, sa, sd, E, P, η, i, φ):
        self.v = v
        self.τa = τa
        self.ta = ta
        self.τd = τd
        self.td = td
        self.sa = sa
        self.sd = sd
        self.E = E
        self.P = P
        self.η = η
        self.i = i
        self.φ = φ
        
    @classmethod
    def from_list(cls, inlist):
        v, τa, ta, τd, td, sa, sd, E, P, η, i, φ = inlist
        return cls(v, τa, ta, τd, td, sa, sd, E, P, η, i, φ)

''' power--每辆车的充电功率;
    efficiency--每辆车的充电效率
    capacity--每辆车的电池容量
    ρ--电价
    R--惩罚系数
    n--种群数量
    probability--变异概率'''

list2 = []
arrival_time_step = np.full((row), fill_value=5)
departure_time_step = np.full((row), fill_value=90)
power = np.zeros((row, 1))
efficiency = np.zeros((row, 1))
capacity = np.zeros((row, 1))
sa = np.zeros((row))
sd = np.zeros((row))
ρ1 = np.full((30), 1)
ρ2 = np.full((col - 30), 0.1)
ρ = np.hstack((ρ1 , ρ2))
R = 1000
n = 10
probability = 0.95

# 获取ρ1在ρ中的索引位置
t1 = np.where(np.isin(ρ, ρ1))[0]
# 获取ρ2在ρ中的索引位置
t2 = np.where(np.isin(ρ, ρ2))[0]

tasks = jl.JLD2.load("./tasks_s123_n100_py.jld2", "tasks")
phases = jl.JLD2.load("./tasks_s123_n100_py.jld2", "phases")
phase_base_load = jl.JLD2.load("D:/MyProject/Charge_Pyomo/code/notebook/case2-updated/img/data/ILP_s123_n100.jld2", "base_load")
phase_base_load = np.array(phase_base_load)

for i in range(row):
    list1 = []
    list1.append(tasks[i].v)
    list1.append(tasks[i].τa)
    #list1.append(tasks[i].ta)
    #arrival_time_step[i] = tasks[i].ta
    list1.append(5)
    list1.append(tasks[i].τd)
    #list1.append(tasks[i].td)
    #departure_time_step[i] = tasks[i].td
    list1.append(90)
    list1.append(tasks[i].sa)
    sa[i] = tasks[i].sa
    list1.append(tasks[i].sd)
    sd[i] = tasks[i].sd
    list1.append(tasks[i].E)
    capacity[i] = tasks[i].E
    list1.append(tasks[i].P)
    power[i] = tasks[i].P
    list1.append(tasks[i].η)
    efficiency[i] = tasks[i].η
    list1.append(tasks[i].i)
    list1.append(phases[i])
    # 通过列表创建一个示例CTask对象
    ctask_from_list = CTask.from_list(list1)
    list2.append(ctask_from_list)


#获得基础负载和电路限制get_restriction_in_power
base_load = np.sum(phase_base_load, axis=0)
restriction_in_power = 1900 - base_load

# 最大三相不平衡度
max_imbalance_limit = 0.02

solution, f_T, f_log = PILP_algorithm(n)

Plot_circuit_load(solution)
Plot_phase_imbalance(solution)
Plot_time_constraint(solution)