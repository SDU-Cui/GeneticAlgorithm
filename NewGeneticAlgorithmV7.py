import random
import numpy as np
import matplotlib.pyplot as plt
import time

'''
这是遗传算法的版本V7 因为V6多级Tournament Selection也不能解决三相不平衡问题
所以决定重新构想repair策略，全部变为硬约束
'''

# Define row and col number
row = 400
col = 96
times = 400/200 #功率放大倍数

start = time.time()

tasks = np.genfromtxt('data/vehicle_data_400.csv', delimiter=',', names=True, dtype=None, encoding='ANSI')
phase_base_load = np.genfromtxt('data/phase_base_load.csv', delimiter=',', dtype=None, encoding='UTF-8')
phase_base_load *= times

''' power--每辆车的充电功率;
    efficiency--每辆车的充电效率
    capacity--每辆车的电池容量
    ρ--电价
    R--惩罚系数
    n--种群数量
    probability--变异概率'''

arrival_time_step = tasks['ta']
departure_time_step = tasks['td']
power = np.expand_dims(tasks['P'], axis=1)
efficiency = np.expand_dims(tasks['η'], axis=1)
capacity = np.expand_dims(tasks['E'], axis=1)
sa = tasks['sa']
sd = tasks['sd']
Φ = tasks['Φ']
Δ = tasks['Δ']
ρ1 = np.full((30), 1)
ρ2 = np.full((col - 30), 0.1)
ρ = np.hstack((ρ1 , ρ2))
#R = 20 #惩罚系数
n = 160 #种群数量
probability = 0.1 #变异率
α = 10 #三相不平衡系数

#获得基础负载和电路限制get_restriction_in_power
base_load = np.sum(phase_base_load, axis=0)
restriction_in_power = 2200 * times - base_load 

# 最大三相不平衡度
max_imbalance_limit = 0.04

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
        if rowX[0] == 0:
            return
        indexX = np.random.randint(0, rowX[0])
        indexX_row = phaseX_row[indexX]
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
        if rowX[0] == 0:
            return
        indexX = np.random.randint(0, rowX[0])
        indexX_row = phaseX_row[indexX]
        self.x[indexX_row][self.j] = 1

def PILP_algorithm(n):
    P = Custom_Initialization(n)
    k = Updata_Best(P)
    B = P[k][0]
    f_log = [] # 用来记录电价、惩罚、三相三个适应度值
    f0_log = [] # 只记录电价
    f_log.append(P[k][1])
    f0_log.append(P[k][1][0])
    t = 0
    err = 1
    
    while err >= 1e-10:
        Q = Elite(P)
        while len(Q) < n:
            ID1, ID2, ID3, ID4 = random.sample(range(n), 4)
            parent1 = tournament_Selection(P[ID1], P[ID2])
            parent2 = tournament_Selection(P[ID3], P[ID4])
            offspring = Custom_Recombination(parent1[0], parent2[0])
            #repair0 = Custom_Repair0(offspring)
            offspring = Variation(offspring)
            repair1 = Custom_Repair1(offspring)
            repair2 = Custom_Repair2(repair1)
            repair3 = Custom_Repair3(repair2)
            # repair2_2 = Custom_Repair2_2(repair3)
            # repair3 = Variation(repair3)
            fitness = F(repair3)
            Q.append((repair3, fitness))
        
        k = Updata_Best(Q)
        B = Q[k][0]
        #f_T = Q[k][1]
        f_log.append(Q[k][1])
        f0_log.append(Q[k][1][0])
        if len(f0_log) < 4:
            err = 1
        else:
            err = 0
            for i in range(1, 4):
                err += abs(sum(f_log[-i]) - sum(f_log[-i - 1]))
        #f_T = Q[k][1]
        #f_log.append(f_T)
        P = Q
        t += 1
        print(t, f_log[-1])
        #Print_F(B)
    
    f_T = F(B)
    
    return B, f_T, f0_log

def Calculate_k():
    # 计算每辆车需要充几次电
    # 为电池SOC约束做准备，计算每辆车15min的充电增量
    unit_increment = power * efficiency * 0.25 / capacity #每辆车15min的单位增量
    unit = unit_increment.ravel() #将二维的unit_increment将至一维
    k = (sd - sa) // unit

    return k

def Custom_Initialization(n, k):
    # Initialize n solutions,种群数量
    P = []
    # 第 p 个个体
    for p in range(n):
        # 第 n 个个体
        v = []
        # 第 i 辆车
        for i in row:
            # 产生一个 sa 到 sd 时间段的随机数矩阵
            s = np.random.rand(sd[i] - sa[i])
            # 找到第 k 小的值
            maxk = np.sort(s)[k[i]]
            vi = np.where(s <= maxk, 1 , 0)
            v.append(vi)

    return P

def Sort(P, k):
    # 按适应度值从小到大排序，返回索引，k代表按哪一项排序，有电价、惩罚、三相不平衡三项
    # 而函数Sort_sumfitness代表按照三者之和排序
    return sorted(range(len(P)), key=lambda i: P[i][k])

def Sort_sumfitness(P):
    # 这个函数和Sort的区别在于，Sort是按照k代表的电价、惩罚、三相不平衡其中一项排序
    # 这里是按照三者之和排序
    return sorted(range(len(P)), key=lambda i: P[i][2])

def Elite(P):
    #经营策略，将5%的最优解直接到下一代
    num = round(n * 0.02)
    Q = []
    for k in range(3):
        top_index = Sort(P, k)[: num]
        Q += [P[i] for i in top_index]
    
    return Q

def Updata_Best(P):
    # Find the best feasible solution in P
    # best_fitness = float('+inf')
    # best_solution = None

#    for k in range(len(P)):
#        solution, fitness = P[k]
#        if fitness < best_fitness:
#            best_k = k
#            best_fitness = fitness

    best_k = Sort_sumfitness(P)[0] #选择电价、惩罚、三相之和最低的作为最优

    return best_k
    #return best_solution

def tournament_Selection(x1, x2, i):
    # Perform tournament selection and return the better solution
    return x1 if x1[1][i] < x2[1][i] else x2

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

def Part_Full(p):
    # 将一个个体由不完整，即只有[ta, td]时间段的矩阵，补全为 row * col 的矩阵
    front_zero = np.zeros()

def Variation(x):
    pro = random.random()
    if pro < probability:
        var = np.random.random((row, col))
        x = np.where(var < ρ_map, 0, 1)

    return x

def F(solution):
    # Calculate the objective value of the solution
    x = solution
    used_power = Calculate_used_power(x)
    charging_cost = used_power * ρ * 0.25
    punishment = Calculate_punishment(x)
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    result_phase = Calculate_imbalance(three_phase)
    #α = Calculate_α(result_phase)
    ρ2 = np.count_nonzero(result_phase) / col
    c2 = pow(10, α * ρ2)
    
    return (np.sum(charging_cost), punishment, np.sum(c2 * result_phase))

def Plot_circuit_load(x):
    step = np.arange(1, col + 1)
    limit_power = np.full((col), 2200 * times)
    used_power = Calculate_used_power(x)
    total_power = base_load + used_power
    # plt.figure(1, figsize=(12, 8))
    plt.title("Circuit load")
    plt.ylim((0,2600 * times))
    plt.plot(step, used_power, label = 'Charging load')
    plt.plot(step, total_power, label = 'Total load')
    plt.plot(step, base_load, label = 'Basic load')
    plt.plot(step, limit_power, label = 'Power limitation')
    plt.legend(["Charging load","Total load","Basic load","Power limitation"],loc='upper right',fontsize='x-small')
    plt.xlabel("step")
    plt.ylabel("power/kw")
    # plt.show()

def Plot_phase_imbalance(x):
    step = np.arange(1, col + 1)
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    total_power = np.sum(three_phase, axis = 0)
    result_phase = Calculate_result_phase(three_phase)
    imbalance = result_phase / total_power
    limit_imbalance = np.full((col), max_imbalance_limit)
    # plt.figure(2, figsize=(12, 8))
    plt.title("Phase imbalance")
    #plt.ylim((0,0.1))
    plt.plot(step, limit_imbalance, color = 'r', label = 'limit imbalance')
    plt.plot(step, imbalance, marker = '.', color = 'b', label = 'imbalance')
    plt.legend(["limit imbalance", "imbalance"],loc='upper right',fontsize='x-small')
    # plt.show()

def Plot_time_constraint(x):
    # 获取矩阵中数值为1的元素的坐标
    rows, cols = np.where(x.T == 1)
    cols += 1
    # 绘制散点图
    # plt.figure(3, figsize=(12, 8))
    plt.title("Time constraint")
    plt.xlim((0, row + 1))
    plt.ylim((0, col))
    plt.scatter(cols, rows, c='red', marker='o', s = 3)

    # 画出时间上下限
    vehicle = np.arange(1, row + 1)
    plt.plot(vehicle, arrival_time_step, color = 'b', marker = '.')
    plt.plot(vehicle, departure_time_step, color = 'b', marker = '.')
    # plt.show()

def Plot_Evolution_curve(f_log):
    # plt.figure(4)
    plt.title("Evolution curve")
    plt.plot(f_log)
    # plt.show()

def Plot_SOC(x):
    step = np.arange(1, row + 1)
    SOC = Calculate_SOC_increment(x) + sa
    plt.plot(step, SOC, c = 'red', marker = '*')
    plt.plot(step, sd, c = 'blue', marker='o')
    plt.show()

def Plot(x, f_log):
    fig, axes = plt.subplots(2, 2)
    fig.set_size_inches(16, 8)

    Plot_Evolution_curve(f_log)
    plt.sca(axes[0, 0])

    Plot_time_constraint(x)
    plt.sca(axes[0, 1])

    Plot_circuit_load(x)
    plt.sca(axes[1, 0])

    Plot_phase_imbalance(x)
    plt.sca(axes[1, 1])

    plt.tight_layout()  # 自动调整子图的布局
    plt.show()

ρ_map = Price_Probability(ρ, 0.1, 0.9)
ρ_map = np.tile(ρ_map, (row, 1))

solution, f_T, f_log = PILP_algorithm(n)

end = time.time()

print(end - start)
Plot(solution, f_log)
Plot_SOC(solution)