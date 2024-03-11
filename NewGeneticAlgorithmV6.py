import random
import numpy as np
import matplotlib.pyplot as plt
import time

'''
这是遗传算法的版本V6 因为V5无法在电价、惩罚和三相不平衡之间兼顾，尝试多级Tournament Selection
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

# 获取ρ1在ρ中的索引位置
t1 = np.where(np.isin(ρ, ρ1))[0]
# 获取ρ2在ρ中的索引位置
t2 = np.where(np.isin(ρ, ρ2))[0]

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
    #H = Estimate_Heats(w, r, W, η)
    P = Custom_Initialization(n)
    #B = Update_Best(P)
    k = Updata_Best(P)
    B = P[k][0]
    #f_T = P[k][1]
    f_log = [] # 用来记录电价、惩罚、三相三个适应度值
    f0_log = [] # 只记录电价
    f_log.append(P[k][1])
    f0_log.append(P[k][1][0])
    t = 0
    err = 1
    
    while err >= 1e-10:
        Q = Elite(P)
        D1 = [] # 用来记录刚产生的子代
        D2 = [] # 用来记录repair后的子代
        P1 = P # 用来记录三轮竞争后的父代 
        P2 = [] # 暂存竞争的父代，最后赋值给P1
        for i in range(3):
            #通过三次竞争获得父代
            for j in range(len(P1) // 4 - i * 10):
                ID1, ID2 = random.sample(range(len(P1)), 2)
                parent = tournament_Selection(P1[ID1], P1[ID2], 2 - i)
                P2.append(parent)
        P1 = P2
        
        while len(D1) < n:
            ID1, ID2 = random.sample(range(len(P1)), 2)
            offspring = Custom_Recombination(P1[ID1][0], P1[ID2][0])
            D1.append(offspring)
        
        print('before repair diversity:{}'.format(Diversity(D1)))

        for i in D1:
            offspring = i
            offspring = Variation(offspring)
            repair1 = Custom_Repair1(offspring)
            repair2 = Custom_Repair2(repair1)
            repair3 = Custom_Repair3(repair2)
            D2.append((repair3))
        
        print('after repair diversity:{}'.format(Diversity(D2)))

        for i in D2:
            fitness = F(i)
            Q.append((i, fitness, sum(fitness)))
        
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

def hammingDistance(x, y):
    # 计算汉明距离，确定种群多样性
    x = np.ravel(x)
    y = np.ravel(y)
    xor = x ^ y
    distance = 0
    # 每次右移，最左边都会补零，因此截止条件是xor已经是一个零值了
    for i in range(len(xor)):
        if xor[i] & 1:
            distance = distance + 1

    return distance

def Distance(x, y):
    distance = np.linalg.norm(x - y)

    return distance

def Centre(P):
    return np.average(P, axis=0)

def Diversity(P):
    z = 0
    centre = Centre(P)
    for i in range(len(P) - 1):
        x = P[i]
        z += Distance(x, centre)
    return z

def Custom_Initialization(n):
    # Initialize n solutions,种群数量
    P = []
    for k in range(n):
        x = np.random.randint(0, 2, size=(row, col))
        x = Custom_Repair0(x)
        x = Custom_Repair1(x)
        x = Custom_Repair2(x)
        x = Custom_Repair3(x)
        # x = Custom_Repair2_2(x)
        fitness = F(x)

        P.append((x, fitness, sum(fitness)))

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

def Calculate_max_SOC():
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
        if Φ[vehicle] == 'A':
            phase_list[vehicle] = 'A'
        elif Φ[vehicle] == 'B':
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

def Price_Probability(ρ, MIN, MAX):
    """
    归一化映射到任意区间
    :param data: 数据
    :param MIN: 目标数据最小值
    :param MAX: 目标数据最大值
    :return:
    """
    d_min = np.min(ρ)    # 当前数据最大值
    d_max = np.max(ρ)    # 当前数据最小值
    return MIN + (ρ - d_min) * (MAX - MIN)/(d_max - d_min)

def Variation(x):
    pro = random.random()
    if pro < probability:
        var = np.random.random((row, col))
        x = np.where(var < ρ_map, 0, 1)

    return x

def Custom_Repair0(x):
    # x[:, len(ρ1)] = 0
    # x[:, len(ρ1):len(ρ2)] = 1
    variation = np.random.random((row, col))
    high_price = len(ρ1)
    low_price = len(ρ)
    # high price time set 0
    x1 = np.where(variation[:, :high_price] < ρ_map[:, :high_price], 0, 1)
    # low price time set 1
    x2 = np.where(variation[:, high_price:low_price] < ρ_map[:, high_price:low_price], 0, 1)
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
    max_SOC = Calculate_max_SOC()
    for i, row in enumerate(x):
        lower_bound = arrival_time_step[i]
        upper_bound = departure_time_step[i]
        while total_increment[i] != max_SOC[i]:
            # print(total_increment[i], max_SOC[i])
            if total_increment[i] < max_SOC[i]:
                # random_index = np.random.choice(t2) 因为存在t2中已经无解的情况，应该全部搜索
                random_index = np.random.randint(lower_bound, upper_bound)
                x[i][random_index] = 1
            else:
                random_index = np.random.randint(lower_bound, upper_bound)
                row[random_index] = 0
            total_increment = Calculate_SOC_increment(x)
            if abs(total_increment[i] - max_SOC[i]) <= 1e-15:
                break

    return x

def Custom_Repair3(x):
    # Repair offspring for circuit load constraint satisfaction
    used_power = Calculate_used_power(x)
    for j in range(len(x[0])):
        total_increment = Calculate_SOC_increment(x)
        index = np.argsort(-total_increment)
        i = 0
        while (used_power[j] - restriction_in_power[j] > 0):
            power_loss = restriction_in_power - used_power
            max_row = index[i]
            upper_bound = departure_time_step[max_row]
            x[max_row][j] = 0   
            power_loss = power_loss[j:upper_bound + 1]
            if len(power_loss) > 0:
                x[max_row][np.argmax(power_loss) + j] = 1
            used_power = Calculate_used_power(x)
            i += 1
    # Repair offspring for three phase imbalance constraint satisfaction
    # phase_list = Distinguish_phase()
    # three_phase = Calculate_phase(x, phase_list)
    # result_phase = Calculate_result_phase(three_phase)
    # max_imbalance = Calculate_max_imbalance(three_phase)
    # for j in range(len(x[0])):
    #     while result_phase[j] > max_imbalance[j]:
    #         # Get max phase to set 0
    #         max_phase = np.argmax(three_phase[:, j])
    #         if max_phase == 0:
    #             # A is max phase
    #             repairA = three_phase_balance(x, j, 'A', phase_list)
    #             repairA.max_phase()
                
    #         elif max_phase == 1:
    #             # B is max phase
    #             repairB = three_phase_balance(x, j, 'B', phase_list)
    #             repairB.max_phase()

    #         else:
    #             # C is max phase
    #             repairC = three_phase_balance(x, j, 'C', phase_list)
    #             repairC.max_phase()

    #         # Get min phase to set 1
    #         min_phase = np.argmin(three_phase[:, j])
    #         if min_phase == 0:
    #             # A is min phase
    #             repairA = three_phase_balance(x, j, 'A', phase_list)
    #             repairA.min_phase()

    #         elif min_phase == 1:
    #             # B is min phase
    #             repairB = three_phase_balance(x, j, 'B', phase_list)
    #             repairB.min_phase()

    #         else:
    #             # C is min phase
    #             repairC = three_phase_balance(x, j, 'C', phase_list)
    #             repairC.min_phase()
    #         # Recalculate three imbalance
    #         three_phase = Calculate_phase(x, phase_list)
    #         result_phase = Calculate_result_phase(three_phase)
    #         max_imbalance = Calculate_max_imbalance(three_phase)
    
    return x

def Custom_Repair2_2(x):
    """ 在repair3破坏了repair2之后再次修复SOC """
    total_increment = Calculate_SOC_increment(x)
    max_SOC = Calculate_max_SOC()
    flag = 0
    for i, row in enumerate(x):
        lower_bound = arrival_time_step[i]
        upper_bound = departure_time_step[i] 
        while total_increment[i] != max_SOC[i]:
            used_power = Calculate_used_power(x)
            power_loss = restriction_in_power - used_power
            #index = np.argsort(- power_loss)
            index = np.where(power_loss > 1)#得到功率还有富裕的时间序列
            T = np.arange(lower_bound, upper_bound, 1)#得到第i辆车的到达和离开时刻
            index = np.intersect1d(index, T)
            index_lower = np.intersect1d(index, t2)#t2是低电价的时刻
            index_high = np.intersect1d(index, t1)#t1是高电价的时刻
            if len(index_lower) == 0 or len(index_high) == 0:
                break
            if total_increment[i] < max_SOC[i]:
                random_index = np.random.choice(index_lower)#这里低电价的时刻可能都已经是1了
                while x[i][random_index] == 1:#如果已经是1，那就应该全部搜索index，不再区分lower和high
                    random_index = np.random.choice(index)#其实这里可以加上最大尝试次数，超过说明无解
                    m = len(index)
                    q = np.where(row > 0)[0]#这里很奇怪，返回的是元组，不是矩阵，但是index返回的是矩阵
                    n = len(q)
                    if m <= n:
                        flag = 1
                        break
                x[i][random_index] = 1
            else:
                random_index = np.random.choice(index_high)
                while x[i][random_index] == 0:
                    random_index = np.random.choice(index)
                    if len(index) < len(np.where(row == 0)):
                        flag = 1
                        break
                x[i][random_index] = 0
            total_increment = Calculate_SOC_increment(x)
            if abs(total_increment[i] - max_SOC[i]) <= 1e-15 or flag == 1:
                flag = 0
                break

    return x

def Calculate_R():
    Δ = np.min(power * efficiency * 0.25 / capacity)
    return np.sum(ρ * np.full((col), power.max()) * 0.25) / Δ

def Calculate_punishment(x):
    SOC = np.abs(Calculate_SOC_increment(x) - Calculate_max_SOC())
    ρ1 = np.count_nonzero(SOC) / row
    c1 = pow(10, α * ρ1) # c1是term1的惩罚系数
    term1 = np.sum(SOC)
    load_overrun = Calculate_used_power(x) - restriction_in_power
    term2 = np.sum(load_overrun[load_overrun > 0])
    # phase_list = Distinguish_phase()
    # three_phase = Calculate_phase(x, phase_list)
    # result_phase = Calculate_result_phase(three_phase)
    # max_imbalance = Calculate_max_imbalance(three_phase)
    # imbalance_overrun = result_phase - max_imbalance
    # term3 = np.sum(imbalance_overrun[imbalance_overrun > 0])
    # soc_v = term1 / row
    # if soc_v < np.min(Δ):
    #     R = 0
    # else:
    #     R = 1000 * term1
    # R = 1000 * term1

    return c1 * (term1 + term2)

def Print_Punishment(x):
    total_increment = Calculate_SOC_increment(x)
    max_SOC = Calculate_max_SOC()
    term1 = np.sum(np.abs(total_increment - max_SOC))
    # print(total_increment - max_SOC)
    load_overrun = Calculate_used_power(x) - restriction_in_power
    term2 = np.sum(load_overrun[load_overrun > 0])
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    result_phase = Calculate_result_phase(three_phase)
    max_imbalance = Calculate_max_imbalance(three_phase)
    imbalance_overrun = result_phase - max_imbalance
    term3 = np.sum(imbalance_overrun[imbalance_overrun > 0])
    print(term1, term2, term3)

#def Calculate_α(result_phase):
    α = np.where(result_phase > 0.06, 1000 * result_phase, 0)

    return α

def Calculate_imbalance(three_phase):
    avg = np.mean(three_phase, axis=0)
    minus = three_phase - avg
    imbalance = np.sqrt(np.mean(np.square(minus), axis=0)) / avg

    return np.where(imbalance >= 0.01, imbalance, 0)

def Print_F(solution):
    # Print the objective value of the solution
    x = solution
    used_power = Calculate_used_power(x)
    charging_cost = used_power * ρ * 0.25
    punishment = Calculate_punishment(x)
    phase_list = Distinguish_phase()
    three_phase = Calculate_phase(x, phase_list)
    result_phase = Calculate_imbalance(three_phase)
    #α = Calculate_α(result_phase)
    
    print(np.sum(charging_cost), punishment, α * np.sum(result_phase))

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