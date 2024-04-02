import random
import numpy as np
import matplotlib.pyplot as plt
import time

'''
这是遗传算法的版本V8 因为V7虽然实现了约束全部满足，但是因为while循环无法证明是否可以跳出
所以决定采用 v1 + v2 >= v1' + v2' 的思路
'''

# Define row and col number
row = 200
col = 96
times = row/200 #功率放大倍数

start = time.time()

tasks = np.genfromtxt('data/vehicle_data_200(6).csv', delimiter=',', names=True, dtype=None, encoding='ANSI')
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
power = tasks['P']
efficiency = tasks['η']
capacity = tasks['E']
sa = tasks['sa']
sd = tasks['sd']
Φ = tasks['Φ']
Δ = tasks['Δ']
ρ1 = np.full((30), 1)
ρ2 = np.full((col - 30), 0.1)
ρ = np.hstack((ρ1 , ρ2))
R = row * col * max(ρ) #惩罚系数
n = 20 #种群数量
probability = 0.15 #变异率
α = 10 #三相不平衡系数

#获得基础负载和电路限制get_restriction_in_power
base_load = np.sum(phase_base_load, axis=0)
restriction_in_power = 2200 * times - base_load 

# 最大三相不平衡度
max_imbalance_limit = 0.04

def PILP_algorithm(n):
    k = Calculate_k()
    P, Fitness = Custom_Initialization(n, k)
    k_best = Updata_Best(Fitness)
    # B 记录最优个体
    B = P[k_best]
    # 用来记录每一代中最优的适应度值个体
    f_log = []
    f_log.append(Fitness[k_best])
    t = 0
    print(t, f_log[-1], F(B))
    err = 1
    
    while err >= 1e-10:
        # Q 为下一代种群
        Q = [P[k_best]]
        # Q_Fitness 为下一代种群的适应度值列表
        Q_Fitness = [Fitness[k_best]]
        while len(Q) < n:
            ID1, ID2, ID3, ID4 = random.sample(range(n), 4)
            parent1 = P[tournament_Selection(ID1, ID2, Fitness)]
            parent2 = P[tournament_Selection(ID3, ID4, Fitness)]
            offspring = Custom_Recombination(parent1, parent2)
            x1 = mutation(offspring)
            x2 = Repair_Load(x1)
            x3 = Repair_Imbalance(x2)
            fitness = F(x3)
            Q_Fitness.append(fitness)
            Q.append(x3)
        
        k_best = Updata_Best(Q_Fitness)
        B = Q[k_best]
        f_log.append(Q_Fitness[k_best])
        if len(f_log) < 4:
            err = 1
        else:
            differences = [abs(f_log[-1] - f_log[-2]), 
               abs(f_log[-2] - f_log[-3]), 
               abs(f_log[-3] - f_log[-4])]
            err = sum(differences)
        
        P = Q
        Fitness = Q_Fitness
        t += 1
        print(t, f_log[-1], F(B))
    
    return B, f_log

def Calculate_k():
    # 计算每辆车需要充几次电
    # 为电池SOC约束做准备，计算每辆车15min的充电增量
    unit_increment = power * efficiency * 0.25 / capacity #每辆车15min的单位增量
    # unit = unit_increment.ravel() #将二维的unit_increment将至一维
    k = (sd - sa) // unit_increment

    return k.astype(int)

def Custom_Initialization(n, k):
    """ 
    输入种群数量 n, 每辆车(行)的期望 k 值(SOC)
    Initialize n solutions,种群数量
    返回一个种群列表 list(list(array())) P 和 适应度值列表 Fitness
    """
    P = []
    Fitness = []
    # 第 p 个个体
    for p in range(n):
        # 第 n 个个体
        v = []
        # 第 i 辆车
        for i in range(row):
            # 产生一个 ta 到 td 时间段的随机数矩阵
            while True:
                s = np.random.rand(departure_time_step[i] - arrival_time_step[i] + 1)
                s_set = set(s.flatten())
                if len(s_set) == s.size:
                    # 说明随机产生的矩阵 s 不存在相同元素
                    break
                else:
                    continue
                    
            # 找到第 k 小的值
            maxk = np.sort(s)[k[i] - 1]
            vi = np.where(s <= maxk, 1 , 0).astype(int)
            v.append(vi)

        v1 = Repair_Load(v)
        v2 = Repair_Imbalance(v1)
        P.append(v2)
        Fitness.append(F(v2))

    return (P, Fitness)

def Updata_Best(Fitness):
    '''
    输入一个种群的适应度值列表 Fitness
    函数是为了找到适应度值最小的个体
    返回适应度值最小的个体的在种群 P 中的索引值
    '''
    best_k = np.argmin(Fitness)
    
    return best_k

def tournament_Selection(x1, x2, Fitness):
    '''
    输入为两个个体在种群中的索引值(不完整的列表) 和适应度值列表
    函数可以返回适应度值更优的个体 # Perform tournament selection and return the better solution
    返回更优的个体索引值
    '''
    
    return x1 if Fitness[x1] < Fitness[x2] else x2

def Custom_Recombination(parent1, parent2):
    '''
    输入为两个父代个体(不完整的列表)
    函数是交叉，行交叉，可以保证时间约束和SOC约束不破坏 # Perform custom recombination to create a new solution
    返回一个子代(不完整的列表)
    '''
    parent1_arr = Part_Full(parent1)
    parent2_arr = Part_Full(parent2)
    charging_col_parent1 = np.sum(np.expand_dims(power, axis=1)
                                   * parent1_arr * np.expand_dims(ρ, axis=0), axis=1)
    charging_col_parent2 = np.sum(np.expand_dims(power, axis=1)
                                   * parent2_arr * np.expand_dims(ρ, axis=0), axis=1)
    
    # offspring 是新产生的子代 
    offspring = []

    for i in range(0, row):
        if charging_col_parent1[i] > charging_col_parent2[i]:
            # If the first parent has better charging
            offspring.append(np.copy(parent2[i]))
        else:
            # Else, child solution inherits assignment from second parent
            offspring.append(np.copy(parent1[i]))

    return offspring

def Part_Full(p):
    """
    输入 p 是一个个体(不完整的列表)
    函数将一个个体由不完整，即只有[ta, td]时间段的矩阵，补全为 row * col 的矩阵
    返回一个 row * col 矩阵
    """
    # 新建补全的矩阵，不然会改变原来的不完整列表
    p_arr = []
    for i, vehicle in enumerate(p):
        front_zero = np.zeros((arrival_time_step[i] - 1), dtype=int)
        after_zero = np.zeros((col - departure_time_step[i]), dtype=int)
        p_arr.append(np.concatenate((front_zero, vehicle, after_zero), axis=0))

    return np.array(p_arr, dtype=int)

def Full_Part(x_arr):
    '''
    输入是一个完整个体(矩阵)
    函数将一个完整矩阵切割为不完整的列表 只有 [ta, td] 时间段
    返回一个不完整的列表
    '''
    x = []
    for i, vehicle in enumerate(x_arr):
        x.append(vehicle[arrival_time_step[i] - 1 : departure_time_step[i]])

    return x

def Change(x, y):
    '''
    输入两个数
    函数可以交换两个数的值
    返回交换后的两个数
    '''

    return (y, x)

def mutation(x):
    """
    输入 x 是一个个体，是不完整的列表个体
    函数根据变异率交换两个位置的置，即可保证一行中 k 不变，不破坏SOC约束
    返回一个变异后的个体
    """
    for i, row in enumerate(x):
        # 按行进行变异处理
        lgth = len(row)
        for j in range(lgth):
            # 逐个时间步尝试变异
            m = random.random()
            if m < probability:
                # 交换 j 时间步和随机某个时间步 temp_index 的值
                temp_index = np.random.randint(lgth)
                row[j], row[temp_index] = Change(row[j], row[temp_index])

    return x

def Get_Col(x, j):
    """ 
    输入个体 x(完整矩阵), 和需要得到的列 j
    函数得到第 j 列的在 [ta, td] 内的行号，并以 A B C 三相区分
    返回一个字典 
    """
    column = {'A':[], 'B':[], 'C':[]}
    for i, row in enumerate(x):
        if arrival_time_step[i] <= j + 1 <= departure_time_step[i]:
            if Φ[i] == 'A':
                column['A'].append(i)
            elif Φ[i] == 'B':
                column['B'].append(i)
            else:
                column['C'].append(i)

    return column

def Distinguish_phase():
    '''
    不需要输入
    函数可以区分所有车在哪一相充电 # Get a array to distinguish phase A,B,C
    返回一个字典，类似于 Get_Col 
    '''
    phase_dict = {'A':[], 'B':[], 'C':[]}
    for vehicle in range(row):
        # Get three phaseA,B,C load
        if Φ[vehicle] == 'A':
            phase_dict['A'].append(vehicle)
        elif Φ[vehicle] == 'B':
            phase_dict['B'].append(vehicle)
        else:
            phase_dict['C'].append(vehicle)
    
    return phase_dict

def Calculate_overload(x):
    '''
    输入一个个体(不完整的列表)
    函数是为 Repair_violate 中 Calculate_V 计算违反最大功率约束设计
    返回所有时刻最大功率违反度
    '''
    # 先将 x 补全为 row * col 矩阵
    x_arr = Part_Full(x)
    power_col = x_arr * np.expand_dims(power, axis=1)
    used_power = np.sum(power_col, axis = 0)
    # 计算每个时刻的超过负载量 overload
    overload = restriction_in_power - used_power

    return overload

def Calculate_powerABC(x):
    '''
    输入一个个体(不完整的列表)
    函数是为 Repair_violate 中 Calculate_V 计算 A B C 三相总负载设计
    返回一个 3 * col 的矩阵
    '''
    x_arr = Part_Full(x)
    phase_dict = Distinguish_phase()
    # A B C 三相的总功率(带基础负载的)
    power_A = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['A']),
                                 dtype='(96,)f')) + phase_base_load[0]
    power_B = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['B']),
                                 dtype='(96,)f')) + phase_base_load[1]
    power_C = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['C']),
                                 dtype='(96,)f')) + phase_base_load[2]
    power_ABC = np.array([power_A, power_B, power_C])

    return power_ABC


def Calculate_V(x):
    '''
    输入不完整个体列表
    函数是为 Repair_violate 计算约束违反度设计的
    返回个体的约束违反度
    '''
    # 计算每个时刻违反最大负载约束的功率
    load_overrun = Calculate_overload(x)
    # 获得 A B C 三相每个时刻的总功率
    power_ABC = Calculate_powerABC(solution)
    # 计算每个时刻的三相不平衡值 
    imbalance = 3 * (np.max(power_ABC, axis=0) - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    # 由于最大负载约束和三相不平衡约束的违反值相差过大，需要标准化
    violation = (np.sum(Normalize_0_1(load_overrun[load_overrun < 0])) 
                  + np.sum(Normalize_0_1(imbalance[imbalance > max_imbalance_limit])))

    return violation

def Repair_violate(x):
    '''
    输入不完整个体列表
    这是 V8 的修复函数 最大功率上限和三相不平衡都在这里完成
    返回修复好的不完整个体列表
    '''


    return x

def Normalize_0_1(data):
    """
    输入是一个一维矩阵
    函数可以将数据线性缩放到 [0, 1] 的范围
    输出是一个范围 [0, 1] 的一维矩阵
    """
    # 需要考虑 data 为空的情况
    # 有可能约束完全满足，不存在约束惩罚项
    # 需要考虑 max_val 和 min_val 相等的情况
    if np.size(data) == 0:
        return data

    min_val = np.min(data)
    max_val = np.max(data)

    if max_val == min_val:
        return np.ones_like(data)

    scaled_data = (data - min_val) / (max_val - min_val)

    return scaled_data

def F(solution):
    '''
    输入一个个体(不完整的列表)
    函数计算个体的适应度值 # Calculate the objective value of the solution
    返回个体适应度值
    '''
    # 将个体补全为 row * col 的矩阵
    x_arr = Part_Full(solution)
    # 计算用电表 (row * col)
    used_power = x_arr * np.expand_dims(power, axis=1)
    # 计算电费表 (row * col)
    charging_cost = used_power * np.expand_dims(ρ, axis=0) * 0.25
    # 计算每个时刻违反最大负载约束的功率
    load_overrun = np.sum(used_power, axis=0) - restriction_in_power
    # 获得 A B C 三相每个时刻的总功率
    power_ABC = Calculate_powerABC(solution)
    # 计算每个时刻的三相不平衡值
    imbalance = 3 * (np.max(power_ABC, axis=0) - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    # 由于最大负载约束和三相不平衡约束的违反值相差过大，需要标准化
    punishment = (np.sum(Normalize_0_1(load_overrun[load_overrun > 0])) 
                  + np.sum(Normalize_0_1(imbalance[imbalance > max_imbalance_limit])))
    
    return np.sum(charging_cost) + R * punishment

def Plot_circuit_load(x):
    x_arr = Part_Full(x)
    step = np.arange(1, col + 1)
    limit_power = np.full((col), 2200 * times)
    used_power = np.sum(x_arr * np.expand_dims(power, axis=1), axis=0)
    total_power = base_load + used_power
    plt.title("Circuit load")
    plt.ylim((0,2400 * times))
    plt.plot(step, used_power, label = 'Charging load')
    plt.plot(step, total_power, label = 'Total load')
    plt.plot(step, base_load, label = 'Basic load')
    plt.plot(step, limit_power, label = 'Power limitation')
    plt.legend(["Charging load","Total load","Basic load","Power limitation"],loc='upper right',fontsize='x-small')
    plt.xlabel("step")
    plt.ylabel("power/kw")

def Plot_phase_imbalance(x):
    x_arr = Part_Full(x)
    step = np.arange(1, col + 1)
    limit_imbalance = np.full((col), max_imbalance_limit)
    # 获得 A B C 三相每个时刻的总功率
    power_ABC = Calculate_powerABC(x)
    # 计算每个时刻的三相不平衡值
    imbalance = 3 * (np.max(power_ABC, axis=0) - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    plt.title("Phase imbalance")
    plt.plot(step, limit_imbalance, color = 'r', label = 'limit imbalance')
    plt.plot(step, imbalance, marker = '.', color = 'b', label = 'imbalance')
    plt.legend(["limit imbalance", "imbalance"],loc='upper right',fontsize='x-small')

def Plot_time_constraint(x):
    # 获取矩阵中数值为1的元素的坐标
    x_arr = Part_Full(x)
    rows, cols = np.where(x_arr.T == 1)
    cols += 1
    rows += 1
    # 绘制散点图
    plt.title("Time constraint")
    plt.xlim((0, row + 1))
    plt.ylim((0, col))
    plt.scatter(cols, rows, c='red', marker='o', s = 3)

    # 画出时间上下限
    vehicle = np.arange(1, row + 1)
    plt.plot(vehicle, arrival_time_step, color = 'b', marker = '.')
    plt.plot(vehicle, departure_time_step, color = 'b', marker = '.')

def Plot_Evolution_curve(f_log):
    plt.title("Evolution curve")
    plt.plot(f_log)

def Plot_SOC(x):
    step = np.arange(1, row + 1)
    x_arr = Part_Full(x)
    x_k = np.sum(x_arr, axis=1)
    k = Calculate_k()
    plt.scatter(step, x_k, c = 'red', marker = '*', label = 'SOC')
    plt.scatter(step, k, c = 'blue', marker='o', label = 'sd')
    # 添加栅格，控制栅格数量
    plt.grid(True, which='both', axis='both', linewidth=0.5)
    plt.legend(['SOC_k', 'sd_k'])
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

solution, f_log = PILP_algorithm(n)

end = time.time()

Plot(solution, f_log)
Plot_SOC(solution)