import random
import numpy as np
import matplotlib.pyplot as plt

'''
这是对之前简化算法写过的函数的总结
有遗传算法的版本V7 有、无低电价引导的两个进化版本
def ***No() 代表这是无低电价引导的版本
'''

''' power--每辆车的充电功率;
    efficiency--每辆车的充电效率
    capacity--每辆车的电池容量
    restriction_power--limit_power - base_load
    Φ--充电象限
    ρ--电价
    R--惩罚系数
    n--种群数量
    probability--变异概率'''

def Calculate_k():
    '''
    函数是为计算计算每辆车需要充几次电设计的
    为电池SOC约束做准备, 计算每辆车15min的充电增量
    Returns:
        arr: 每辆车需要充电的次数 也就是 1 的个数
    '''
    unit_increment = power * efficiency * 0.25 / capacity #每辆车15min的单位增量
    k = (sd - sa) // unit_increment

    return k.astype(int)

def Custom_Initialization(n1, k):
    """ 
    Initialize n solutions
    Args:
        n: 种群数量
        k: 每辆车(行)的期望 k 值(SOC)
    Returns:
        tuple: 一个种群列表 list(list(array())) P 和 适应度值列表 Fitness
    """
    P = []
    Fitness = []
    # 第 p 个个体
    for p in range(n1):
        # 第 n 个个体
        v = []
        # 第 i 辆车
        for i in range(row):
            # 产生一个 ta 到 td 时间段的随机数矩阵
            while True:
                s = np.random.rand(td[i] - ta[i] + 1)
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
    shared_fitness = Calculate_shared_fitness(P, Fitness, 7000 * times)
    print(Diversity(P))

    return (P, shared_fitness)

def Updata_Best(Fitness):
    '''
    找到适应度值最小的个体
    Args:
        Fitness(list): 一个种群的适应度值列表
    Returns:
        int: 适应度值最小的个体的在种群 P 中的索引值
    '''
    best_k = np.argmin(Fitness)
    
    return best_k

def tournament_Selection(x1, x2, Fitness):
    '''
    返回适应度值更优的个体
    Perform tournament selection and return the better solution
    Args:
        x1, x2: 两个个体在种群中的索引值
        Fitness: 适应度值列表
    Returns:
        int: 更优的个体索引值
    '''
    
    return x1 if Fitness[x1] < Fitness[x2] else x2

def Custom_Recombination(parent1, parent2):
    '''
    Perform custom recombination to create a new solution
    Args:
        parent1, parent2(list): 两个父代个体(不完整的列表)
    Returns:
        list: 一个子代(不完整的列表)
    '''
    parent1_arr = Part_Full(parent1)
    parent2_arr = Part_Full(parent2)
    charging_col_parent1 = np.sum(np.expand_dims(power, axis=1)
                                   * parent1_arr * np.expand_dims(ρ, axis=0), axis=1)
    charging_col_parent2 = np.sum(np.expand_dims(power, axis=1)
                                   * parent2_arr * np.expand_dims(ρ, axis=0), axis=1)
    
    # offspring 是新产生的子代 
    offspring = []
    row = len(parent1)

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
    函数将一个个体由不完整，即只有[ta, td]时间段的矩阵，补全为 row * col 的矩阵
    Args:
        p(list): 一个个体
    Returns:
        arr: row * col 矩阵
    """
    # 新建补全的矩阵，不然会改变原来的不完整列表
    p_arr = []
    for i, vehicle in enumerate(p):
        front_zero = np.zeros((ta[i] - 1), dtype=int)
        after_zero = np.zeros((col - td[i]), dtype=int)
        p_arr.append(np.concatenate((front_zero, vehicle, after_zero), axis=0))

    return np.array(p_arr, dtype=int)

def Full_Part(x_arr):
    '''
    函数将一个完整矩阵切割为不完整的列表 只有 [ta, td] 时间段
    Args:
        x_arr(arr): 一个个体
    Returns:
        list: 一个不完整的个体
    '''
    x = []
    for i, vehicle in enumerate(x_arr):
        x.append(vehicle[ta[i] - 1 : td[i]])

    return x

def Change(x, y):
    '''
    函数可以交换两个数的值
    '''

    return (y, x)

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

def mutationNo(x):
    """
    这是无低电价引导版本的变异函数
    函数根据变异率交换两个位置的置, 即可保证一行中 k 不变, 不破坏SOC约束
    Args:
        x(list): 一个个体
    Returns:
        list: 一个变异后的个体
    """
    for i, row in enumerate(x):
        # 按行进行变异处理
        lgth = len(row)
        for j in range(lgth):
            # 逐个时间步尝试变异
            m = random.random()
            if m < mutation_rate:
                # 交换 j 时间步和随机某个时间步 temp_index 的值
                temp_index = np.random.randint(lgth)
                row[j], row[temp_index] = Change(row[j], row[temp_index])

    return x

def mutation(x):
    """
    这是有低电价引导版本的变异函数
    函数根据变异率交换两个位置的置, 即可保证一行中 k 不变, 不破坏SOC约束
    Args:
        x(list): 一个个体
        probability: 根据电价映射的选择概率
    Returns:
        list: 一个变异后的个体
    """
    for i, row in enumerate(x):
        # 按行进行变异处理
        pro = np.copy(probability[ta[i] - 1 : td[i]])
        # 将 0 1 分为两组
        col0 = np.where(row == 0)[0]
        col1 = np.where(row == 1)[0]

        # 对 1 组变异 1 的数总是比 0 多 高电价高变异率
        # 存在所有时刻都需要充电的情况 这时没有 0 组 直接不变异就好
        if col0.size > 0:
            for j in col1:
                m = random.random()
                if m < mutation_rate:
                    # 在 0 组中选择另一个数组成变换对 低电价高变异率prolow
                    prolow = (1 - pro[col0])
                    # 概率和为 1
                    p = prolow / np.sum(prolow)
                    obj = np.random.choice(col0, p=p)
                    row[j], row[obj] = Change(row[j], row[obj])

    return x

def Get_Col(x, j):
    """ 
    函数得到第 j 列的在 [ta, td] 内的行号，并以 A B C 三相区分
    Args:
        x(arr): 个体
        j(int): 要找的列号
    Returns:
        dict: 一个字典 
    """
    column = {'A':[], 'B':[], 'C':[]}
    for i, row in enumerate(x):
        if ta[i] <= j + 1 <= td[i]:
            if Φ[i] == 'A':
                column['A'].append(i)
            elif Φ[i] == 'B':
                column['B'].append(i)
            else:
                column['C'].append(i)

    return column

def Distinguish_phase():
    '''
    函数可以区分所有车在哪一相充电 # Get a array to distinguish phase A,B,C
    Returns:
        一个字典，类似于 Get_Col 
    '''
    phase_dict = {'A':[], 'B':[], 'C':[]}
    for vehicle in range(Φ.size):
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
    函数是为 Repair_load 计算多有时刻违反最大功率约束设计
    Args:
        x(list): 一个个体
    Returns:
        arr: 所有时刻最大功率违反度
    '''
    # 先将 x 补全为 row * col 矩阵
    x_arr = Part_Full(x)
    power_col = x_arr * np.expand_dims(power, axis=1)
    used_power = np.sum(power_col, axis = 0)
    # 计算每个时刻的超过负载量 overload
    overload = restriction_power - used_power

    return overload

def Repair_LoadNo(x):
    """
    这是没有低电价引导的版本
    函数对每个时间步的最大功率上限进行修复
    Args:
        x(list): 一个个体
    Returns:
        list: 修复好的个体
    """
    # 计算每个时刻的超过负载量 overload
    overload = Calculate_overload(x)
    # 获得 minID 所有时刻中违反功率上限最严重的时刻
    minID = np.argmin(overload)
    while overload[minID] < 0:
        x_arr = Part_Full(x)
        row1 = []
        for i, row in enumerate(x_arr):
            if ta[i] <= minID + 1 <= td[i] and row[minID] == 1:
                row1.append(i)
        # 在满足时间约束且值为1的行号里随机选择一行 index_row
        index_row = np.random.choice(row1)
        # 获得所有时刻中满足最大功率约束的时刻 satisload
        satisload = np.where(overload[ta[index_row] - 1:
                                      td[index_row]] >= power[index_row])[0]
        # 在满足最大功率约束的时刻中随机选择一列 index_col 同时需要减去 ta
        index_col = np.random.choice(satisload)
        # 交换数值(可以保证 index_row k 值不变)后重新计算 overload
        x[index_row][minID - ta[index_row] + 1], x[index_row][index_col] = Change(
            x[index_row][minID - ta[index_row] + 1], x[index_row][index_col])
        # 计算每个时刻的超过负载量 overload
        overload = Calculate_overload(x)
        # 获得 minID 所有时刻中违反功率上限最严重的时刻
        minID = np.argmin(overload)
            
    return x

def Search_row1(x, j):
    '''
    输入一个个体(不完整的列表) 需要寻找的列
    函数是为 Repair_Load 寻找 minID 列符合时间约束和 x 值为1的行
    返回符合条件的行号矩阵
    '''
    x_arr = Part_Full(x)
    row1 = []
    for i, row in enumerate(x_arr):
        if ta[i] <= j + 1 <= td[i] and row[j] == 1:
            row1.append(i)

    return row1

def Repair_Load(x):
    """
    输入一个个体(不完整的列表)
    函数对每个时间步的最大功率上限进行修复
    返回修复好的个体
    """
    # 计算每个时刻的超过负载量 overload
    overload = Calculate_overload(x)
    # 获得 minID 所有时刻中违反功率上限最严重的时刻
    minID = np.argmin(overload)
    while overload[minID] < 0:
        row1 = Search_row1(x, minID)
        # 在满足时间约束且值为1的行号里随机选择一行 index_row
        index_row = np.random.choice(row1)
        # 获得所有时刻中满足最大功率约束的时刻 satisload
        satisload = np.where(overload[ta[index_row] - 1:
                                      td[index_row]] >= power[index_row])[0]
        # 在满足最大功率约束的时刻中选择一列 index_col 同时需要减去 ta
        # 选择列时 低电价高概率
        pro = np.copy(probability[ta[index_row] - 1
                                   : td[index_row]])
        prolow = (1 - pro[satisload])
        p = prolow / np.sum(prolow)
        index_col = np.random.choice(satisload, p=p)
        # 交换数值(可以保证 index_row k 值不变)后重新计算 overload
        x[index_row][minID - ta[index_row] + 1], x[index_row][index_col] = Change(
            x[index_row][minID - ta[index_row] + 1], x[index_row][index_col])
        # 计算每个时刻的超过负载量 overload
        overload = Calculate_overload(x)
        # 获得 minID 所有时刻中违反功率上限最严重的时刻
        minID = np.argmin(overload)
            
    return x

def Calculate_powerABC(x,):
    '''
    函数是为 Repair_imbalance 计算 A B C 三相总负载设计
    Args:
        x(list): 一个个体
        phase_base_load(arr): 每个象限的基础负载 (3, col)
    Returns:
        arr: 一个 (3, col) 的矩阵
    '''
    x_arr = Part_Full(x)
    phase_dict = Distinguish_phase()
    # A B C 三相的总功率(带基础负载的)
    power_A = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['A']),
                                 dtype='(96,)f'), axis=0) + phase_base_load[0]
    power_B = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['B']),
                                 dtype='(96,)f'), axis=0) + phase_base_load[1]
    power_C = np.sum(np.fromiter((x_arr[i] * power[i] for i in phase_dict['C']),
                                 dtype='(96,)f'), axis=0) + phase_base_load[2]
    power_ABC = np.array([power_A, power_B, power_C])

    return power_ABC

def Search_imbalancecolNo(x, index_row):
    '''
    无电价引导版本
    函数是为 Repair_Imbalance 寻找满足条件的时刻
    1. 是 maxphase 象限的车
    2. 不破坏最大功率约束
    3. 最好找到的时刻此象限正好是 minphase
    Args:
        x(list): 一个个体
        index_row: 最大象限选择的行
    Returns:
        int: 满足条件车的行号(是不完整列表的行号)
    '''
    overload = Calculate_overload(x)
    indexrowoverload = overload[ta[index_row] - 1 : td[index_row]]
    # index_row 行满足功率上限的时刻
    satisload = np.where(indexrowoverload > power[index_row])[0]
    # satisload 中值为0的时刻
    satiscol0 = np.where(x[index_row] == 0)[0]
    satisrow = np.intersect1d(satisload, satiscol0)
    if satisrow.size == 0:
        return False
    else:
        return np.random.choice(satisrow)

def Search_imbalancerow(x, column, maxphase_row, maxID):
    '''
    这个函数不区分有无电价引导
    函数是为了最小象限找合适的行设计 需要满足 1. 不破坏最大功率上限 2. 是最小象限的行 3. 值为0
    Args:
        x(list): 个体
        column: 最小象限的所有行
        maxphase_row: 最大象限挑出的行
        maxID: 三相不平衡最大的一列
    Returns:
        int: 一个行号
    '''
    x_arr = Part_Full(x)
    # mixID 中所有是 0 的行
    satisrow0 = np.where(x_arr[:, maxID] == 0)[0]
    # 最小象限中是 0 的行
    satisrowminphase0 = np.intersect1d(satisrow0, column)
    overload = Calculate_overload(x)
    # 计算将 0 变为 1 后不破坏最大功率上限的行
    overloadrow =  overload[maxID] + power[maxphase_row] - power
    satisloadrow = np.where(overloadrow >= 0)[0]
    satisrow = np.intersect1d(satisrowminphase0, satisloadrow)
    if satisrow.size == 0:
        return False
    else:
        return np.random.choice(satisrow)

def Repair_ImbalanceNo(x):
    '''
    输入一个个体(不完整的列表)
    函数对每个时间步的三相不平衡进行修复
    返回一个修复好的个体
    '''
    # 计算所有时刻的三相不平衡
    power_ABC = Calculate_powerABC(x)
    imbalance = 3 * (np.max(power_ABC, axis=0)
                      - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    maxID = np.argmax(imbalance)
    while imbalance[maxID] >= max_imbalance_limit:
        # maxphase 为 maxID 时刻功率最大的象限
        phase = ['A', 'B', 'C']
        maxphase = phase[np.argmax(power_ABC[:, maxID])]
        minphase = phase[np.argmin(power_ABC[:, maxID])]
        # 获得 maxID 列的 A B C 三相行号字典 column
        column = Get_Col(x, maxID)
        x_arr = Part_Full(x)
        # 最大象限且值为1的行号矩阵
        satisrow1 = np.where(x_arr[:, maxID] == 1)[0]
        maxphase_arr = np.intersect1d(satisrow1, column[maxphase])
        # 随机选择最大象限且值为1一行 maxphase_row
        maxphase_row = np.random.choice(maxphase_arr)
        # 寻找 maxphase_row 行满足条件的列
        maxphase_col = Search_imbalancecolNo(x, maxphase_row)
        # 随机选择最小象限且值为0一行 minphase_row
        minphase_row = Search_imbalancerow(x, column[minphase], maxphase_row, maxID)
        # 随机选择 minphase_row 行中的一列值为1 minphase_col
        satisminphaserow1 = np.where(x[minphase_row] == 1)[0]
        minphase_col = np.random.choice(satisminphaserow1)
        # 交换最大象限
        if maxphase_col != False:
            x[maxphase_row][maxID - ta[maxphase_row] + 1], x[maxphase_row][maxphase_col] = Change(
                x[maxphase_row][maxID - ta[maxphase_row] + 1], x[maxphase_row][maxphase_col]
            )

        # 交换最小象限
        if minphase_row != False:
            x[minphase_row][maxID - ta[minphase_row] + 1], x[minphase_row][minphase_col] = Change(
                x[minphase_row][maxID - ta[minphase_row] + 1], x[minphase_row][minphase_col]
            )

        # 如果最大和最小象限都没有用找到合适的交换对，直接返回未修复好的个体
        if maxphase_col == False and minphase_row == False:
            return x
        
        power_ABC = Calculate_powerABC(x)
        imbalance = 3 * (np.max(power_ABC, axis=0)
                            - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
        maxID = np.argmax(imbalance)

    return x

def Search_imbalancecol(x, index_row):
    '''
    有电价引导版本
    函数是为 Repair_Imbalance 寻找满足条件的时刻
    1. 是 maxphase 象限的车
    2. 不破坏最大功率约束
    3. 最好找到的时刻此象限正好是 minphase
    Args:
        x(list): 一个个体
        index_row: 最大象限选择的行
    Returns:
        int: 满足条件车的行号(是不完整列表的行号)
    '''
    overload = Calculate_overload(x)
    indexrowoverload = overload[ta[index_row] - 1 : td[index_row]]
    # index_row 行满足功率上限的时刻
    satisload = np.where(indexrowoverload > power[index_row])[0]
    # satisload 中值为0的时刻
    satiscol0 = np.where(x[index_row] == 0)[0]
    satisrow = np.intersect1d(satisload, satiscol0)
    if satisrow.size == 0:
        return False
    else:
        # 选择列时 低电价高概率
        pro = np.copy(probability[ta[index_row] - 1 : td[index_row]])
        prolow = (1 - pro[satisrow])
        p = prolow / np.sum(prolow)
        return np.random.choice(satisrow, p=p)

def Repair_Imbalance(x):
    '''
    输入一个个体(不完整的列表)
    函数对每个时间步的三相不平衡进行修复
    返回一个修复好的个体
    '''
    # 计算所有时刻的三相不平衡
    power_ABC = Calculate_powerABC(x)
    imbalance = 3 * (np.max(power_ABC, axis=0)
                      - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    maxID = np.argmax(imbalance)
    while imbalance[maxID] >= max_imbalance_limit:
        # maxphase 为 maxID 时刻功率最大的象限
        phase = ['A', 'B', 'C']
        maxphase = phase[np.argmax(power_ABC[:, maxID])]
        minphase = phase[np.argmin(power_ABC[:, maxID])]
        # 获得 maxID 列的 A B C 三相行号字典 column
        column = Get_Col(x, maxID)
        x_arr = Part_Full(x)
        # 最大象限且值为1的行号矩阵
        satisrow1 = np.where(x_arr[:, maxID] == 1)[0]
        maxphase_arr = np.intersect1d(satisrow1, column[maxphase])
        # 随机选择最大象限且值为1一行 maxphase_row
        maxphase_row = np.random.choice(maxphase_arr)
        # 寻找 maxphase_row 行满足条件的列
        maxphase_col = Search_imbalancecol(x, maxphase_row)
        # 随机选择最小象限且值为0一行 minphase_row
        minphase_row = Search_imbalancerow(x, column[minphase], maxphase_row, maxID)
        
        # 交换最大象限
        if maxphase_col != False:
            x[maxphase_row][maxID - ta[maxphase_row] + 1], x[maxphase_row][maxphase_col] = Change(
                x[maxphase_row][maxID - ta[maxphase_row] + 1], x[maxphase_row][maxphase_col]
            )

        # 交换最小象限
        if minphase_row != False:
            # 选择 minphase_row 行中的一列值为1 minphase_col 高电价高概率
            satisminphaserow1 = np.where(x[minphase_row] == 1)[0]
            pro = np.copy(probability[ta[minphase_row] - 1 : td[minphase_row]])
            prolow = pro[satisminphaserow1]
            p = prolow / np.sum(prolow)
            minphase_col = np.random.choice(satisminphaserow1, p=p)
            x[minphase_row][maxID - ta[minphase_row] + 1], x[minphase_row][minphase_col] = Change(
                x[minphase_row][maxID - ta[minphase_row] + 1], x[minphase_row][minphase_col]
            )

        # 如果最大和最小象限都没有用找到合适的交换对，直接返回未修复好的个体
        if maxphase_col == False and minphase_row == False:
            return x
        
        power_ABC = Calculate_powerABC(x)
        imbalance = 3 * (np.max(power_ABC, axis=0)
                            - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
        maxID = np.argmax(imbalance)

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
    Calculate the objective value of the solution
    Args:
        solution(list): 一个个体
        R : 惩罚系数
    Returns:
        个体适应度值
    '''
    # 将个体补全为 row * col 的矩阵
    x_arr = Part_Full(solution)
    # 计算用电表 (row * col)
    used_power = x_arr * np.expand_dims(power, axis=1)
    # 计算电费表 (row * col)
    charging_cost = used_power * np.expand_dims(ρ, axis=0) * 0.25
    # 计算每个时刻违反最大负载约束的功率
    load_overrun = np.sum(used_power, axis=0) - restriction_power
    # 获得 A B C 三相每个时刻的总功率
    power_ABC = Calculate_powerABC(solution)
    # 计算每个时刻的三相不平衡值
    imbalance = 3 * (np.max(power_ABC, axis=0) - np.min(power_ABC, axis=0)) / np.sum(power_ABC, axis=0)
    # 由于最大负载约束和三相不平衡约束的违反值相差过大，需要标准化
    punishment = (np.sum(Normalize_0_1(load_overrun[load_overrun > 0])) 
                  + np.sum(Normalize_0_1(imbalance[imbalance > max_imbalance_limit])))
    
    return np.sum(charging_cost) + R * punishment

def Local_Search(x, F_best):
    '''
    输入一个不完整个体，这个个体是种群中最优个体 和这个个体的适应度值
    函数是为迭代过程中出现不进化情况对最优个体局部搜索设计
    返回局部搜索结果
    '''
    overload = Calculate_overload(x)
    for i, row in enumerate(x):
        ioverload = np.copy(overload[ta[i] - 1 : td[i]])
        satisrow = np.where(ioverload > power[i])[0]
        if satisrow.size == 0:
            continue
        # 将 0 1 分为两组
        col0 = np.where(row[satisrow] == 0)[0]
        col1 = np.where(row[satisrow] == 1)[0]
        for i0 in col0:
            # lowcol1 是比 i0 电价低的 col1 中的时刻
            lowcol1 = col1[col1 > i0]
            for j in lowcol1:
                row[i0], row[j] = Change(row[i0], row[j])
                if F(x) < F_best:
                    # 如果改变后适应度值比之前更好就返回这个更好的个体
                    return x
                else:
                    # 如果改变后适应度值没有更好就改回来 不做改变
                    row[i0], row[j] = Change(row[i0], row[j])
    return x                

def hammingDistance(x, y):
    '''
    输入两个不完整个体列表
    计算汉明距离，帮助 Diversity 函数确定种群多样性
    返回两个个体的汉明距离
    '''
    x_arr = Part_Full(x)
    y_arr = Part_Full(y)
    x = np.ravel(x_arr)
    y = np.ravel(y_arr)
    xor = x ^ y
    distance = 0
    # 每次右移，最左边都会补零，因此截止条件是xor已经是一个零值了
    for i in range(len(xor)):
        if xor[i] & 1:
            distance = distance + 1

    return distance

def Diversity(P):
    '''
    输入一个种群 P
    函数是为了计算整个种群的多样性
    返回种群多样性
    '''
    distance = []
    for i in P:
        for j in P:
            distance.append(hammingDistance(i, j))
    return np.sum(distance) / 2

def Calculate_shared_fitness(P, Fitness, sigma_share):
    """计算每个个体的共享适应度"""
    shared_fitness = np.zeros(len(P))
    for i, ind in enumerate(P):
        sumsh = 0
        for j, other_ind in enumerate(P):
            if i != j:
                distance = hammingDistance(ind, other_ind)
                if distance < sigma_share:
                    sumsh += (distance / sigma_share)
        if sumsh == 0:
            shared_fitness[i] = 1
        if sumsh != 0:
            shared_fitness[i] = sumsh
    
    return Fitness / shared_fitness

def Plot_circuit_load(x):
    x_arr = Part_Full(x)
    step = np.arange(1, col + 1)
    limit_power = np.full((col), max_power_limit)
    used_power = np.sum(x_arr * np.expand_dims(power, axis=1), axis=0)
    total_power = base_load + used_power
    plt.title("Circuit load")
    plt.ylim((0, max_power_limit))
    plt.plot(step, used_power, label = 'Charging load')
    plt.plot(step, total_power, label = 'Total load')
    plt.plot(step, base_load, label = 'Basic load')
    plt.plot(step, limit_power, label = 'Power limitation')
    plt.legend(["Charging load","Total load","Basic load","Power limitation"],loc='upper right',fontsize='x-small')
    plt.xlabel("step")
    plt.ylabel("power/kw")

def Plot_phase_imbalance(x):
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
    row, col = x_arr.shape
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
    plt.plot(vehicle, ta, color = 'b', marker = '.')
    plt.plot(vehicle, td, color = 'b', marker = '.')

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
mutation_rate = 0.4
# 价格映射
probability = Price_Probability(ρ, 0.1, 0.9)
# 三相不平衡系数
α = 10

# 最大三相不平衡度
max_imbalance_limit = 0.04

# 最大功率上限
max_power_limit = 2200 * times

# 获得基础负载和电路限制restriction_power
base_load = np.sum(phase_base_load, axis=0)
restriction_power = max_power_limit - base_load