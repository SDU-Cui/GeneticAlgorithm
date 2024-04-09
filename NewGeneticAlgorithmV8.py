import random
import NewGeneticAlgorithmV7Function as ga
import NewGeneticAlgorithmReadcsv as var

'''
这是遗传算法的版本 V8 因为 V7 存在多样性不足过快收敛的问题
V8 尝试将种群分为两组 一组加低电价引导 一组不加引导 分别进化
'''

k = ga.Calculate_k()
P, Fitness = ga.Custom_Initialization(var.n, k)
k_best = ga.Updata_Best(Fitness)
# B 记录最优个体
B = P[k_best]
# 用来记录每一代中最优的适应度值个体
f_log = []
f_log.append(Fitness[k_best])
t = 0
print(t, f_log[-1], ga.F(B))
err = 1

while err >= 1e-10:
    # Q 为下一代种群
    Q = [P[k_best]]
    # Q_Fitness 为下一代种群的适应度值列表
    Q_Fitness = [Fitness[k_best]]
    while len(Q) < var.n:
        ID1, ID2, ID3, ID4 = random.sample(range(var.n), 4)
        parent1 = P[ga.tournament_Selection(ID1, ID2, Fitness)]
        parent2 = P[ga.tournament_Selection(ID3, ID4, Fitness)]
        offspring = ga.Custom_Recombination(parent1, parent2)
        x1 = ga.mutation(offspring)
        x2 = ga.Repair_Load(x1)
        x3 =  ga.Repair_Imbalance(x2)
        fitness = ga.F(x3)
        Q_Fitness.append(fitness)
        Q.append(x3)
        
    print(ga.Diversity(Q))
    
    k_best = ga.Updata_Best(Q_Fitness)
    B = Q[k_best]
    f_log.append(Q_Fitness[k_best])
    if len(f_log) < 4:
        err = 1
    else:
        if f_log[-1] == f_log[-2]:
            # 当出现不再进化的情况，对最优解局部搜索
            Q[k_best] = ga.Local_Search(B, Q_Fitness[k_best])
            B = Q[k_best]
            Q_Fitness[k_best] = ga.F(B)
            f_log[-1] = Q_Fitness[k_best]
        differences = [abs(f_log[-1] - f_log[-2]), 
            abs(f_log[-2] - f_log[-3]), 
            abs(f_log[-3] - f_log[-4])]
        err = sum(differences)
    
    P = Q
    Fitness = Q_Fitness
    t += 1
    print(t, f_log[-1], ga.F(B))

ga.Plot(B, f_log)
ga.Plot_SOC(B)