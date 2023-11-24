# GeneticAlgorithm
Optimal ordered charging algorithm for electric vehicles based on integer programming

 NewPyomo.py -- 利用Gurobi求解，数据是CSV的不是jld2的；

 GeneticAlgorithm.py -- 最原始的版本； 5车24小时，初始化和迭代都有repair0； repair2是'!='；

 NewGeneticAlgorithm.py -- 从这一版开始使用CSV的数据； 100车24小时，只有初始化有repair0； repair2是'!='； 有变异；

 NewGeneticAlgorithmV2.py -- 这是遗传算法的版本V2 因为原版存在计算误差repair2无法跳出while循环的情况，在V2中做出修改； repair2是'while total_increment[i] < max_SOC[i]:'；

 NewGeneticAlgorithmV3.py -- 这是遗传算法的版本V3 因为V2repair3中三相不平衡修复存在死循环，还没有修复好；

 这之前的算法都没有解决100辆车的求解。

 现在开始新的方法，将三相不平衡加入适应度值的求解中，不在作为约束存在，降低难度。
