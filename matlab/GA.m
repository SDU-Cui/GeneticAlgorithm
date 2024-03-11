clear;
clc;
row = 9;
col = 96;
tasks = importdata("D:\MyProject\Charge_Pyomo\GeneticAlgorithm\data\vehicle_data_9.csv");
phase_base_load = importdata("D:\MyProject\Charge_Pyomo\GeneticAlgorithm\data\phase_base_load.csv");
ta = cell2double(tasks.textdata(2:end, 2)); %到达时间
td = cell2double(tasks.textdata(2:end, 3)); %离开时间
sa = cell2double(tasks.textdata(2:end, 4)); %到达SOC
sd = cell2double(tasks.textdata(2:end, 5)); %离开SOC
E = cell2double(tasks.textdata(2:end, 6));  %电池容量
P = cell2double(tasks.textdata(2:end, 7));  %充电功率
eta = cell2double(tasks.textdata(2:end, 8));%充电效率
phi = tasks.textdata(2:end, 9);             %充电象限
delta = tasks.data(:, 1);                %开关控制一次充电SOC
k = tasks.data(:, 2);                    %需要充电次数

% 为电池SOC约束做准备，计算每辆车15min的充电增量
unit_increment = P .* eta .* 0.25 ./ E; %每辆车15min的单位增量,(col,)
% 获得基础负载和电路限制get_restriction_in_power
base_load = sum(phase_base_load, 1); %对于二维矩阵来讲，1是按列相加，2是按行相加
restriction_in_power = (2200 - base_load);
% 最大三相不平衡度
max_imbalance_limit = 0.04;
% 电价
price = ones(1, col);
price(1, 30:end) = 0.1;

% 设置变量 x
x = optimvar("x", row, col, "Type", "integer", "LowerBound", 0, "UpperBound", 1);

% 创建优化问题 model
model = optimproblem("ObjectiveSense", "minimize");

% 设置优化目标
model.Objective = F(x, price, P);

% 时间约束
model.Constraints.cons1 = fix_model_value(model, ta, td);
% SOC约束
model.Constraints.cons2 = energy_constraint_rule(model, sa, sd, unit_increment);
% 功率上限约束
model.Constraints.cons3 = power_constraint_rule(model, P, restriction_in_power);
% 三相不平衡约束
model.Constraints.cons4 = phase_upper_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit);
model.Constraints.cons5 = phase_lower_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit);

% 添加可视化
options = optimoptions(@ga,...
    'PlotFcn',{@gaplotbestf,@gaplotmaxconstr},...
    'Display','iter');

[sol, fval] = solve(model, "Solver", "ga", "Options",options)

function double_data = cell2double(x)
    % 将 cell 数组转换为 double 数组的一种方法是使用 cellfun 函数
    % cellfun(@str2double, tasks.textdata) 将 tasks.textdata 中的每个元素都转换为 double 类型
    % 然后将其转换为矩阵形式，即 cell2mat(cellfun(@str2double, tasks.textdata))
    % 如果 x 中的每个元素不是字符串，需要先进行处理，确保其为字符串类型
    
    % 下面解释一下 cellfun 函数的四个参数：

    % 1. @str2double：表示要应用的函数句柄。
    % @str2double 是一个函数句柄，指向 MATLAB 的内置函数 str2double，它用于将字符串转换为 double 类型。在这里，我们希望将 x 中的每个字符串转换为相应的数值。

    % 2. x ：表示要操作的输入 cell 数组。这是一个包含字符串的 cell 数组，我们希望将其中的每个字符串转换为数值。

    % 3. 'UniformOutput'：指定输出的类型。
    % 在这里，我们将它设置为 false，表示输出将是一个 cell 数组，而不是一个统一类型的数组。因为 str2double 函数的输出类型可能会不同，所以我们希望将其保存在 cell 数组中。

    % 4. false：指定了 'UniformOutput' 的值，即输出的类型为 cell 数组。
    double_data = cell2mat(cellfun(@str2double, x, 'UniformOutput', false));
end

function fitness = F(x, price, P)
    % 计算适应度值，即所有车花费的电价
    fitness = P' * x * price' *  0.25;
end

function fixed_model = fix_model_value(model, ta, td)
    % model 是你的优化模型
    % arrival_time_step 是到达时间步列表
    % departure_time_step 是离开时间步列表
    
    % 获取模型的行和列数
    [num_rows, num_cols] = size(model.Variables.x);
    
    % 遍历模型的每一个变量
    for i = 1:num_rows
        lower_bound = ta(i);
        upper_bound = td(i);
        
        for j = 1:num_cols
            % 检查是否在允许的时间范围内
            if j >= lower_bound && j <= upper_bound
                continue;
            else
                % 如果不在允许的时间范围内，将变量值固定为0
                eq(i) = model.Variables.x(i, j) == 0;
            end
        end
    end
    
    % 返回固定值后的模型
    fixed_model = eq;
end

% function fixed_model = fix_model_value(model, ta, td)
%     % model 是你的优化模型
%     % arrival_time_step 是到达时间步列表
%     % departure_time_step 是离开时间步列表
% 
%     % 获取模型的行和列数
%     [num_rows, num_cols] = size(model.Variables.x);
%     
%     % 初始化非线性等式约束列表
%     nonlinear_eq_cons = [];
%     
%     % 遍历模型的每一个变量
%     for i = 1:num_rows
%         lower_bound = round(ta(i));
%         upper_bound = round(td(i));
%         % 添加非线性等式约束
%         cons = struct('type', 'eq', 'func', {@(x) x(1:lower_bound) == 0, @(x) x(upper_bound:end) == 0});
%         nonlinear_eq_cons{end+1} = cons;
%     end
%     
%     % 将非线性等式约束添加到模型中
%     model.Constraints.nonlcon = @(x) cellfun(@(con) con.func{1}(x) && con.func{2}(x), nonlinear_eq_cons);
%     
%     % 返回固定值后的模型
%     fixed_model = model;
% end

function constraint = energy_constraint_rule(model, sa, sd, unit_increment)
    % model.Variables.x 是优化变量，这里 model.Variables.x 是一个大小为 [row, col] 的矩阵
    % unit_increment 是单位增量
    % sd 是离开SOC
    % sa 是到达SOC
    
    % 计算增量列表
    % 将 unit_increment 从(400, 1)拓展为(400, 96)和 model.Variables.x 的维度匹配
    % B = repmat(A, m, n)
    % 其中，A 是要复制的矩阵，m 是指定复制的行数，n 是指定复制的列数
    % repmat 函数会将矩阵 A 复制 m 行、n 列，生成一个新的矩阵 B。
    % unit_increment_extended = repmat(unit_increment, 1, 96);
    % increment_list = model.Variables.x .* unit_increment_extended;
    
    % 计算每辆车所有时间步的增量总和
    % total_increment = sum(increment_list, 2);
    total_increment = sum(model.Variables.x, 2);
    
    % 计算每辆车可由开关控制实现的最大 SOC 值
    % max_SOC = floor((sd - sa) / unit_increment) .* unit_increment;
    max_SOC = floor((sd - sa) ./ unit_increment);
    
    % 计算SOC约束条件，这种表示方式隐含了一组不等式约束，即每辆车的 SOC 增量大于等于允许的最大 SOC 值。
    % constraint = ... ≤ 0 
    constraint = max_SOC - total_increment <= 0;
end

function constraint = power_constraint_rule(model, P, restriction_in_power)
    % model.Variables.x 是优化变量，这里 model.Variables.x 是一个大小为 [row, col] 的矩阵
    % P 是每辆车的功率列表
    % restriction_in_power 是电路负荷限制列表
    
    % 计算每个时间步总用电功率
    % 将 P 从(400, 1)拓展为(400, 96)和 model.Variables.x 的维度匹配
    P_extended = repmat(P, 1, 96);
    used_power = sum(model.Variables.x .* P_extended, 1);
    
    % 计算功率上限约束条件
    constraint = used_power - restriction_in_power <= 0;
end

function three_phase = Calculate_phase(model, phi, P, phase_base_load)
    % x 是优化变量，这里假设 x 是一个大小为 [row, col] 的矩阵
    % phase_list 是车辆对应的充电相列表
    % power 是每辆车的充电功率列表
    % phase_base_load 是每个相的基础负载
        
    % 将 P 从(400, 1)拓展为(400, 96)和 model.Variables.x 的维度匹配
    P_extended = repmat(P, 1, 96);
    power = model.Variables.x .* P_extended;
    
    % 根据充电相列表获取各相充电功率,(.*, col)
    % 获得A相充电的车序号
    search_A = 'A';
    indices_A = [];
    % 获得B相充电的车序号
    search_B = 'B';
    indices_B = [];
    % 获得C相充电的车序号
    search_C = 'C';
    indices_C = [];
    
    for i = 1:numel(P)
        if strcmp(phi{i}, search_A)
            indices_A = [indices_A, i];
        
        elseif strcmp(phi{i}, search_B)
            indices_B = [indices_B, i];
            
        else
            indices_C = [indices_C, i];
        end
    end
            
    power_phaseA = power(indices_A, :);
    power_phaseB = power(indices_B, :);
    power_phaseC = power(indices_C, :);
    
    % 计算各相总充电负载,(1, col)
    phaseA = sum(power_phaseA, 1);
    phaseB = sum(power_phaseB, 1);
    phaseC = sum(power_phaseC, 1);
    
    % 计算每个相的总负载,(3, col)
    three_phase = [phaseA; phaseB; phaseC] + phase_base_load;
end

function minus_phase = Calculate_minus_phase(three_phase)
    % three_phase 是三相负载矩阵，每一列代表一个时间步的三相负载
    
    % 计算每个时间步的最大和最小负载
    % max_phase = max(three_phase, [], 1);
    % min_phase = min(three_phase, [], 1);
    
    % 计算 A、B、C 三相的功率
    A_phase = three_phase(1, :);
    B_phase = three_phase(2, :);
    C_phase = three_phase(3, :);
    
    % 计算 AB、AC、BC差
    AB_minus = A_phase - B_phase;
    AC_minus = A_phase - C_phase;
    BC_minus = B_phase - C_phase;
    
    % 计算每个时间步的负载不平衡
    minus_phase = [AB_minus; AC_minus; BC_minus];
end

function max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit)
    % three_phase 是三相负载矩阵，每一列代表一个时间步的三相负载
    % max_imbalance_limit 是最大不平衡限制
    
    % 计算每个时间步总负载
    total_power = sum(three_phase, 1);
    
    % 计算最大不平衡
    max_imbalance = max_imbalance_limit * total_power;
end

function constraint = phase_lower_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit)
    % 计算三相的负载功率
    three_phase = Calculate_phase(model, phi, P, phase_base_load);
    
    % 计算 AB、AC、BC 之差
    minus_phase = 3 * Calculate_minus_phase(three_phase);
    
    % 计算最大不平衡约束
    max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit);
    % 将 max_imbalance 维度从(1, 96) 拓展为 (3, 96) 匹配 minus_phase 维度
    max_imbalance_extended = repmat(max_imbalance, 3, 1);
    
    % 返回三相不平衡约束
    constraint = minus_phase - max_imbalance_extended <= 0;
end

function constraint = phase_upper_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit)
    % 计算三相的负载功率
    three_phase = Calculate_phase(model, phi, P, phase_base_load);
    
    % 计算 AB、AC、BC 之差
    minus_phase = 3 * Calculate_minus_phase(three_phase);
    
    % 计算最大不平衡约束
    max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit);
    % 将 max_imbalance 维度从(1, 96) 拓展为 (3, 96) 匹配 minus_phase 维度
    max_imbalance_extended = repmat(max_imbalance, 3, 1);
    
    % 返回三相不平衡约束
    constraint = -minus_phase - max_imbalance_extended <= 0;
end