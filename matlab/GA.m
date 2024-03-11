clear;
clc;
row = 9;
col = 96;
tasks = importdata("D:\MyProject\Charge_Pyomo\GeneticAlgorithm\data\vehicle_data_9.csv");
phase_base_load = importdata("D:\MyProject\Charge_Pyomo\GeneticAlgorithm\data\phase_base_load.csv");
ta = cell2double(tasks.textdata(2:end, 2)); %����ʱ��
td = cell2double(tasks.textdata(2:end, 3)); %�뿪ʱ��
sa = cell2double(tasks.textdata(2:end, 4)); %����SOC
sd = cell2double(tasks.textdata(2:end, 5)); %�뿪SOC
E = cell2double(tasks.textdata(2:end, 6));  %�������
P = cell2double(tasks.textdata(2:end, 7));  %��繦��
eta = cell2double(tasks.textdata(2:end, 8));%���Ч��
phi = tasks.textdata(2:end, 9);             %�������
delta = tasks.data(:, 1);                %���ؿ���һ�γ��SOC
k = tasks.data(:, 2);                    %��Ҫ������

% Ϊ���SOCԼ����׼��������ÿ����15min�ĳ������
unit_increment = P .* eta .* 0.25 ./ E; %ÿ����15min�ĵ�λ����,(col,)
% ��û������غ͵�·����get_restriction_in_power
base_load = sum(phase_base_load, 1); %���ڶ�ά����������1�ǰ�����ӣ�2�ǰ������
restriction_in_power = (2200 - base_load);
% ������಻ƽ���
max_imbalance_limit = 0.04;
% ���
price = ones(1, col);
price(1, 30:end) = 0.1;

% ���ñ��� x
x = optimvar("x", row, col, "Type", "integer", "LowerBound", 0, "UpperBound", 1);

% �����Ż����� model
model = optimproblem("ObjectiveSense", "minimize");

% �����Ż�Ŀ��
model.Objective = F(x, price, P);

% ʱ��Լ��
model.Constraints.cons1 = fix_model_value(model, ta, td);
% SOCԼ��
model.Constraints.cons2 = energy_constraint_rule(model, sa, sd, unit_increment);
% ��������Լ��
model.Constraints.cons3 = power_constraint_rule(model, P, restriction_in_power);
% ���಻ƽ��Լ��
model.Constraints.cons4 = phase_upper_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit);
model.Constraints.cons5 = phase_lower_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit);

% ��ӿ��ӻ�
options = optimoptions(@ga,...
    'PlotFcn',{@gaplotbestf,@gaplotmaxconstr},...
    'Display','iter');

[sol, fval] = solve(model, "Solver", "ga", "Options",options)

function double_data = cell2double(x)
    % �� cell ����ת��Ϊ double �����һ�ַ�����ʹ�� cellfun ����
    % cellfun(@str2double, tasks.textdata) �� tasks.textdata �е�ÿ��Ԫ�ض�ת��Ϊ double ����
    % Ȼ����ת��Ϊ������ʽ���� cell2mat(cellfun(@str2double, tasks.textdata))
    % ��� x �е�ÿ��Ԫ�ز����ַ�������Ҫ�Ƚ��д���ȷ����Ϊ�ַ�������
    
    % �������һ�� cellfun �������ĸ�������

    % 1. @str2double����ʾҪӦ�õĺ��������
    % @str2double ��һ�����������ָ�� MATLAB �����ú��� str2double�������ڽ��ַ���ת��Ϊ double ���͡����������ϣ���� x �е�ÿ���ַ���ת��Ϊ��Ӧ����ֵ��

    % 2. x ����ʾҪ���������� cell ���顣����һ�������ַ����� cell ���飬����ϣ�������е�ÿ���ַ���ת��Ϊ��ֵ��

    % 3. 'UniformOutput'��ָ����������͡�
    % ��������ǽ�������Ϊ false����ʾ�������һ�� cell ���飬������һ��ͳһ���͵����顣��Ϊ str2double ������������Ϳ��ܻ᲻ͬ����������ϣ�����䱣���� cell �����С�

    % 4. false��ָ���� 'UniformOutput' ��ֵ�������������Ϊ cell ���顣
    double_data = cell2mat(cellfun(@str2double, x, 'UniformOutput', false));
end

function fitness = F(x, price, P)
    % ������Ӧ��ֵ�������г����ѵĵ��
    fitness = P' * x * price' *  0.25;
end

function fixed_model = fix_model_value(model, ta, td)
    % model ������Ż�ģ��
    % arrival_time_step �ǵ���ʱ�䲽�б�
    % departure_time_step ���뿪ʱ�䲽�б�
    
    % ��ȡģ�͵��к�����
    [num_rows, num_cols] = size(model.Variables.x);
    
    % ����ģ�͵�ÿһ������
    for i = 1:num_rows
        lower_bound = ta(i);
        upper_bound = td(i);
        
        for j = 1:num_cols
            % ����Ƿ��������ʱ�䷶Χ��
            if j >= lower_bound && j <= upper_bound
                continue;
            else
                % ������������ʱ�䷶Χ�ڣ�������ֵ�̶�Ϊ0
                eq(i) = model.Variables.x(i, j) == 0;
            end
        end
    end
    
    % ���ع̶�ֵ���ģ��
    fixed_model = eq;
end

% function fixed_model = fix_model_value(model, ta, td)
%     % model ������Ż�ģ��
%     % arrival_time_step �ǵ���ʱ�䲽�б�
%     % departure_time_step ���뿪ʱ�䲽�б�
% 
%     % ��ȡģ�͵��к�����
%     [num_rows, num_cols] = size(model.Variables.x);
%     
%     % ��ʼ�������Ե�ʽԼ���б�
%     nonlinear_eq_cons = [];
%     
%     % ����ģ�͵�ÿһ������
%     for i = 1:num_rows
%         lower_bound = round(ta(i));
%         upper_bound = round(td(i));
%         % ��ӷ����Ե�ʽԼ��
%         cons = struct('type', 'eq', 'func', {@(x) x(1:lower_bound) == 0, @(x) x(upper_bound:end) == 0});
%         nonlinear_eq_cons{end+1} = cons;
%     end
%     
%     % �������Ե�ʽԼ����ӵ�ģ����
%     model.Constraints.nonlcon = @(x) cellfun(@(con) con.func{1}(x) && con.func{2}(x), nonlinear_eq_cons);
%     
%     % ���ع̶�ֵ���ģ��
%     fixed_model = model;
% end

function constraint = energy_constraint_rule(model, sa, sd, unit_increment)
    % model.Variables.x ���Ż����������� model.Variables.x ��һ����СΪ [row, col] �ľ���
    % unit_increment �ǵ�λ����
    % sd ���뿪SOC
    % sa �ǵ���SOC
    
    % ���������б�
    % �� unit_increment ��(400, 1)��չΪ(400, 96)�� model.Variables.x ��ά��ƥ��
    % B = repmat(A, m, n)
    % ���У�A ��Ҫ���Ƶľ���m ��ָ�����Ƶ�������n ��ָ�����Ƶ�����
    % repmat �����Ὣ���� A ���� m �С�n �У�����һ���µľ��� B��
    % unit_increment_extended = repmat(unit_increment, 1, 96);
    % increment_list = model.Variables.x .* unit_increment_extended;
    
    % ����ÿ��������ʱ�䲽�������ܺ�
    % total_increment = sum(increment_list, 2);
    total_increment = sum(model.Variables.x, 2);
    
    % ����ÿ�������ɿ��ؿ���ʵ�ֵ���� SOC ֵ
    % max_SOC = floor((sd - sa) / unit_increment) .* unit_increment;
    max_SOC = floor((sd - sa) ./ unit_increment);
    
    % ����SOCԼ�����������ֱ�ʾ��ʽ������һ�鲻��ʽԼ������ÿ������ SOC �������ڵ����������� SOC ֵ��
    % constraint = ... �� 0 
    constraint = max_SOC - total_increment <= 0;
end

function constraint = power_constraint_rule(model, P, restriction_in_power)
    % model.Variables.x ���Ż����������� model.Variables.x ��һ����СΪ [row, col] �ľ���
    % P ��ÿ�����Ĺ����б�
    % restriction_in_power �ǵ�·���������б�
    
    % ����ÿ��ʱ�䲽���õ繦��
    % �� P ��(400, 1)��չΪ(400, 96)�� model.Variables.x ��ά��ƥ��
    P_extended = repmat(P, 1, 96);
    used_power = sum(model.Variables.x .* P_extended, 1);
    
    % ���㹦������Լ������
    constraint = used_power - restriction_in_power <= 0;
end

function three_phase = Calculate_phase(model, phi, P, phase_base_load)
    % x ���Ż�������������� x ��һ����СΪ [row, col] �ľ���
    % phase_list �ǳ�����Ӧ�ĳ�����б�
    % power ��ÿ�����ĳ�繦���б�
    % phase_base_load ��ÿ����Ļ�������
        
    % �� P ��(400, 1)��չΪ(400, 96)�� model.Variables.x ��ά��ƥ��
    P_extended = repmat(P, 1, 96);
    power = model.Variables.x .* P_extended;
    
    % ���ݳ�����б��ȡ�����繦��,(.*, col)
    % ���A����ĳ����
    search_A = 'A';
    indices_A = [];
    % ���B����ĳ����
    search_B = 'B';
    indices_B = [];
    % ���C����ĳ����
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
    
    % ��������ܳ�縺��,(1, col)
    phaseA = sum(power_phaseA, 1);
    phaseB = sum(power_phaseB, 1);
    phaseC = sum(power_phaseC, 1);
    
    % ����ÿ������ܸ���,(3, col)
    three_phase = [phaseA; phaseB; phaseC] + phase_base_load;
end

function minus_phase = Calculate_minus_phase(three_phase)
    % three_phase �����ฺ�ؾ���ÿһ�д���һ��ʱ�䲽�����ฺ��
    
    % ����ÿ��ʱ�䲽��������С����
    % max_phase = max(three_phase, [], 1);
    % min_phase = min(three_phase, [], 1);
    
    % ���� A��B��C ����Ĺ���
    A_phase = three_phase(1, :);
    B_phase = three_phase(2, :);
    C_phase = three_phase(3, :);
    
    % ���� AB��AC��BC��
    AB_minus = A_phase - B_phase;
    AC_minus = A_phase - C_phase;
    BC_minus = B_phase - C_phase;
    
    % ����ÿ��ʱ�䲽�ĸ��ز�ƽ��
    minus_phase = [AB_minus; AC_minus; BC_minus];
end

function max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit)
    % three_phase �����ฺ�ؾ���ÿһ�д���һ��ʱ�䲽�����ฺ��
    % max_imbalance_limit �����ƽ������
    
    % ����ÿ��ʱ�䲽�ܸ���
    total_power = sum(three_phase, 1);
    
    % �������ƽ��
    max_imbalance = max_imbalance_limit * total_power;
end

function constraint = phase_lower_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit)
    % ��������ĸ��ع���
    three_phase = Calculate_phase(model, phi, P, phase_base_load);
    
    % ���� AB��AC��BC ֮��
    minus_phase = 3 * Calculate_minus_phase(three_phase);
    
    % �������ƽ��Լ��
    max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit);
    % �� max_imbalance ά�ȴ�(1, 96) ��չΪ (3, 96) ƥ�� minus_phase ά��
    max_imbalance_extended = repmat(max_imbalance, 3, 1);
    
    % �������಻ƽ��Լ��
    constraint = minus_phase - max_imbalance_extended <= 0;
end

function constraint = phase_upper_constraint_rule(model, phi, P, phase_base_load, max_imbalance_limit)
    % ��������ĸ��ع���
    three_phase = Calculate_phase(model, phi, P, phase_base_load);
    
    % ���� AB��AC��BC ֮��
    minus_phase = 3 * Calculate_minus_phase(three_phase);
    
    % �������ƽ��Լ��
    max_imbalance = Calculate_max_imbalance(three_phase, max_imbalance_limit);
    % �� max_imbalance ά�ȴ�(1, 96) ��չΪ (3, 96) ƥ�� minus_phase ά��
    max_imbalance_extended = repmat(max_imbalance, 3, 1);
    
    % �������಻ƽ��Լ��
    constraint = -minus_phase - max_imbalance_extended <= 0;
end