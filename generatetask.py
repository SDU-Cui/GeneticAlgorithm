import numpy as np
import pandas as pd
row = 200
col = 96
elements = ['A', 'B', 'C']

def Scale(ρ, MIN, MAX):
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

ta = np.random.randint(1, 10, size=(row))
td = np.random.randint(86, 96, size=(row))
sa = np.random.random(size=(row))
sd = np.random.random(size=(row))
E = np.random.randint(70, 96, size=(row))
P = np.random.randint(3, 7, size=(row))
η = np.random.random(size=(row))
Φ = np.random.choice(elements, size=(row))
sa = Scale(sa, 0.25, 0.4)
sd = Scale(sd, 0.8, 0.9)
η = np.floor(Scale(η, 0.95, 1) * 100) / 100
Δ = P * η * 0.25 / E
k= np.floor((sd - sa) / Δ)

names = ['ta', 'td', 'sa', 'sd', 'E', 'P', 'η', 'Φ', 'Δ', 'k']
data = np.transpose(np.vstack((ta, td, sa, sd, E, P, η, Φ, Δ, k)))

df = pd.DataFrame(data, columns=names)
df.to_csv('data/vehicle_data_200(2).csv', index_label='v', encoding='ANSI')