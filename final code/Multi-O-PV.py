import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, value, LpStatus
import os

# ========================= 参数配置 =========================
GOOD_PRED_PATH = './InputData_v2/hainan_1.csv'
BAD_PRED_PATH = './InputData_v2/hainan_0.csv'
LOAD_PATH = None          # 若提供实际负荷文件，将覆盖所有生成模式

WINDOW_SIZE = 72
KEEP_FIRST = True

# ---------- 成本参数（可调） ----------
C_G = 0.35
C_BUY = 0.85
C_BAT_CYCLE = 2.0

# ---------- 系统物理参数 ----------
P_G_MAX = 200
P_G_MIN = 0
ETA_CH = 0.95
ETA_DIS = 0.95

# ---------- 场景定义 ----------
BATTERY_RATIOS = [0.05, 0.01, 0.0001]
HORIZONS = [24, 48, 72]
LOAD_MODES = ['ratio', 'complementary', 'stochastic']   # 三种模式
LOAD_RATIO = 0.55          # 用于 'ratio' 模式的固定比例

NUM_EPSILONS = 15
EPSILON_START_RATIO = 0.2
BASE_OUTPUT_DIR = 'results_scenarios_v3'

# ========================= 辅助函数 =========================
def load_dedup(seq, wsize, keep):
    raw = np.asarray(seq).flatten()
    K = len(raw) // wsize
    if K == 0:
        return np.array([])
    T = K + wsize - 1
    res = np.full(T, np.nan)
    for i in range(K):
        s, e = i, i + wsize
        if keep:
            mask = np.isnan(res[s:e])
            res[s:e][mask] = raw[i*wsize:(i+1)*wsize][mask]
        else:
            res[s:e] = raw[i*wsize:(i+1)*wsize]
    return res

def compute_max_curt(pv_pred, load):
    total = 0.0
    n_days = len(pv_pred) // 24
    for d in range(n_days):
        s = d * 24
        for t in range(24):
            surplus = pv_pred[s+t] + P_G_MIN - load[s+t]
            if surplus > 0:
                total += surplus
    return total

def solve_day_epsilon(pv_pred_day, load_day, epsilon, bat_cap, bat_pmax, horizon):
    T = horizon
    prob = LpProblem("DayEps", LpMinimize)
    Pg = [LpVariable(f"Pg_{t}", P_G_MIN, P_G_MAX) for t in range(T)]
    Pc = [LpVariable(f"Pc_{t}", 0, pv_pred_day[t]) for t in range(T)]
    Pch = [LpVariable(f"Pch_{t}", 0, bat_pmax) for t in range(T)]
    Pdis = [LpVariable(f"Pdis_{t}", 0, bat_pmax) for t in range(T)]
    E = [LpVariable(f"E_{t}", 0.1*bat_cap, bat_cap) for t in range(T+1)]

    for t in range(T):
        prob += (Pg[t] + pv_pred_day[t] + Pdis[t] - Pch[t] - Pc[t] == load_day[t])

    prob += E[0] == 0.5 * bat_cap
    prob += E[T] == 0.5 * bat_cap
    for t in range(1, T+1):
        prob += E[t] == E[t-1] + ETA_CH * Pch[t-1] - Pdis[t-1] / ETA_DIS

    prob += lpSum(C_G * Pg[t] for t in range(T)) + C_BAT_CYCLE * lpSum(Pch[t] + Pdis[t] for t in range(T))
    prob += lpSum(Pc[t] for t in range(T)) <= epsilon

    prob.solve()
    if LpStatus[prob.status] == 'Optimal':
        return ([value(Pg[t]) for t in range(T)],
                [value(Pc[t]) for t in range(T)],
                [value(Pch[t]) for t in range(T)],
                [value(Pdis[t]) for t in range(T)],
                [value(E[t]) for t in range(T+1)])
    else:
        prob2 = LpProblem("DayEps_Relax", LpMinimize)
        Pg2 = [LpVariable(f"Pg2_{t}", P_G_MIN, P_G_MAX) for t in range(T)]
        Pc2 = [LpVariable(f"Pc2_{t}", 0, pv_pred_day[t]) for t in range(T)]
        Pch2 = [LpVariable(f"Pch2_{t}", 0, bat_pmax) for t in range(T)]
        Pdis2 = [LpVariable(f"Pdis2_{t}", 0, bat_pmax) for t in range(T)]
        E2 = [LpVariable(f"E2_{t}", 0.1*bat_cap, bat_cap) for t in range(T+1)]
        for t in range(T):
            prob2 += (Pg2[t] + pv_pred_day[t] + Pdis2[t] - Pch2[t] - Pc2[t] == load_day[t])
        prob2 += E2[0] == 0.5 * bat_cap
        prob2 += E2[T] == 0.5 * bat_cap
        for t in range(1, T+1):
            prob2 += E2[t] == E2[t-1] + ETA_CH * Pch2[t-1] - Pdis2[t-1] / ETA_DIS
        prob2 += lpSum(C_G * Pg2[t] for t in range(T)) + C_BAT_CYCLE * lpSum(Pch2[t] + Pdis2[t] for t in range(T))
        prob2.solve()
        return ([value(Pg2[t]) for t in range(T)],
                [value(Pc2[t]) for t in range(T)],
                [value(Pch2[t]) for t in range(T)],
                [value(Pdis2[t]) for t in range(T)],
                [value(E2[t]) for t in range(T+1)])

def rolling_epsilon_eval(pv_pred_all, pv_real_all, load_all, eps_list, bat_cap, bat_pmax, horizon):
    n_days = len(pv_pred_all) // 24
    n_eps = len(eps_list)
    cost_tot = np.zeros(n_eps)
    curt_tot = np.zeros(n_eps)
    buy_tot = np.zeros(n_eps)
    exec_hours = min(horizon, 24)
    for d in range(n_days):
        start = d * 24
        end = min(start + horizon, len(pv_pred_all))
        pv_pred_seg = pv_pred_all[start:end]
        load_seg = load_all[start:end]
        pv_real_seg = pv_real_all[start:end]
        if len(pv_pred_seg) < horizon:
            break
        for i, eps in enumerate(eps_list):
            day_eps = eps / n_days
            sched = solve_day_epsilon(pv_pred_seg, load_seg, day_eps, bat_cap, bat_pmax, horizon)
            Pg, Pc_plan, Pch, Pdis, _ = sched
            fuel = 0.0; buy = 0.0; curt = 0.0
            for t in range(exec_hours):
                net = Pg[t] + pv_real_seg[t] + Pdis[t] - Pch[t] - load_seg[t]
                if net >= 0:
                    curt += net
                else:
                    buy -= net
                fuel += C_G * Pg[t]
            cost_tot[i] += fuel + C_BUY * buy
            curt_tot[i] += curt
            buy_tot[i] += buy
    return cost_tot, curt_tot, buy_tot

# ========== 合成负荷生成函数 ==========
def generate_complementary_load(pv_series, p_max, D0=0.4, alpha=0.5, noise_std=0.08):
    p_norm = pv_series / p_max
    load = D0 + alpha * (1 - p_norm) + np.random.normal(0, noise_std, size=len(pv_series))
    return np.maximum(load, 0)  # 归一化负荷（0~1左右）

def generate_stochastic_load(pv_series, mu=None, noise_std_ratio=0.1):
    if mu is None:
        positive = pv_series[pv_series > 0]
        mu = np.mean(positive) if len(positive) > 0 else np.mean(pv_series)
    noise_std = noise_std_ratio * mu
    load = mu + np.random.normal(0, noise_std, size=len(pv_series))
    return np.maximum(load, 0)

# ========================= 主程序 =========================
def main():
    # 读取数据
    df_good = pd.read_csv(GOOD_PRED_PATH, header=None)
    real_raw = df_good.iloc[1].values
    pred_good_raw = df_good.iloc[2].values
    df_bad = pd.read_csv(BAD_PRED_PATH, header=None)
    pred_bad_raw = df_bad.iloc[2].values

    P_pv_real = load_dedup(real_raw, WINDOW_SIZE, KEEP_FIRST)
    P_pv_good = load_dedup(pred_good_raw, WINDOW_SIZE, KEEP_FIRST)
    P_pv_bad  = load_dedup(pred_bad_raw, WINDOW_SIZE, KEEP_FIRST)

    cap_estimate = np.max(P_pv_real) if np.max(P_pv_real) > 0 else 1.0
    print(f"装机容量估计: {cap_estimate:.2f} MW")

    # ===== 生成三种负荷序列 =====
    np.random.seed(2025)  # 固定随机种子
    load_dict = {}

    # 模式1: ratio
    load_dict['ratio'] = np.full(len(P_pv_real), LOAD_RATIO * cap_estimate)

    # 模式2: complementary (归一化后乘以 cap_estimate)
    load_norm = generate_complementary_load(P_pv_real, p_max=cap_estimate, D0=0.4, alpha=0.5, noise_std=0.08)
    load_dict['complementary'] = load_norm * cap_estimate

    # 模式3: stochastic (均值取日间光伏平均值)
    mean_daytime = np.mean(P_pv_real[P_pv_real > 0]) if np.any(P_pv_real > 0) else np.mean(P_pv_real)
    load_dict['stochastic'] = generate_stochastic_load(P_pv_real, mu=mean_daytime, noise_std_ratio=0.1)

    # 若提供了实际负荷文件，则覆盖所有模式
    if LOAD_PATH and os.path.exists(LOAD_PATH):
        load_raw = pd.read_csv(LOAD_PATH, header=None).values.flatten()
        L_real = load_dedup(load_raw, WINDOW_SIZE, KEEP_FIRST)
        if len(L_real) < len(P_pv_real):
            L_real = np.pad(L_real, (0, len(P_pv_real)-len(L_real)), constant_values=np.mean(L_real))
        for mode in load_dict:
            load_dict[mode] = L_real[:len(P_pv_real)]
        print("使用外部负荷文件，已覆盖所有生成模式。")

    # 创建输出目录
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    summary_records = []

    # 场景循环
    for bat_ratio in BATTERY_RATIOS:
        bat_cap = max(bat_ratio * cap_estimate, 1e-6)
        bat_pmax = bat_cap
        for horizon in HORIZONS:
            for mode in LOAD_MODES:
                L = load_dict[mode]
                if len(L) > len(P_pv_real):
                    L = L[:len(P_pv_real)]
                elif len(L) < len(P_pv_real):
                    L = np.pad(L, (0, len(P_pv_real)-len(L)), constant_values=np.mean(L))

                # 计算ε列表
                max_curt_good = compute_max_curt(P_pv_good, L)
                max_curt_bad = compute_max_curt(P_pv_bad, L)
                eps_good = np.linspace(EPSILON_START_RATIO * max_curt_good, max_curt_good, NUM_EPSILONS) if max_curt_good > 0 else np.linspace(0, 1e-3, NUM_EPSILONS)
                eps_bad = np.linspace(EPSILON_START_RATIO * max_curt_bad, max_curt_bad, NUM_EPSILONS) if max_curt_bad > 0 else np.linspace(0, 1e-3, NUM_EPSILONS)

                cost_g, curt_g, buy_g = rolling_epsilon_eval(P_pv_good, P_pv_real, L, eps_good, bat_cap, bat_pmax, horizon)
                cost_b, curt_b, buy_b = rolling_epsilon_eval(P_pv_bad, P_pv_real, L, eps_bad, bat_cap, bat_pmax, horizon)

                idx_mid = NUM_EPSILONS // 2
                summary_records.append({
                    'Battery_Ratio': bat_ratio,
                    'Horizon': horizon,
                    'Load_Mode': mode,
                    'Good_Cost': cost_g[idx_mid],
                    'Good_Curtail': curt_g[idx_mid],
                    'Good_Buy': buy_g[idx_mid],
                    'Good_Renewable': 1 - curt_g[idx_mid] / np.sum(P_pv_real),
                    'Bad_Cost': cost_b[idx_mid],
                    'Bad_Curtail': curt_b[idx_mid],
                    'Bad_Buy': buy_b[idx_mid],
                    'Bad_Renewable': 1 - curt_b[idx_mid] / np.sum(P_pv_real),
                })

                # 保存单个场景图和数据
                fig, ax = plt.subplots(figsize=(8,6))
                ax.plot(cost_g, curt_g, 'o-', color='green', label='Good pred')
                ax.plot(cost_b, curt_b, 's-', color='red', label='Bad pred')
                ax.set_xlabel('Total Cost ($)')
                ax.set_ylabel('Curtailment (MWh)')
                title = f'Battery={bat_ratio*100:.2f}%, Horizon={horizon}h, Load={mode}'
                ax.set_title(title)
                ax.legend()
                ax.grid(True)
                subdir = os.path.join(BASE_OUTPUT_DIR, f'bat_{bat_ratio:.4f}_hor{horizon}_{mode}')
                os.makedirs(subdir, exist_ok=True)
                fig.savefig(os.path.join(subdir, 'pareto.png'), dpi=300)
                plt.close(fig)

                df_scene = pd.DataFrame({
                    'Epsilon_Good': eps_good,
                    'Good_Cost': cost_g,
                    'Good_Curtail': curt_g,
                    'Good_Buy': buy_g,
                    'Epsilon_Bad': eps_bad,
                    'Bad_Cost': cost_b,
                    'Bad_Curtail': curt_b,
                    'Bad_Buy': buy_b,
                })
                df_scene.to_csv(os.path.join(subdir, 'results.csv'), index=False)
                print(f"完成场景: bat={bat_ratio:.2%}, horizon={horizon}, load={mode}")

    # 汇总表
    df_summary = pd.DataFrame(summary_records)
    df_summary.to_csv(os.path.join(BASE_OUTPUT_DIR, 'summary_all_scenarios.csv'), index=False)

    # 绘制按负荷模式分组的成本对比图
    for mode in LOAD_MODES:
        subset = df_summary[df_summary['Load_Mode'] == mode]
        if subset.empty:
            continue
        fig, ax = plt.subplots(figsize=(10,6))
        for horizon in HORIZONS:
            sub = subset[subset['Horizon'] == horizon]
            ax.plot(sub['Battery_Ratio'], sub['Good_Cost'], 'o-', label=f'Good, H={horizon}')
            ax.plot(sub['Battery_Ratio'], sub['Bad_Cost'], 's--', label=f'Bad, H={horizon}')
        ax.set_xlabel('Battery Ratio')
        ax.set_ylabel('Cost ($)')
        ax.set_title(f'Load Mode: {mode}')
        ax.legend()
        ax.grid(True)
        fig.savefig(os.path.join(BASE_OUTPUT_DIR, f'summary_cost_{mode}.png'), dpi=300)
        plt.close(fig)

    print(f"所有结果已保存至 '{BASE_OUTPUT_DIR}'")

if __name__ == '__main__':
    main()