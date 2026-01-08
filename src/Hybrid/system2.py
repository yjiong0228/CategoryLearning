import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import math
import pickle  # 用于保存复杂的 simulation 结果 (如 cluster 历史)
from joblib import Parallel, delayed

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================

class RationalModelSystem2:
    def __init__(self, D, alpha, sigma, l0):
        self.D = D
        self.alpha = alpha
        self.sigma = sigma
        self.l0 = l0
        self.clusters = [] 
        self.total_N = 0
        self.history = [] 

    def _gaussian_likelihood(self, x, cluster):
        mu = cluster['sum_x'] / cluster['N']
        prefactor = (1.0 / (np.sqrt(2 * np.pi) * self.sigma)) ** self.D
        dist_sq = np.sum((x - mu) ** 2)
        exponent = np.exp(-dist_sq / (2 * self.sigma ** 2))
        return prefactor * exponent

    def get_posteriors(self, x):
        priors = []
        likelihoods = []
        for cluster in self.clusters:
            prior = cluster['N'] / (self.total_N + self.alpha)
            priors.append(prior)
            lik = self._gaussian_likelihood(x, cluster)
            likelihoods.append(lik)
        priors.append(self.alpha / (self.total_N + self.alpha))
        likelihoods.append(self.l0)
        unnormalized = np.array(priors) * np.array(likelihoods)
        if np.sum(unnormalized) == 0:
            return np.ones(len(unnormalized)) / len(unnormalized)
        return unnormalized / np.sum(unnormalized)

    def get_choice_probs(self, x):
        if self.total_N == 0:
            return {1: 0.5, 2: 0.5}
        posteriors = self.get_posteriors(x)
        prob_y = {1: 0.0, 2: 0.0}
        for k, cluster in enumerate(self.clusters):
            w_k = posteriors[k]
            n_1 = cluster['y_counts'].get(1, 0)
            n_2 = cluster['y_counts'].get(2, 0)
            p_1_k = (n_1 + 1) / (cluster['N'] + 2)
            p_2_k = (n_2 + 1) / (cluster['N'] + 2)
            prob_y[1] += w_k * p_1_k
            prob_y[2] += w_k * p_2_k
        w_new = posteriors[-1]
        prob_y[1] += w_new * 0.5
        prob_y[2] += w_new * 0.5
        return prob_y

    def update(self, x, y):
        self._record_history()
        if self.total_N == 0:
            self._create_new_cluster(x, y)
        else:
            posteriors = self.get_posteriors(x)
            winner_idx = np.argmax(posteriors)
            if winner_idx == len(self.clusters):
                self._create_new_cluster(x, y)
            else:
                self._update_cluster(winner_idx, x, y)
        self.total_N += 1

    def _create_new_cluster(self, x, y):
        new_cluster = {'N': 1, 'sum_x': x.copy(), 'y_counts': {1: 0, 2: 0}}
        new_cluster['y_counts'][y] = 1
        self.clusters.append(new_cluster)

    def _update_cluster(self, idx, x, y):
        cluster = self.clusters[idx]
        cluster['N'] += 1
        cluster['sum_x'] += x
        cluster['y_counts'][y] = cluster['y_counts'].get(y, 0) + 1

    def _record_history(self):
        snapshot = []
        for c in self.clusters:
            mu = c['sum_x'] / c['N']
            snapshot.append(mu.copy())
        self.history.append(snapshot)

def neg_log_likelihood(params, df_sub):
    alpha, sigma, l0 = params
    if sigma <= 1e-4 or alpha <= 1e-4 or l0 <= 1e-6: return 1e10
    model = RationalModelSystem2(4, alpha, sigma, l0)
    nll = 0.0
    for _, row in df_sub.iterrows():
        x = np.array([row['feature1'], row['feature2'], row['feature3'], row['feature4']])
        choice = row['choice'] 
        probs = model.get_choice_probs(x)
        p_choice = max(probs[choice], 1e-6)
        nll -= np.log(p_choice)
        true_cat = row['category']
        model.update(x, true_cat)
    return nll

# ==========================================
# 2. 模块 A: 拟合 (Fitting)
# ==========================================

def fit_single_subject(sub_id, sub_df):
    """只负责拟合参数"""
    print(f"[Fit] Sub {sub_id} 正在拟合...")
    initial_params = [1.0, 0.15, 0.001]
    bounds = [(0.01, 5.0), (0.01, 0.5), (1e-6, 0.05)]
    
    try:
        res = minimize(neg_log_likelihood, initial_params, args=(sub_df,), 
                       bounds=bounds, method='L-BFGS-B')
        return {
            'iSub': sub_id,
            'alpha': res.x[0],
            'sigma': res.x[1],
            'l0': res.x[2],
            'nll': res.fun
        }
    except Exception as e:
        print(f"[Fit] Sub {sub_id} 失败: {e}")
        return None

def run_fitting(data_path, output_param_file):
    print(f"\n=== 阶段 1/3: 参数拟合 ===")
    df = pd.read_csv(data_path)
    sub_ids = df['iSub'].unique()
    
    results = Parallel(n_jobs=-1)(
        delayed(fit_single_subject)(sub_id, df[df['iSub'] == sub_id].copy().reset_index(drop=True))
        for sub_id in sub_ids
    )
    
    params_df = pd.DataFrame([r for r in results if r is not None])
    params_df.sort_values('iSub', inplace=True)
    params_df.to_csv(output_param_file, index=False)
    print(f"拟合完成，参数已保存至: {output_param_file}")

# ==========================================
# 3. 模块 B: 模拟 (Simulation)
# ==========================================

def simulate_subject(sub_id, sub_df, params):
    """
    运行模型，生成并返回所有需要保存的详细数据。
    这里不画图，只产生数据。
    """
    alpha, sigma, l0 = params['alpha'], params['sigma'], params['l0']
    model = RationalModelSystem2(4, alpha, sigma, l0)
    
    # 用于保存每一试次的标量数据
    trial_data = []
    
    for i, row in sub_df.iterrows():
        x = np.array([row['feature1'], row['feature2'], row['feature3'], row['feature4']])
        true_cat = row['category']  # 1 或 2
        choice = row['choice']      # 1 或 2
        
        # 1. 获取模型预测概率
        probs = model.get_choice_probs(x)
        
        # 2. 核心：计算模型对【真实类别】的预测概率
        # 如果 true_cat 是 1，取 probs[1]；如果是 2，取 probs[2]
        prob_of_correct_cat = probs.get(true_cat, 0.0)
        
        # 3. 记录被试行为 (0:错误, 1:正确)
        human_is_correct = 1 if choice == true_cat else 0
        
        # 4. 记录熵
        entropy = -sum(p*np.log2(p+1e-9) for p in probs.values())
        
        trial_data.append({
            'trial': i+1,
            'prob_correct': prob_of_correct_cat, # 模型预测正确类别的概率
            'human_correct': human_is_correct,   # 被试是否正确
            'num_clusters': len(model.clusters), # 当前 Cluster 数量
            'entropy': entropy                   # 熵
        })
        
        # 5. 更新模型
        model.update(x, true_cat)
        
    model._record_history() # 记录最后状态用于画 Cluster 轨迹
    
    return {
        'iSub': sub_id,
        'trial_df': pd.DataFrame(trial_data), # 标量数据 (用于画 Accuracy, Entropy, N_Clusters)
        'cluster_history': model.history      # 复杂数据 (用于画 Dim Evolution)
    }

def run_simulation(data_path, param_file, output_sim_file):
    print(f"\n=== 阶段 2/3: 模型模拟 ===")
    if not os.path.exists(param_file):
        print("错误：缺少参数文件，请先运行拟合。")
        return

    df_raw = pd.read_csv(data_path)
    df_params = pd.read_csv(param_file).set_index('iSub')
    sub_ids = df_raw['iSub'].unique()
    
    all_sim_results = {} # 字典: {sub_id: 模拟结果对象}

    print("正在生成模拟数据...")
    for sub_id in sub_ids:
        if sub_id not in df_params.index: continue
        
        sub_df = df_raw[df_raw['iSub'] == sub_id].copy().reset_index(drop=True)
        params = df_params.loc[sub_id]
        
        sim_res = simulate_subject(sub_id, sub_df, params)
        all_sim_results[sub_id] = sim_res
    
    # 使用 Pickle 保存所有结果 (因为 cluster_history 是复杂的嵌套列表，CSV存不下)
    with open(output_sim_file, 'wb') as f:
        pickle.dump(all_sim_results, f)
        
    print(f"模拟结束，详细数据已打包保存至: {output_sim_file}")

# ==========================================
# 4. 模块 C: 绘图 (Plotting)
# ==========================================

def run_plotting(data_path, sim_file):
    print(f"\n=== 阶段 3/3: 结果绘图 ===")
    if not os.path.exists(sim_file):
        print("错误：缺少模拟数据文件，请先运行模拟。")
        return

    # 1. 加载数据
    print("正在加载模拟数据...")
    with open(sim_file, 'rb') as f:
        all_sim_results = pickle.load(f)
    
    # 2. 遍历每个被试，绘制个体图表
    os.makedirs('plots', exist_ok=True)
    
    for sub_id, res in all_sim_results.items():
        save_dir = f'plots/iSub_{sub_id}'
        os.makedirs(save_dir, exist_ok=True)
        
        trial_df = res['trial_df']
        history = res['cluster_history']
        trials = trial_df['trial']
        
        # --- 图 1: Cluster 数量变化 ---
        plt.figure(figsize=(8, 4))
        plt.plot(trials, trial_df['num_clusters'], 'k-', linewidth=2)
        plt.title(f'Subject {sub_id}: Number of Clusters')
        plt.xlabel('Trial')
        plt.ylabel('Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/num_clusters.png')
        plt.close()

        # --- 图 2: 熵 ---
        plt.figure(figsize=(8, 4))
        plt.plot(trials, trial_df['entropy'], 'r-', linewidth=1)
        plt.title(f'Subject {sub_id}: Prediction Entropy')
        plt.xlabel('Trial')
        plt.ylabel('Entropy (bits)')
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{save_dir}/entropy.png')
        plt.close()

        # --- 图 3: Cluster 均值轨迹 (Dim Evolution) ---
        # 需要解析 history 数据
        max_clusters = trial_df['num_clusters'].max()
        if max_clusters > 0:
            vals_per_dim = {d: pd.DataFrame(index=trials, columns=range(max_clusters)) for d in range(4)}
            for i, snapshot in enumerate(history):
                t_idx = i + 1
                for k, mu in enumerate(snapshot):
                    for d in range(4):
                        vals_per_dim[d].loc[t_idx, k] = mu[d]
            
            n_cols = 3
            n_rows = math.ceil(max_clusters / n_cols)
            for d in range(4):
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), sharex=True, sharey=True)
                fig.suptitle(f'Subject {sub_id} - Feature Dim {d+1}', fontsize=16)
                if max_clusters == 1: axes = [axes]
                else: axes = axes.flatten()
                
                for k in range(max_clusters):
                    ax = axes[k]
                    series = vals_per_dim[d][k].dropna()
                    if len(series) > 0:
                        ax.plot(series.index, series.values, label=f'C{k+1}', color=f'C{k}')
                        ax.set_title(f'Cluster {k+1}')
                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(0.5, 0.5, 'Empty', ha='center', va='center')
                    
                    if k >= max_clusters - n_cols: ax.set_xlabel('Trial')
                
                for j in range(max_clusters, len(axes)): axes[j].axis('off')
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f'{save_dir}/dim_{d+1}_clusters.png')
                plt.close(fig)

    # 3. 绘制总图：Accuracy 对比
    print("正在生成 Accuracy 汇总图...")
    sub_ids_sorted = sorted(all_sim_results.keys())
    n_subs = len(sub_ids_sorted)
    n_cols = 5
    n_rows = math.ceil(n_subs / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows), sharex=True, sharey=True)
    fig.suptitle('Model Prob (Correct Category) vs Human Accuracy (Smoothed)', fontsize=16)
    
    axes_flat = axes.flatten() if n_subs > 1 else [axes]
    
    for idx, sub_id in enumerate(sub_ids_sorted):
        ax = axes_flat[idx]
        res = all_sim_results[sub_id]
        trial_df = res['trial_df']
        
        # 数据提取
        model_p = trial_df['prob_correct']   # 模型对正确类别的概率 (原始)
        human_acc = trial_df['human_correct'] # 被试是否正确 (0/1)
        
        # 被试数据平滑
        human_smooth = human_acc.rolling(window=16, min_periods=1).mean()
        
        ax.plot(trial_df['trial'], model_p, color='tab:blue', alpha=0.4, linewidth=1, label='Model Prob')
        ax.plot(trial_df['trial'], human_smooth, color='tab:red', linewidth=2, label='Human (MA-16)')
        
        ax.set_title(f'Subject {sub_id}')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle=':', alpha=0.6)
        
        if idx == 0: ax.legend(loc='lower right', fontsize='small')
        if idx >= n_subs - n_cols: ax.set_xlabel('Trial')
        if idx % n_cols == 0: ax.set_ylabel('Acc / Prob')

    for j in range(n_subs, len(axes_flat)): axes_flat[j].axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('plots/all_subjects_accuracy_comparison.png', dpi=150)
    plt.close()
    
    print("所有绘图工作完成！")

# ==========================================
# 主程序入口
# ==========================================

if __name__ == "__main__":
    
    # ---------------- 开关设置 ----------------
    # 设为 True 运行对应步骤，设为 False 跳过
    RUN_FITTING    = False   # 1. 拟合参数 (耗时，仅需运行一次)
    RUN_SIMULATION = True   # 2. 生成并保存模拟数据 (快，参数改变后需重新运行)
    RUN_PLOTTING   = True   # 3. 读取数据并画图 (调整绘图风格时只运行这个)
    # ----------------------------------------
    
    DATA_FILE = 'df_con1.csv'
    PARAM_FILE = 'fitted_params.csv'
    SIM_FILE = 'simulation_results.pkl' # 保存所有中间结果的二进制文件
    
    # 1. 拟合
    if RUN_FITTING:
        if os.path.exists(DATA_FILE):
            run_fitting(DATA_FILE, PARAM_FILE)
        else:
            print(f"文件不存在: {DATA_FILE}")

    # 2. 模拟
    if RUN_SIMULATION:
        if os.path.exists(DATA_FILE) and os.path.exists(PARAM_FILE):
            run_simulation(DATA_FILE, PARAM_FILE, SIM_FILE)
        else:
            print("无法运行模拟：缺少数据文件或参数文件。")

    # 3. 绘图
    if RUN_PLOTTING:
        if os.path.exists(SIM_FILE):
            run_plotting(DATA_FILE, SIM_FILE)
        else:
            print(f"无法运行绘图：缺少模拟结果文件 {SIM_FILE}，请先运行模拟步骤。")