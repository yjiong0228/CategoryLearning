import os
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 设定项目根目录
project_root = Path(os.getcwd())

from src.Bayesian import *
from src.Bayesian.problems.config import config_fgt
from src.Bayesian.problems import *
from src.Bayesian.utils.optimizer import Optimizer
from src.Bayesian.utils.model_evaluation import ModelEval
from src.Bayesian.utils.stream import StreamList


model_name = 'M7_PMH'
result_path = Path(project_root) / 'results' / 'Model_results_new' 
results = joblib.load(result_path / f'{model_name}.joblib')

def compute_single_entry(iSub, p_bin, info, oral_hit):
    p_lower, p_upper = p_bin
    path, n = info['raw_step_results']
    sample_errors = np.array(info['sample_errors'])
    k_special = 0 if oral_hit['condition'] == 1 else 42
    oral_rolling = np.array(oral_hit['rolling_hits'], dtype=float)
    slist = StreamList(path, n)

    sorted_indices = np.argsort(sample_errors)
    low_idx = int(n * p_lower / 100)
    high_idx = int(n * p_upper / 100)
    top_idxs = sorted_indices[low_idx:high_idx]

    acc_errors = [sample_errors[int(i)] for i in top_idxs]

    k_oral_errors = []
    for i in top_idxs:
        traj = slist[int(i)]
        posts = []
        for sr in traj:
            post = sr['hypo_details'].get(k_special, {}).get('post_max', 0.0)
            try:
                posts.append(float(post))
            except Exception:
                posts.append(0.0)
        posts = pd.Series(posts).rolling(window=16, min_periods=16).mean().to_numpy()
        min_len = min(len(posts), len(oral_rolling))
        valid = ~np.isnan(posts[:min_len])
        if valid.sum() > 0:
            err = np.mean(np.abs(posts[:min_len][valid] - oral_rolling[:min_len][valid]))
        else:
            err = np.nan
        k_oral_errors.append(err)

    return {
        'Subject': iSub,
        'Percentile_Bin': f"{p_lower}-{p_upper}",
        'Accuracy_Error': np.mean(acc_errors),
        'KOral_Error': np.nanmean(k_oral_errors)
    }


def compute_errors_fine_parallel_with_progress(results, oral_hypo_hits,
                                               bin_edges=range(0, 101, 5),
                                               n_jobs=-1):

    tasks, percentile_bins = [], list(zip(bin_edges[:-1], bin_edges[1:]))
    for iSub, info in results.items():
        if iSub not in oral_hypo_hits or 'raw_step_results' not in info:
            continue
        for p_bin in percentile_bins:
            tasks.append((iSub, p_bin, info, oral_hypo_hits[iSub]))

    with tqdm_joblib(tqdm(total=len(tasks),
                          desc="Computing errors",
                          ncols=80)):

        out = Parallel(n_jobs=n_jobs)(
            delayed(compute_single_entry)(iSub, p_bin, info, oral_hit)
            for iSub, p_bin, info, oral_hit in tasks
        )

    return pd.DataFrame([r for r in out if r is not None])


from src.Bayesian.utils.oral_process import Oral_to_coordinate
oral_to_coordinate = Oral_to_coordinate()

processed_path = Path(project_root) / 'data' / 'processed'
learning_data = pd.read_csv(processed_path / 'Task2_processed.csv')
oral_hypo_hits = oral_to_coordinate.get_oral_hypo_hits(learning_data)


df_eval = compute_errors_fine_parallel_with_progress(results, oral_hypo_hits, n_jobs=120)