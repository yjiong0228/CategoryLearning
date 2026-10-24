"""Build a reproducible development report from audited trajectory diagnostics.

Use --analysis RESULTS --figures PNG_DIRECTORY --output NEW. This summarizes
all parameter points, preserves failed phenomena and computes paired simulation
contrasts. It never selects a new fitted winner from evaluation behavior.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ..analysis.model_1022_belief_summary import digest, jackknife_se, read
from ..analysis.model_1024_trajectory_diagnosis import (
    checked_new_paths, feature_matrix, probability_record, write)
from ...metrics.autonomous_fit import energy_u

ROOT = Path(__file__).resolve().parents[4]


def table(frame: pd.DataFrame, digits: int = 4) -> str:
    """Small Markdown renderer that retains explicit missing cells."""
    def cell(value):
        if isinstance(value, (float, np.floating)):
            return f'{value:.{digits}f}' if np.isfinite(value) else '—'
        return str(value)
    return '\n'.join(['| ' + ' | '.join(map(str, frame.columns)) + ' |',
                      '| ' + ' | '.join(['---']*len(frame.columns)) + ' |'] +
                     ['| ' + ' | '.join(cell(v) for v in row) + ' |'
                      for row in frame.itertuples(index=False, name=None)])


def paired_contrasts(analysis: Path, complete: dict, paths: pd.DataFrame) -> pd.DataFrame:
    generation = Path(complete['generation'])
    protocol = read(generation/'protocol.json')
    plan = read(Path(protocol['nomination'])/'nomination.json')
    rows = []
    for sid in complete['policy']['sensitivity']['subjects']:
        for mode in complete['policy']['sensitivity']['modes']:
            nominees = [r for r in plan['rows'] if r['subject'] == sid and r['mode'] == mode]
            folder = Path(nominees[0]['template']['folder'])
            with np.load(folder/'sealed_test.npz') as z:
                categories, observed = z['categories'], z['choices']
            meta = read(folder/'prepared.json')
            if digest(folder/'sealed_test.npz') != meta['array_sha256']['sealed_test.npz']:
                raise ValueError('Evaluation observations changed')
            cut = meta['training_trials']
            with np.load(folder/'training.npz') as z:
                full_categories = np.concatenate([z['categories'], categories])
            ncat = 2 if meta['condition'] == 1 else 4
            empirical = feature_matrix(observed[None, :], categories, ncat, complete['policy'])[0]
            own = paths[(paths.subject == sid) & (paths['mode'] == mode) & (paths.batch == 'independent')]
            baseline = own[own.role == 'nominee'].sort_values('seed')
            bank = {}
            for r in nominees:
                raw = checked_new_paths(protocol['jobs'], r, full_categories)
                index = np.argsort(raw['seeds'])
                np.testing.assert_array_equal(raw['seeds'][index], baseline.seed.to_numpy())
                bank[r['role']] = feature_matrix(raw['choices'][index, cut:], categories, ncat, complete['policy'])
            x0 = bank['nominee']
            for r in nominees[1:]:
                alt = own[own.role == r['role']].sort_values('seed')
                np.testing.assert_array_equal(alt.seed, baseline.seed)
                x = bank[r['role']]
                deleted = np.array([energy_u(np.delete(x, i, 0), empirical)['value']-
                                    energy_u(np.delete(x0, i, 0), empirical)['value'] for i in range(len(x))])
                row = dict(subject=sid, mode=mode, role=r['role'], candidate=r['candidate']['id'],
                    late_energy_delta=energy_u(x, empirical)['value']-energy_u(x0, empirical)['value'],
                    late_energy_delta_mcse=jackknife_se(deleted))
                for key in ('late_accuracy', 'late_target_present', 'late_target_contribution',
                            'jump_and_maintenance', 'low_and_bias_adjusted'):
                    delta = alt[key].to_numpy(dtype=float)-baseline[key].to_numpy(dtype=float)
                    row[key+'_delta'] = float(delta.mean())
                    row[key+'_delta_mcse'] = float(delta.std(ddof=1)/np.sqrt(len(delta)))
                rows.append(row)
    return pd.DataFrame(rows)


def run(analysis: Path, figures: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new report directory')
    complete = read(analysis/'completion.json')
    if complete['status'] != 'complete' or complete['policy']['protocol'] != 'model1024_trajectory_diagnosis_v1':
        raise ValueError('This report requires the registered completed diagnosis')
    for name, sha in complete['artifacts_sha256'].items():
        if digest(analysis/name) != sha:
            raise ValueError('Analysis artifact changed: '+name)
    figure_meta = read(figures/'provenance.json')
    if figure_meta['completion_sha256'] != digest(analysis/'completion.json'):
        raise ValueError('Figures refer to another analysis')
    frames = {n: pd.read_csv(analysis/(n+'.csv')) for n in
              ('paths', 'summary', 'events', 'clusters', 'observed', 'stability')}
    paths, summary, events, clusters, obs, stability = [frames[n] for n in frames]
    generation = Path(complete['generation'])
    generation_protocol = read(generation/'protocol.json')
    nomination = Path(generation_protocol['nomination'])
    train = pd.read_csv(nomination/'training_candidates.csv')
    sensitivity = summary[summary.batch == 'independent'].merge(train, on=['subject', 'mode', 'candidate'])
    contrasts = paired_contrasts(analysis, complete, paths)
    output.mkdir(parents=True)
    contrasts.to_csv(output/'paired_parameter_contrasts.csv', index=False)
    sensitivity.to_csv(output/'parameter_sensitivity.csv', index=False)
    group_keys = ['subject', 'mode', 'batch', 'role', 'candidate', 'cluster']
    cluster_process = paths.groupby(group_keys, sort=True).agg(
        paths=('seed', 'size'), late_accuracy=('late_accuracy', 'mean'),
        ever_target=('ever_target', 'mean'), target_present=('late_target_present', 'mean'),
        target_contribution=('late_target_contribution', 'mean'),
        target_precision=('late_target_precision', 'mean'),
        precision_paths=('late_target_precision', 'count'),
        entry_paths=('first_target_trial', 'count'), first_entry=('first_target_trial', 'mean'),
        absent_after_entry=('absent_after_entry', 'mean')).reset_index()
    cluster_process.to_csv(output/'cluster_process.csv', index=False)
    error_rows = []
    for key, group in paths[paths['mode'] != 'MIX'].groupby(group_keys[:-1], sort=True):
        row = dict(zip(group_keys[:-1], key))
        total = int(group.total_errors.sum())
        row['errors'] = total
        for name in ('error_absent', 'error_present_other', 'error_correct_executed'):
            row[name] = int(group[name].sum())
            row[name+'_fraction'] = float(group[name].sum()/total) if total else np.nan
        assert sum(row[k] for k in ('error_absent', 'error_present_other', 'error_correct_executed')) == total
        for name in ('absent', 'present_other', 'correct_executed'):
            weights = group['trials_'+name]
            row['trials_'+name] = int(weights.sum())
            row['pooled_correct_probability_'+name] = float(
                (group['correct_probability_'+name].fillna(0)*weights).sum()/weights.sum()) if weights.sum() else np.nan
        error_rows.append(row)
    pd.DataFrame(error_rows).to_csv(output/'generating_error_sources.csv', index=False)
    links = lambda path: os.path.relpath(path, output)
    original = summary[summary.batch == 'original']
    overview = obs[['subject', 'late_accuracy']].rename(columns={'late_accuracy': '真人后段'})
    for mode in ('MIX', 'MAP', 'STICKY'):
        overview = overview.merge(original[original['mode'] == mode][['subject', 'late_accuracy']].rename(
            columns={'late_accuracy': mode}), on='subject')
    human_clusters = clusters[(clusters.clusters == 3) & (clusters.cluster == clusters.observed_cluster) &
                              (clusters.batch == 'original') & (clusters.split == 'evaluation')]
    cluster_table = human_clusters[human_clusters['mode'] == 'MIX'][['subject', 'observed_cluster',
        'observed_distance', 'reference_radius90', 'observed_distance_percentile']].copy()
    cluster_table['observed_cluster'] += 1
    cluster_table = cluster_table.merge(stability[stability.clusters == 3][['subject', 'ari']], on='subject')
    for mode in ('MIX', 'MAP', 'STICKY'):
        cluster_table = cluster_table.merge(human_clusters[human_clusters['mode'] == mode][['subject', 'probability']].rename(
            columns={'probability': mode+'_mass'}), on='subject')
    focus_events = events[((events.subject.isin([129, 229])) & (events.event == 'jump_and_maintenance')) |
                         ((events.subject == 317) & (events.event == 'low_and_bias_adjusted'))]
    event_table = focus_events[['subject', 'mode', 'role', 'batch', 'count', 'paths', 'low', 'high']]
    zero_bound = probability_record(np.zeros(128, bool))['high']
    lines = [
        '# Model 1024：完整自主路径与参数敏感性诊断', '',
        '完成日期：2026-10-25。范围为exp123九名开发被试，三种作答机制。'
        'MIX按规则信念加权选择；MAP由最高信念规则作答；STICKY每题执行一条规则，并具有跨题维持倾向。', '',
        '## 主要发现', '',
        '路径类别可以描述模型生成的多种学习形态，但129、229和317仍远离其最近类别的中心。'
        '训练评分接近的参数能明显改变后段生成表现和内部过程；本次定向模拟仍未补足129、229的最大陡升与保持，'
        '317的低表现也只在少数路径中出现。研究据此保留参数竞争解释，并将生成表现与状态准确性分别验收。', '',
        '## 本次实际完成的工作', '',
        f'- 复用全部{complete["original_paths"]}条原自主路径；新增{complete["independent_paths"]}条完整路径，'
        '共18个参数点，每点128条。新模拟复用共享认知机制、任务刺激与自主反馈。',
        '- 对九名被试各建立三机制共用的路径类别。每机制前64条建类、后64条检查；三类为主分析，两类与四类检查敏感性。',
        '- 对129、229、317的MIX和持续执行各保留原提名，再按训练NLL与网格距离选两点；原提名继续保留。',
        '- 保存全部类别概率、完整路径特征、几何正确规则状态及错误来源，计算配对种子下的参数差异。', '',
        '参数资格只使用原训练确认记录。NLL为每题负对数似然，值越低，模型赋给实际选择的概率越高。'
        '0.02 NLL门槛是工作容忍量；部分候选差值明显大于数值标准误，'
        '因此候选集合体现指定容忍范围内的敏感性。候选没有后验权重。真人后段此前已被查看，继续作为开发评价。'
        '口述未进入本次选择或评分，新种子仅检验模拟稳定性。', '',
        f'协议：[分析定义与预算]({links(ROOT/"src/Bayesian_state/docs/model_architecture/model_1024_trajectory_diagnosis.md")})。'
        f'全部表：[数值结果]({links(analysis)})。', '',
        '## 完整分布中的后段表现', '',
        '表中模型值是全部128条从任务起点生成的路径在真人冻结后段的平均正确率。'
        '全部九人在三种机制下的模型均值均低于其真人值；部分真人值仍落在生成分布之内。', '',
        table(overview), '',
        f'![九人完整路径分布]({links(figures/"nine_subject_path_distribution.png")})', '',
        '各点是一条完整路径的后段正确率，菱形为中位数，竖线为5%—95%分位区间。'
        '这些区间描述固定参数下的过程变异；真人黑线的位置同时受到参数估计和机制的影响。', '',
        '## 最近类别的概率质量与相似程度', '',
        '下表列出真人最近类别、真人到中心的距离、该类独立检查路径距离的90%分位数，以及真人距离的经验百分位。'
        '三机制共用每人一套中心，因此这四项距离量相同；最后三列分别是各机制64条检查路径中的该类概率。'
        'ARI比较前后两批重新建类的一致性。', '', table(cluster_table), '',
        '129的最近类别有约17%—23%的原检查路径，真人距离处于该类检查路径的100%经验百分位；229约为98%，'
        '317也为100%。最近类别可以相当常见，同时仍与真人明显有差距。'
        '108、110、209、309在当前特征中的位置相对居中；这说明全分布检查能够区分不同个案的缺口。', '',
        '三类ARI在0.40—0.89之间，129为0.40，三分类的细节对模拟批次较敏感。两类ARI总体较高，'
        '约0.77—0.99；这提示粗略分组较稳定，当前三类标签保留描述用途。', '',
        '## 陡升、保持与低表现的联合事件', '',
        '129、229的事件是：在保留完整随访的位置中选出最大相邻32题改善，该改善及其随后第二个32题窗口的'
        '正确率均至少达到真人值。129的两个参照值为0.46875和1，229为0.65625和0.9375。'
        '位置并列取最早，全部生成路径使用相同搜索规则。它评价最大改善所在片段及其保持。', '',
        '317的事件是：最低128题窗口正确率及其相对类别—反应边际独立基准的差值均不高于真人值。'
        '真人分别为0.14844和−0.01404，边际基准约0.16248；均匀猜测基准为0.25。'
        '该差别说明反应偏好需要单独考虑。窗口经过选择，比例用于描述个案位置。', '',
        table(event_table), '',
        f'0/128的95% Wilson蒙特卡洛区间上界约为{zero_bound:.2%}；有限模拟保留小概率事件的不确定性。'
        '不同参数点形成不同预测分布，表中分母分别保留，所有参数点合并后不能作为一组同分布样本解释。', '',
        '## 训练相近参数可以改变哪些解释', '',
        table(sensitivity[['subject', 'mode', 'role', 'training_nll_delta', 'delta_mcse', 'late_accuracy',
                           'late_energy', 'full_energy', 'late_target_present', 'late_target_contribution',
                           'late_target_precision']]), '',
        'Energy在规定的块正确率与类别条件反应频率特征中计算，越低表示全分布预测越接近真人。'
        '精度列先在每条路径内按正确规则信念加权，再对末段有正确规则信念的路径取平均；'
        '其参与路径数见类别过程表。其余概率量对全部路径平均。', '',
        table(contrasts[['subject', 'mode', 'role', 'late_accuracy_delta', 'late_accuracy_delta_mcse',
                         'late_energy_delta', 'late_energy_delta_mcse']]), '',
        '配对表比较同一组新种子下的候选与原提名。标准误衡量模拟计算波动；个体参数及真人观测不确定性另行保留。', '',
        '- **129：精度与发现有不同影响。** MIX候选1的训练NLL差约0.0011，末段在场比例约0.382，'
        '与原点0.383接近；条件精度从约6.62升至18.29，平均后段正确率从0.690升至0.759。'
        '达到真人后段正确率的路径由2/128增至42/128。该候选仍没有达到预定陡升—保持联合事件。'
        '多个参数同时变化，因此精度是应进一步隔离检验的过程线索。',
        '- **229：规则在场与判断质量共同限制生成。** MIX候选2的训练NLL差约0.0013，末段在场比例'
        '从0.444升至0.902，贡献从0.422升至0.850，正确率从0.581升至0.707；达到真人后段正确率'
        '的路径由16/128增至44/128。候选1也提高在场比例，却因贡献和判断质量不同而降低正确率。',
        '- **317：平均后段成绩会掩盖完整路径的差距。** MIX两个候选提高后段正确率，但完整序列Energy'
        '从0.136升至约0.162、0.165。持续执行候选2的在场比例从0.440降至0.176，后段成绩从0.600'
        '降至0.441。训练接近参数仍支持很不同的学习过程；较好的后段均值不足以确定整条路径解释。', '',
        f'![参数与内部过程]({links(figures/"parameter_process_sensitivity.png")})', '',
        '持续执行的原提名在129和317具有正维持参数；相近候选包含维持参数为零的点。'
        '这些点同时改变搜索、信念或精度参数，当前比较无法给维持参数分配独立因果效应。'
        '离散执行路径的三个错误来源已逐路径核对相加等于全部错误；生成路径中的错误分类与真人状态恢复准确性分别评价。', '',
        '## 对下一步的具体决策', '',
        '第一优先是让参数估计同时面对训练内的选择预测和自主分布约束，并保留训练表现相近的候选。'
        '此前已有训练生成约束流程；现在可将其系统接到最新执行比较，采用训练内验证选择权重，冻结后再评价后段。'
        '本次找到的后段改善点继续作为开发诊断，不能据其后段成绩直接替换最终估计。', '',
        '随后做最小的过程干预：从同一训练前段状态续跑，分别隔离精度校准与正确规则发现/保持。'
        '129优先检查正确规则在场后的判断精度和稳定高表现，229优先检查发现/保持与精度的联合作用；'
        '317保留反应偏好和低表现形态作为额外约束。每次机制比较重新拟合规定参数，并同时检查状态恢复。', '',
        '执行模块是否需要更复杂的试次变化，继续由冻结预测增益和可恢复性共同判断。当前执行状态恢复中，'
        '“正确规则在场但执行其他规则”的60次模拟错误在重新拟合后正确分类0次、真实参数重建2次。'
        '本次增加了生成分布与参数敏感性证据；口述的外部状态检验及执行状态可识别性仍是后续核心验收。', '',
        '## 图与数据入口', '',
    ]
    for sid in (129, 229, 317):
        lines += [f'![S{sid}完整代表路径与概率]({links(figures/f"subject_{sid}_path_shapes.png")})', '']
    lines += [
        '代表路径是各类中新模拟路径的medoid，即到同类其他路径平均距离最小的完整路径。'
        '每个代表都保留自己的历史。原检查批次与新批次的类别比例同时显示，误差线为95% Wilson区间。', '',
        '- `paired_parameter_contrasts.csv`：新种子配对的行为与过程差异。',
        '- `parameter_sensitivity.csv`：18点全部训练与生成指标。',
        '- `cluster_process.csv`：每类的发生数、发现、在场、贡献、精度及有效分母。',
        '- `generating_error_sources.csv`：离散执行的错误分解与按试次数加权的条件正确概率。',
        f'- [逐路径、逐类及评分原表]({links(analysis)})。', '',
        '## 复现与限制', '',
        '从仓库根目录依次执行，各输出使用新目录：', '', '```bash',
        'python -m src.Bayesian_state.workflows.analysis.model_1024_trajectory_diagnosis --phase nominate --config configs/shared/model1024_trajectory_diagnosis.yaml --output NEW_NOMINATION',
        'python -m src.Bayesian_state.workflows.analysis.model_1024_trajectory_diagnosis --phase generate --config configs/shared/model1024_trajectory_diagnosis.yaml --nomination NEW_NOMINATION --output NEW_GENERATION',
        'python -m src.Bayesian_state.workflows.analysis.model_1024_trajectory_diagnosis --phase analyze --config configs/shared/model1024_trajectory_diagnosis.yaml --generation NEW_GENERATION --output NEW_ANALYSIS',
        'python -m CategoryLearning_codes.figures.fig3.model1024_trajectory_diagnosis --analysis NEW_ANALYSIS --output NEW_FIGURES',
        'python -m src.Bayesian_state.workflows.reports.model_1024_trajectory_report --analysis NEW_ANALYSIS --figures NEW_FIGURES --output NEW_REPORT',
        '```', '',
        '首次数值分析v1因相对路径记录错误在读取路径前结束，记录保留；修复路径规范化后v2完成。'
        '图v1审阅发现图例覆盖低表现片段，v2将图例移至绘图区外，所有数值及路径选择保持不变。'
        '验收记录见`validation.json`。', '',
        '固定三类可以人为分割连续形态，ARI提供有限的批次稳定性信息。特征维数有限，真人距离百分位来自'
        '三机制合并检查样本，包含共享种子的依赖。128条路径对小概率形态的分辨率有限。参数点来自既有搜索，'
        '工作容忍门槛及有限候选覆盖进一步限制一般性。内部状态均指模型已知生成状态，真人心理过程仍需恢复和口述检验。', '',
        '本轮完成完整分布与参数敏感性诊断。九名个案的独立人群推广、完整参数后验和新增机制的有效性尚待检验。', '',
    ]
    (output/'report.md').write_text('\n'.join(lines))
    write(output/'completion.json', dict(status='complete', numerical_diagnosis_complete=True,
        independent_empirical_validation=False, mechanism_changed=False,
        source_sha256=digest(Path(__file__)), analysis=str(analysis.resolve()),
        analysis_sha256=digest(analysis/'completion.json'), figures=str(figures.resolve()),
        files_sha256={p.name: digest(p) for p in output.iterdir() if p.is_file()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('analysis', 'figures', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.analysis, args.figures, args.output)
