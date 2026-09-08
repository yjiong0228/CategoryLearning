"""Traceable descriptive definitions and figure legend for the current revision."""
import pandas as pd


def readout(trials, subjects, selected, config, audit, rt_summary):
    lines = ['# Fig1 v10：口述编码复用与非 oral 行为', '',
             '## 本轮变化', '',
             '- 删除新增的 add_feature_use_columns。Preprocessor_B.process 内直接调用 Recording_Processor_Center.process_use，随后用 FEATURE_NAME_TO_PART 将部位顺序重排至 F1–F4。没有新增文本解析规则。',
             '- *_use 忠实保留 process_use 的二元输出，缺失文本为全 0；图中仍用原始 text 区分缺失报告（灰色）与有文本但未提及（白色）。',
             '- a 保留简洁版，b 去除键盘图标，c 使用 task 配色短线；d 去掉覆盖率和无关特征两个图，保留特征数并新增反应时。全部 96 人保留，9 个示例不重新挑选。',
             '- 新增 figS4_nonoral_candidates.png，按 task 分列展示反应时、反馈后的反应时和刺激边界距离。均为探索性描述，无显著性筛选或模型拟合。', '',
             '## d：报告范围与作答速度', '',
             '首末各 64 个记录试次，不重叠，至少需 128 个试次。Task1 的 S105 只有 64 个试次，保留于行为图但不构成首末配对。',
             '左：每期 choice RT 的中位数，所有正且有限 RT 均保留，不剪裁长尾；纵轴为对数，标签为秒。右：每期明确提及的特征数均值，仅使用至少提及一个特征的报告。每条线代表一个被试。', '',
             '| Task | 配对人数 | 首期 RT 中位数（s） | 末期 RT 中位数（s） | 个体末/首比值中位数 | 变快人数 |',
             '|---|---:|---:|---:|---:|---:|']
    for task in config['tasks']:
        sub=rt_summary[rt_summary.condition.eq(task['condition'])].dropna(subset=['first_median_rt','last_median_rt'])
        lines.append(f"| {task['task']} | {len(sub)} | {sub.first_median_rt.median():.3f} | {sub.last_median_rt.median():.3f} | {(sub.last_median_rt/sub.first_median_rt).median():.3f} | {(sub.last_median_rt<sub.first_median_rt).sum()} |")
    lines += ['', '表中先计算每人的期内中位数，再汇总人群中位数；末/首比值同样先在个体内计算。各任务普遍变快，但首末阶段对应的总训练量不同，不能直接归因为任务条件的因果作用。', '',
              '## 非 oral 候选图', '',
              '1. **速度变化**：同 d 右图，另以灰虚线呈现只保留正确反应的敏感性检查。分母按人/时期保存在 nonoral_subject_summary.csv；不将快反应直接当作掌握规则。',
              '2. **反馈后的速度调整**：只配对同一人、session、block 内 iTrial 连续的前后试次。每个 RT 先取对数、减去本 block 的对数 RT 中位数，再按上一试次反馈分组取中位数并指数还原。1 表示本 block 的典型速度。Task2 单独展示 0、0.5、1，其他两任务为 0、1。',
              '   所有可用被试保留；图用共同的对数轴显示，包括 S117 在错误后约 8.10 倍的个体值（仅 6 个有效前序错误试次）。有效数量逐人逐反馈保存，不能把小样本极值当作稳定效应。分块中心化仅减少缓慢漂移，未匹配刺激难度、当前反应正确性和其他历史，因此不称因果反馈效应。',
              '3. **刺激难度**：使用已验证的客观类别结构。Task1 的边界距离为 |F1−0.5|；Task2/3 为该距离与当前分支特征到 0.5 的距离中的较小值（F1≤0.5 使用 F2，否则 F3）。区间为 [0,.05]、(.05,.10]、(.10,.20]、(.20,.50]。',
              '   每人每区间计算准确率，粗线为等权被试均值，细线展示所有个体；不把 trial 当独立被试。没有剔除模糊刺激。各 task 的机会水平不同，且距离分布与学习阶段可能相关，因此只描述距离—正确率关系，不据此断言知觉噪声或机制差异。', '',
              '## 审计与复现', '',
              f"- {audit['rows']:,} 个试次、{audit['subjects']} 名被试；重复键 {audit['duplicate_keys']}；无行为剔除。",
              f"- 类别来源差异 {audit['category_source_mismatch']}；反馈不一致 {audit['label_feedback_mismatch']}；任务规则差异 {audit['task_rule_mismatch']}。",
              f"- 缺失文本 {audit['missing_text']}；process_use 识别到特征的报告 {audit['recognized_reports']}。",
              f"- 无效 RT {int(rt_summary.invalid_rt_n.sum())}。所有 RT 检查仅影响相应描述量，不剔除行为记录。",
              '- outputs/fig1/process_use_update_v2 中保留更新前 CSV 与逐字段审计。相对 v7，13 个实质提及值纠正，另有 664×4 个缺失字段由留空恢复为 0；原 22 列完全不变，S319 类别修正保留。',
              '- nonoral_subject_summary / nonoral_periods / nonoral_feedback_rt / nonoral_boundary / nonoral_trial_source.csv 保存来源、分母、分组和计算值。原有 oral 指标仍保存于表中，但不再全部占用主图。',
              '- source_snapshot 和 manifest 记录输入、代码、配置及输出；仅 PNG，论文 figures 目录尚未放入未确认图。', '',
              '## 图注草案', '',
              '**Fig. 1 | Task structure, learning trajectories, reported features and response speed.** '
              '**a**, A stimulus with four continuously varying lengths and three category-learning tasks. '
              'Task 1 is condition 1; Task 2 is condition 3, with hierarchical partial feedback; Task 3 is condition 2. '
              '**b**, Schematic trial procedure, including a verbal report before feedback. '
              '**c**, Trailing 32-trial accuracy for all 32 participants per task, ordered by recording duration, '
              'with recording coverage and nine fixed examples. F1–F4 report rows are aligned using each participant’s '
              'feature assignment. Task-colored short marks indicate mentions from the existing process_use encoder; white indicates '
              'no mention in available text, and gray indicates missing text. Dashed accuracy lines mark uniform-choice '
              'chance and dotted vertical lines mark session boundaries. '
              '**d**, Paired first/last disjoint 64-trial periods: mean named-feature count among reports with at least '
              'one recognized feature (right), and median choice response time on a logarithmic axis (left). Each line '
              'represents one participant; 31, 32 and 32 participants contribute pairs. Full correctness is feedback=1; '
              'partial credit is not counted as fully correct. Results are descriptive and do not assign latent learner types.', '']
    return '\n'.join(lines)
