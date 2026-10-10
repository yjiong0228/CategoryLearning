"""Small synthetic oral-readout demonstration; never fits human data."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import scipy
import yaml

from ...evaluation.oral.structure_0923 import encode_report
from ...model import ModelContext, StateModel
from ...model.oral_readout import OralReadout, OralReadoutParameters
from ...model.modules.readout.report_space import report_code
from ...model.modules.readout.choice import resolve_choice_readout_config, resolve_output_noise_config

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = Path('configs/exp123/oral_readout_0923_demo.yaml')


def run_demo(config_path: Path, output_dir: Path) -> dict:
    """Write a new report bundle with full synthetic inputs and source hashes."""
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text())
    output_dir.mkdir(parents=True, exist_ok=False)
    sources = set((ROOT / 'src/Bayesian_state').rglob('*.py'))
    sources.add(config_path)
    model_configs, live, illustration, parameters = {}, [], [], None
    with TemporaryDirectory(prefix='model0923_oral_demo_') as cache:
        for entry in config['models']:
            condition = entry['condition']
            path = ROOT / entry['path']
            sources.add(path)
            original = yaml.safe_load(path.read_text())
            resolved = deepcopy(original)
            resolved['partition']['kwargs'].update(
                similarity_n_samples=config['similarity_n_samples'], similarity_cache_dir=cache)
            resolved['modules']['perception_mod']['kwargs'] = {
                'features': 4, 'mean': [0.] * 4, 'std': [0.] * 4,
            }
            resolved['modules']['hypo_transitions_mod']['kwargs']['module_seed'] = config['transition_seed']
            model = StateModel(resolved, context=ModelContext(condition=condition))
            archived = deepcopy(resolved)
            archived['partition']['kwargs']['similarity_cache_dir'] = '<temporary directory>'
            model_configs[str(condition)] = archived
            current_parameters = OralReadoutParameters.from_config(original['oral_readout'])
            if parameters is None:
                parameters = current_parameters
            elif current_parameters != parameters:
                raise ValueError('demonstration requires shared oral parameters across conditions')
            if condition == config['illustration']['condition']:
                reader = OralReadout(model.partition_model.hypothesis_space, parameters,
                                     boundary_geometry=model.partition_model.boundary_geometry)
                for stimulus in config['illustration']['stimuli']:
                    mass, extra = reader.distribution(config['illustration']['hypothesis_index'],
                                                      config['illustration']['choice'], stimulus['values'])
                    examples = []
                    for item in config['illustration']['reports']:
                        encoded = encode_report(item['text'], tuple(config['feature_order']))
                        if encoded.status != 'coded':
                            raise ValueError(f"demo sentence is not completely coded: {item['text']}")
                        code = report_code(encoded.tokens)
                        probability = float(mass[reader.codes.index(code)]) if code in reader.codes else 0.
                        examples.append({**item, 'code': code, 'probability': probability})
                    illustration.append({**stimulus, 'vocabulary_size': len(reader.codes),
                                         'probability_sum': float(mass.sum()),
                                         'extra_event_probability': float(extra.sum()), 'reports': examples})
            choices, feedback = [], []
            choice_rng = np.random.default_rng(config['seed'] + condition)
            oral_rng = np.random.default_rng(config['seed'] + 100 + condition)
            if len(config['live_stimuli']) != len(entry['feedback']):
                raise ValueError('demo feedback must align with its synthetic stimuli')
            for trial, (stimulus, outcome) in enumerate(zip(config['live_stimuli'], entry['feedback']), start=1):
                model.begin_trial(np.asarray(stimulus, dtype=float))
                decision = model.predict_choice(
                    choices=choices, feedback=feedback,
                    choice_readout_config=resolve_choice_readout_config(None, model.engine_config),
                    output_noise_config=resolve_output_noise_config(None, model.engine_config),
                    rng=choice_rng, sticky_state={}, post_error_lapse_state=0.)
                choice = int(choice_rng.choice(model.n_cats, p=decision.observed_probabilities)) + 1
                oral = model.predict_oral(choice)
                code = oral.sample(oral_rng)
                posterior, _, _ = model.complete_trial(choice, outcome)
                choices.append(choice); feedback.append(outcome)
                live.append({'condition': condition, 'trial': trial, 'stimulus': stimulus,
                             'choice': choice, 'feedback': outcome,
                             'choice_probabilities': decision.observed_probabilities.tolist(),
                             'oral_probability_sum': float(oral.probabilities.sum()),
                             'oral_sample_code': code, 'oral_sample_probability': oral.probability(code),
                             'extra_event_probability': float(oral.extra_probabilities.sum()),
                             'hypotheses': list(oral.hypothesis_indices),
                             'rule_weights_given_choice': oral.rule_weights_given_choice.tolist(),
                             'posterior_sum_after_feedback': float(posterior.sum()),
                             'vocabulary_size': len(oral.codes)})
    if parameters is None or not illustration:
        raise ValueError('no matching model for the configured illustration')
    result = {'status': 'synthetic_uncalibrated_demonstration', 'config': config,
              'parameters': asdict(parameters), 'resolved_model_configs': model_configs,
              'illustration': illustration, 'live_trials': live,
              'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                              'scipy': scipy.__version__, 'pyyaml': yaml.__version__},
              'source_sha256': {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
                                hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
              'git_head_at_run': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    (output_dir / 'demonstration.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    labels = ' | '.join(item['label'] for item in illustration)
    lines = ['# Model 0923：oral readout 首版演示', '',
             '这是合成刺激与未校准参数的演示，没有拟合被试，也不作为拟合改善证据。', '',
             '同一内部规则在所选类别使用两个判断：“脖子长，且脖子长于尾巴”。口述可以说全、',
             '省略其中一项、用“躯干”命名长短边界，或者补充一个未用于该规则的部位描述。',
             '后者只是描述当前刺激，不能据此断定这个部位参与了分类。', '',
             '两幅合成刺激依次为 (脖子, 腿, 尾巴, 头)：' +
             '；'.join(f"{item['label']} {item['values']}" for item in illustration) + '。', '',
             f'| 口述示例 | 生成方式 | {labels} |',
             '|---|---|' + '---:|' * len(illustration)]
    for index, item in enumerate(illustration[0]['reports']):
        cells = ' | '.join(f"{stimulus['reports'][index]['probability']:.3%}" for stimulus in illustration)
        lines.append(f"| {item['text']} | {item['meaning']} | {cells} |")
    lines += ['', '表内各行为指定的完整口述事件，未列出所有事件，所以表列不会合计为 100%。',
              '概率以已确定这条规则和类别为前提；真实模型还会按本题选择所支持的规则加权。', '',
              '## 首版的边界', '',
              '- 每个核心判断独立地以 q 被说出；其余省略。',
              '- 最多补充一个单部位长短描述，均匀选自整个局部规则没有使用的部位。',
              '  补充内容取自本题同一份知觉；值恰好为 0.5 时不生成该部位的长短判断。',
              '- 躯干措辞沿用模糊命名概率，不改变行为边界或物理躯干 0.75。',
              '- 缺失、解析不确定的真实记录不自动当作 EMPTY 或 OTHER。',
              '- 标签学习与 S122 加权和机制继续跳过。没有增加认知规则数。', '',
              '共用演示参数：`' + json.dumps(asdict(parameters), ensure_ascii=False) + '`。',
              '它们都不是个人估计或心理测量值；补充发生在非 OTHER 分支，因此总补充概率为',
              '(1 − other_probability) × extra_probability（有可补充部位时）。', '',
              '## 三条件实际接口检查', '',
              '每个 condition 运行 3 个合成 trial，实际依次调用：', '',
              '`begin_trial → predict_choice → predict_oral(choice) → complete_trial(choice, feedback)`', '',
              '| Condition | 规则数 | 报告词表大小 | 最大口述概率和误差 |', '|---|---:|---:|---:|']
    for condition, resolved in model_configs.items():
        trials = [r for r in live if str(r['condition']) == condition]
        count = 29 if resolved['partition']['kwargs']['n_cats'] == 2 else 128
        error = max(abs(t['oral_probability_sum'] - 1) for t in trials)
        lines.append(f"| {condition} | {count} | {trials[0]['vocabulary_size']} | {error:.3g} |")
    lines += ['', 'condition 3 的三题实际经过 0、0.5、1 反馈。相似度只用 64 个采样点，知觉噪声置零，',
              '这些都是可复现演示设置，不能替代正式拟合。选择与口述抽样使用不同随机数流。',
              '本模块尚未自动接入行为 PF 的口述评分；这里只验证显式接口及生成行为。', '',
              '所有配置、逐题选择与口述、参数、环境版本和源码哈希见 [demonstration.json](demonstration.json)。',
              '源码哈希记录未提交代码在运行时的实际内容；git_head_at_run 仅是当时的基线提交。', '',
              '复现入口：`python -m src.Bayesian_state.workflows.analysis.demo_model_0923_oral',
              '--config-path configs/exp123/oral_readout_0923_demo.yaml --output-dir <新的输出目录>`。']
    (output_dir / 'README.md').write_text('\n'.join(lines) + '\n')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-path', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing outputs are never overwritten.')
    args = parser.parse_args()
    result = run_demo(args.config_path, args.output_dir)
    print(f"Saved {len(result['live_trials'])} synthetic trials and oral examples to {args.output_dir}")


if __name__ == '__main__':
    main()
