"""Audit symbolic body comparisons against task geometry, without model fits.

--oral-audit supplies the frozen coded reports; --output must be new. The task
sets body length to 1 and component length to 0.25+x, so equality with the body
occurs at x=0.75. The pre-existing bare-long/short midpoint 0.5 is shown as a
separate descriptive reference. Neither threshold is selected from these data.
All symbolic claims, including those inside partially parsed reports, remain.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from .model_0923_decision_basis_replay import ROOT, digest, read, write


def task_geometry(program: Path, transform: Path, coordinates: Path) -> dict:
    """Verify the implemented affine length map before applying its constants."""
    source = re.sub(r'\s+', '', program.read_text())
    if 'feature_range.body_length=1;' not in source:
        raise ValueError('Body geometry differs from the reviewed task')
    text = re.sub(r'\s+', '', transform.read_text())
    xy = re.sub(r'\s+', '', coordinates.read_text())
    for name in ('neck', 'head', 'leg', 'tail'):
        if f'feature_range.{name}_length=[feature_range.body_length/4,feature_range.body_length*5/4];' not in source:
            raise ValueError(f'Unexpected physical range for {name}')
        mapping = (f'feature_range.{name}_length(1)+normalized_features.{name}_length*'
                   f'(feature_range.{name}_length(2)-feature_range.{name}_length(1))')
        if mapping not in text or f'info.{name}_length.*cos(info.{name}_angle)' not in xy or f'info.{name}_length.*sin(info.{name}_angle)' not in xy:
            raise ValueError('Need the reviewed affine map and Euclidean segment geometry')
    return dict(body_length=1., feature_offset=.25, feature_scale=1., body_equality_normalized=.75,
                bare_long_short_reference=.5, definition='Euclidean segment length before common pixel scaling')


def run(audit: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError('Use a new body-reference audit destination')
    manifest = read(audit / 'manifest.json')
    if manifest['status'] != 'complete' or digest(audit / 'trials.csv') != manifest['artifacts']['trials.csv']:
        raise ValueError('Need unchanged complete oral audit')
    program = ROOT / 'exp_program/exp123/Task2_learning.m'
    transform = ROOT / 'exp_program/exp123/transform_to_real.m'
    coordinates = ROOT / 'exp_program/exp123/screen_xys.m'
    geometry = task_geometry(program, transform, coordinates)
    data_file = ROOT / 'data/exp123/processed/Task2_processed.csv'
    if digest(data_file) != manifest['input_sha256'][str(data_file.relative_to(ROOT))]:
        raise ValueError('Physical stimulus table changed since oral audit')
    oral = pd.read_csv(audit / 'trials.csv'); data = pd.read_csv(data_file)
    keys = ['iSub', 'iSession', 'iTrial']
    features = [f'feature{i}' for i in range(1, 5)]
    names = [f'feature{i}_name' for i in range(1, 5)]
    frame = oral.merge(data[keys + features + names], on=keys, how='left', validate='one_to_one',
                       suffixes=('', '_physical'), indicator=True)
    if not frame['_merge'].eq('both').all():
        raise ValueError('Oral identities missing from physical records')
    for name in names:
        np.testing.assert_array_equal(frame[name], frame[name + '_physical'])
    rows = []
    for record in frame.to_dict('records'):
        for token in json.loads(record['tokens']):
            if not token.startswith('body:'):
                continue
            _, dimension, direction = token.split(':')
            if direction not in ('lt', 'gt') or int(dimension) not in range(4):
                raise ValueError('Invalid symbolic body claim')
            value = float(record[features[int(dimension)]])
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid normalized physical feature')
            sign = 1 if direction == 'gt' else -1
            rows.append(dict(subject=record['iSub'], condition=record['condition'], session=record['iSession'],
                trial=record['iTrial'], ordinal_trial=record['ordinal_trial'], text=record['text'], scope=record['scope'],
                token=token, feature=record[names[int(dimension)]], direction=direction, normalized_feature=value,
                physical_length=geometry['feature_offset'] + geometry['feature_scale'] * value,
                agrees_with_physical_body=bool(sign * (value - .75) > 0),
                agrees_with_bare_midpoint=bool(sign * (value - .5) > 0)))
    claims = pd.DataFrame(rows)
    if claims.empty:
        raise ValueError('No coded symbolic body claims')
    summary = claims.groupby(['subject', 'scope'], as_index=False).agg(
        claims=('token', 'size'), reports=('ordinal_trial', 'nunique'),
        physical_agreement=('agrees_with_physical_body', 'mean'),
        midpoint_agreement=('agrees_with_bare_midpoint', 'mean'))
    output.mkdir(parents=True)
    claims.to_csv(output / 'claims.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    (output / 'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8">'
        '<title>躯干参照的物理核对</title><h1>躯干参照与报告方向</h1>'
        '<p>图形程序定义部位线段长度为0.25+x、躯干长度为1，因此物理等长点为x=0.75。'
        '0.5来自原有普通长短编码约定。这两个参照在本次核对前固定。</p>'
        '<p>表中一致率检验报告方向与屏幕物理长度的关系；心理知觉、表达近似与转录误差均可能影响一致率。'
        '本审查未选择心理阈值，未修改口述编码或任何模型评分。分母为已识别的比较判断，部分报告包含多个判断。</p>'
        + summary.to_html(index=False, float_format=lambda x: f'{x:.4f}')
        + '<p><a href="claims.csv">全部判断及物理值</a></p></html>')
    inputs = [program, transform, coordinates, data_file, audit / 'trials.csv', audit / 'manifest.json']
    write(output / 'manifest.json', dict(status='complete', geometry=geometry, claims=len(claims),
        inputs_sha256={str(p.resolve()): digest(p) for p in inputs}, source_sha256=digest(Path(__file__)),
        artifacts={p.name: digest(p) for p in output.iterdir()},
        data_use='oral_and_physical_stimulus_audit; choices_feedback_model_states_not_used',
        interpretation='descriptive reference calibration; original symbolic coding retained'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oral-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); run(args.oral_audit, args.output)
