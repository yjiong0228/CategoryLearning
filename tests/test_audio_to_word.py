import csv

import pytest

from src.audio_to_word import build_prompt, discover_audio, process_folder


def test_numeric_order_and_three_column_roundtrip(tmp_path):
    folder = tmp_path / 'audio'
    folder.mkdir()
    for name in ('10.wav', '2.wav', '1.wav'):
        (folder / name).touch()
    output = tmp_path / 'rec.csv'
    process_folder(folder, output, 2, lambda p: '头长，\n可能是腿短。')
    with output.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    assert rows[0] == ['iSession', 'iTrial', 'text']
    assert [row[1] for row in rows[1:]] == ['1', '2', '10']
    assert all(row[0] == '2' and row[2] == '头长，\n可能是腿短。' for row in rows[1:])
    with pytest.raises(ValueError, match='输出已存在'):
        process_folder(folder, output, 2, lambda p: 'changed')


def test_failure_and_resume(tmp_path):
    for name in ('1.wav', '2.wav'):
        (tmp_path / name).touch()
    output = tmp_path / 'rec.csv'
    with pytest.raises(RuntimeError, match='2.wav'):
        process_folder(tmp_path, output, 1, lambda p: '第一条' if p.stem == '1' else '', attempts=1)
    called = []
    def transcribe(path):
        called.append(path.name)
        return '第二条'
    process_folder(tmp_path, output, 1, transcribe, resume=True)
    assert called == ['2.wav']
    assert '第一条' in output.read_text(encoding='utf-8-sig')
    assert 'Error' not in output.read_text(encoding='utf-8-sig')


def test_ambiguous_trial_rejected(tmp_path):
    (tmp_path / '1.wav').touch()
    (tmp_path / '01.mp3').touch()
    with pytest.raises(ValueError, match='重复试次'):
        discover_audio(tmp_path)


def test_unknown_filename_rejected(tmp_path):
    (tmp_path / 'subject_12.wav').touch()
    with pytest.raises(ValueError, match='无法确定试次'):
        discover_audio(tmp_path)


def test_prompt_has_no_assumed_stimulus():
    assert '动物' not in build_prompt()
    assert '颜色' in build_prompt('刺激包括颜色。')
    assert '不要补全未提及的特征' in build_prompt('刺激包括颜色。')
