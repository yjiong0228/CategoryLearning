"""Feature involvement from shared oral semantics and direct mentions."""

import pandas as pd
import pytest

from src.oral_coding import Recording_Processor_Center


@pytest.mark.parametrize(
    "text, expected",
    [
        ("脖子长，其他短", [1, 1, 1, 1]),
        ("头和脖子长，其余短", [1, 1, 1, 1]),
        ("尾长，剩下三个短", [1, 1, 1, 1]),
        ("脖长，其它都短", [1, 1, 1, 1]),
        ("四个部位都比较长", [1, 1, 1, 1]),
        ("所有部位一样长", [1, 1, 1, 1]),
        ("除了脖子，其他都短", [1, 1, 1, 1]),
        ("除了脖子其他一样", [1, 1, 1, 1]),
        ("只有脖子长", [1, 1, 1, 1]),
        ("脖子最长", [1, 1, 1, 1]),
        ("尾巴最短", [1, 1, 1, 1]),
        ("脖子比其他部位长", [1, 1, 1, 1]),
        ("脖子在头和脖子中最长", [1, 1, 0, 0]),
        ("头和脖子都短", [1, 1, 0, 0]),
        ("脖子比头长", [1, 1, 0, 0]),
        ("脖子很长", [1, 0, 0, 0]),
        ("脖子不长", [1, 0, 0, 0]),
        ("尾巴不知道", [0, 0, 0, 1]),
        ("脖子长于身体", [1, 0, 0, 0]),
        ("脖子达到最长的长度", [1, 0, 0, 0]),
        ("从长到短为头、脖子、腿、尾巴", [1, 1, 1, 1]),
        ("从长到短为头、尾巴", [0, 1, 0, 1]),
        ("其他都一样", [0, 0, 0, 0]),
        ("两长两短", [0, 0, 0, 0]),
        ("不知道", [0, 0, 0, 0]),
        (None, [0, 0, 0, 0]),
        (pd.NA, [0, 0, 0, 0]),
        (float("nan"), [0, 0, 0, 0]),
        ("  ", [0, 0, 0, 0]),
    ],
)
def test_process_use_marks_semantically_involved_parts(text, expected):
    frame = pd.DataFrame({"iSession": [1], "iTrial": [1], "text": [text]})
    actual = Recording_Processor_Center().process_use(frame)
    columns = [f"feature{i}_oraluse" for i in range(1, 5)]
    assert actual[columns].iloc[0].tolist() == expected


def test_process_use_preserves_rows_and_keeps_reference_context_within_each_report():
    frame = pd.DataFrame(
        {
            "iSession": [2, 1, 1, 1],
            "iTrial": [1, 3, 2, 1],
            "text": ["脖子长", "其他短", "四个部位都比较长", None],
        },
        index=[7, 2, 9, 4],
    )
    original = frame.copy(deep=True)
    coder = Recording_Processor_Center()
    actual = coder.process_use(frame)
    expected = frame[["iSession", "iTrial"]].copy()
    for i, values in enumerate(
        ([1, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]), 1
    ):
        expected[f"feature{i}_oraluse"] = values
    pd.testing.assert_frame_equal(actual, expected)
    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(coder.process_use(frame), expected)


def test_process_use_preserves_schema_for_empty_input():
    frame = pd.DataFrame(columns=["iSession", "iTrial", "text"])
    actual = Recording_Processor_Center().process_use(frame)
    assert actual.empty
    assert actual.columns.tolist() == [
        "iSession", "iTrial", *(f"feature{i}_oraluse" for i in range(1, 5))
    ]
