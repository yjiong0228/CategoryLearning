"""口述报告评价的稳定公共组合接口。"""

from .mapping import OralCenterMapper, OralRegionMapper, RegionOverlapScorer
from .reporting import OralAlignmentReportingMixin


class OralModelAlignmentMixin(OralAlignmentReportingMixin):
    """组合口述映射、对齐计算和报告输出，供 ModelEvaluator 使用。"""


__all__ = [
    "OralCenterMapper",
    "OralModelAlignmentMixin",
    "OralRegionMapper",
    "RegionOverlapScorer",
]
