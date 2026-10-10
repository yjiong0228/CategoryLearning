"""Compatibility imports; implementation lives in model.modules.readout."""

from .modules.readout.report_space import (
    Report,
    EMPTY_REPORT,
    OTHER_REPORT,
    facet_token,
    body_token,
    report_code,
    component_tokens,
    _body_variant,
    ReportKernelParameters,
    _component_distribution,
    CatalogueReportSpace,
)

__all__ = [
    "Report",
    "EMPTY_REPORT",
    "OTHER_REPORT",
    "facet_token",
    "body_token",
    "report_code",
    "component_tokens",
    "ReportKernelParameters",
    "CatalogueReportSpace",
]
