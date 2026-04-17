import logging

logger = logging.getLogger(__name__)

# Minimum rows before we warn about insufficient data
_MIN_ROWS_WARN = 10
# Minimum R² or confidence threshold to skip low-confidence warning
_MIN_CONFIDENCE = 0.3


class PostprocessorAgent:
    """
    Applies business-rule checks to the analysis dict produced by AnalystAgent
    and enriches it with warnings where appropriate.
    """

    def refine(self, analysis: dict, tool_result_metrics: dict | None = None) -> dict:
        analysis = dict(analysis)  # shallow copy
        warnings = []

        # ── 1. Low row-count warning ──────────────────────────────────────
        row_count = (tool_result_metrics or {}).get("row_count")
        if row_count is not None and row_count < _MIN_ROWS_WARN:
            warnings.append(
                f"Only {row_count} data points — results may be unreliable. "
                "Consider uploading more data."
            )

        # ── 2. Low confidence / R² warning ───────────────────────────────
        confidence = analysis.get("confidence")
        r2 = (tool_result_metrics or {}).get("r_squared")

        if confidence is not None and isinstance(confidence, (int, float)):
            if confidence < _MIN_CONFIDENCE:
                warnings.append(
                    f"Low model confidence ({confidence:.2f}) — manual review recommended."
                )

        if r2 is not None and isinstance(r2, (int, float)) and r2 < _MIN_CONFIDENCE:
            warnings.append(
                f"Low regression R² ({r2:.2f}) — the trend line may not explain the data well."
            )

        # ── 3. Missing chart data notice ──────────────────────────────────
        if analysis.get("chart_data_missing"):
            warnings.append("Chart data could not be generated for this analysis.")

        # ── 4. Attach warnings to analysis ────────────────────────────────
        if warnings:
            analysis["warnings"] = warnings
            logger.info(f"Postprocessor attached {len(warnings)} warning(s): {warnings}")
        else:
            analysis.pop("warnings", None)

        return analysis


postprocessor_agent = PostprocessorAgent()
