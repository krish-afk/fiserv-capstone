from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _to_float(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _display_number(
    value: Any,
    *,
    decimals: int = 4,
    pct: bool = False,
    signed_pct: bool = False,
) -> str:
    num = _to_float(value)
    if num is None:
        return "N/A"

    if pct:
        return f"{num * 100:.2f}%"

    if signed_pct:
        sign = "+" if num >= 0 else ""
        return f"{sign}{num * 100:.2f}%"

    return f"{num:.{decimals}f}"


def build_kpi_card(
    *,
    key: str,
    label: str,
    value: Any,
    kind: str = "number",
    decimals: int = 4,
) -> dict:
    num = _to_float(value)

    if kind == "percent":
        display = _display_number(num, pct=True)
    elif kind == "signed_percent":
        display = _display_number(num, signed_pct=True)
    elif kind == "integer":
        display = "N/A" if num is None else str(int(round(num)))
    elif kind == "currency":
        display = "N/A" if num is None else f"${num:,.2f}"
    else:
        display = "N/A" if num is None else f"{num:.{decimals}f}"

    return {
        "key": key,
        "label": label,
        "value": num,
        "display": display,
        "kind": kind,
    }


def build_table(
    *,
    key: str,
    title: str,
    rows: list[dict],
    columns: list[dict],
    empty_message: str = "No rows available.",
) -> dict:
    return {
        "key": key,
        "title": title,
        "columns": columns,
        "rows": rows,
        "empty_message": empty_message,
    }


def build_line_chart(
    *,
    key: str,
    title: str,
    x_key: str,
    series: list[dict],
    rows: list[dict],
) -> dict:
    return {
        "key": key,
        "title": title,
        "chart_type": "line",
        "x_key": x_key,
        "series": series,
        "rows": rows,
    }

def build_bar_chart(
    *,
    key: str,
    title: str,
    x_key: str,
    series: list[dict],
    rows: list[dict],
) -> dict:
    return {
        "key": key,
        "title": title,
        "chart_type": "bar",
        "x_key": x_key,
        "series": series,
        "rows": rows,
    }


def build_forecast_chart_from_payload(plot_payload: dict) -> dict:
    rows = []
    dates = plot_payload.get("dates", [])
    actual = plot_payload.get("actual", [])
    model_fc = plot_payload.get("model_forecast", [])
    naive_fc = plot_payload.get("naive_forecast", [])

    for i, dt in enumerate(dates):
        rows.append(
            {
                "date": dt,
                "actual": actual[i] if i < len(actual) else None,
                "model_forecast": model_fc[i] if i < len(model_fc) else None,
                "naive_forecast": naive_fc[i] if i < len(naive_fc) else None,
            }
        )

    model_name = plot_payload.get("model_name") or "Selected model"

    return build_line_chart(
        key=f"forecast_vs_actual_{plot_payload.get('panel_name')}",
        title=f"{plot_payload.get('panel_name')} — Forecast vs Actual vs Naive",
        x_key="date",
        series=[
            {"key": "actual", "label": "Actual"},
            {"key": "model_forecast", "label": model_name},
            {"key": "naive_forecast", "label": "Naive"},
        ],
        rows=rows,
    )


def build_top_models_table(panel_name: str, rows: list[dict]) -> dict:
    return build_table(
        key=f"top_models_{panel_name}",
        title=f"Top Models — {panel_name}",
        columns=[
            {"key": "model_name", "label": "Model"},
            {"key": "feature_set", "label": "Feature Set"},
            {"key": "rmse", "label": "RMSE"},
            {"key": "mae", "label": "MAE"},
            {"key": "mape", "label": "MAPE"},
            {"key": "dir_acc", "label": "Directional Acc"},
            {"key": "r2", "label": "R²"},
        ],
        rows=rows,
        empty_message="No ranked models available for this panel.",
    )


def build_metrics_table(rows: list[dict]) -> dict:
    return build_table(
        key="forecast_metrics",
        title="Forecasting Metrics",
        columns=[
            {"key": "panel_name", "label": "Panel"},
            {"key": "model_name", "label": "Model"},
            {"key": "feature_set", "label": "Feature Set"},
            {"key": "rmse", "label": "RMSE"},
            {"key": "mae", "label": "MAE"},
            {"key": "mape", "label": "MAPE"},
            {"key": "dir_acc", "label": "Directional Acc"},
            {"key": "r2", "label": "R²"},
        ],
        rows=rows,
        empty_message="No forecasting metrics available.",
    )


def build_forecasting_ui(forecasting: dict) -> dict:
    metrics_table = forecasting.get("metrics_table", [])
    top_models_by_panel = forecasting.get("top_models_by_panel", {})
    plots = forecasting.get("plots", {})

    ranking_panels = []
    chart_cards = []

    for panel_name, top_rows in top_models_by_panel.items():
        ranking_panels.append(build_top_models_table(panel_name, top_rows))

    for panel_name, plot_payload in (plots.get("forecast_comparison") or {}).items():
        chart_cards.append(
            {
                "panel_name": panel_name,
                "group": "forecast_comparison",
                "chart": build_forecast_chart_from_payload(plot_payload),
            }
        )

    for panel_name, error_payload in (plots.get("forecast_error") or {}).items():
        chart_cards.append(
            {
                "panel_name": panel_name,
                "group": "forecast_error",
                "chart": build_line_chart(
                    key=f"forecast_error_{panel_name}",
                    title=f"{panel_name} — Absolute Error: Model vs Naive",
                    x_key="date",
                    series=[
                        {"key": "model_abs_error", "label": "Model Abs Error"},
                        {"key": "naive_abs_error", "label": "Naive Abs Error"},
                    ],
                    rows=error_payload.get("rows", []),
                ),
            }
        )

    for panel_name, rank_payload in (plots.get("top_model_metric") or {}).items():
        metric = rank_payload.get("metric", "rmse")
        chart_cards.append(
            {
                "panel_name": panel_name,
                "group": "top_model_ranking",
                "chart": build_bar_chart(
                    key=f"top_model_metric_{panel_name}",
                    title=f"{panel_name} — Top Models by {metric.upper()}",
                    x_key="label",
                    series=[{"key": metric, "label": metric.upper()}],
                    rows=rank_payload.get("rows", []),
                ),
            }
        )

    summary_cards = [
        build_kpi_card(
            key="forecast_panels",
            label="Panels Evaluated",
            value=len((plots.get("forecast_comparison") or {}).keys()),
            kind="integer",
        ),
        build_kpi_card(
            key="forecast_models",
            label="Model Runs",
            value=len(metrics_table),
            kind="integer",
        ),
    ]

    return {
        "summary_cards": summary_cards,
        "tables": {
            "metrics": build_metrics_table(metrics_table),
            "top_models_by_panel": ranking_panels,
        },
        "charts": {
            "all": chart_cards,
        },
    }

def build_trading_ui(trading: Optional[dict]) -> Optional[dict]:
    if not trading:
        return None

    metrics = trading.get("metrics", {})
    equity_curve = trading.get("equity_curve", [])
    pnl_curve = trading.get("pnl_curve", [])
    drawdown_curve = trading.get("drawdown_curve", [])
    period_return_curve = trading.get("period_return_curve", [])
    weights_curve = trading.get("weights_curve")
    confidence_curve = trading.get("confidence_curve")
    # Forecast/prediction metadata curves are intentionally not shown in the Trading tab.
    # The Forecasting tab already owns forecast-vs-actual visualizations.

    cards = [
        build_kpi_card(key="final_value", label="Final Portfolio Value", value=metrics.get("final_value"), kind="currency"),
        build_kpi_card(key="cumulative_return", label="Cumulative Return", value=metrics.get("cumulative_return"), kind="signed_percent"),
        build_kpi_card(key="max_drawdown_pct", label="Max Drawdown", value=metrics.get("max_drawdown_pct"), kind="percent"),
        build_kpi_card(key="win_rate", label="Win Rate", value=metrics.get("win_rate"), kind="percent"),
        build_kpi_card(key="num_trades", label="Number of Trades", value=metrics.get("num_trades"), kind="integer"),
        build_kpi_card(key="value_at_risk_95", label="VaR (95%)", value=metrics.get("value_at_risk_95"), kind="percent"),
        build_kpi_card(key="conditional_loss_95", label="Conditional Loss (95%)", value=metrics.get("conditional_loss_95"), kind="percent"),
        build_kpi_card(key="annualized_return_pct", label="Annualized Return", value=metrics.get("annualized_return_pct"), kind="percent"),
        build_kpi_card(key="sharpe_ratio", label="Sharpe Ratio", value=metrics.get("sharpe_ratio"), kind="number", decimals=3),
    ]

    charts = [
        build_line_chart(
            key="equity_curve",
            title="Equity Curve",
            x_key="date",
            series=[{"key": "equity", "label": "Equity"}],
            rows=equity_curve,
        ),
        build_line_chart(
            key="cumulative_return_curve",
            title="Cumulative Return Curve",
            x_key="date",
            series=[{"key": "cumulative_return", "label": "Cumulative Return"}],
            rows=pnl_curve,
        ),
        build_line_chart(
            key="drawdown_curve",
            title="Drawdown Curve",
            x_key="date",
            series=[{"key": "drawdown", "label": "Drawdown"}],
            rows=drawdown_curve,
        ),
        build_line_chart(
            key="period_return_curve",
            title="Per-Period Return",
            x_key="date",
            series=[{"key": "net_return", "label": "Net Return"}],
            rows=period_return_curve,
        ),
    ]

    if weights_curve:
        charts.append(
            build_line_chart(
                key="weights_curve",
                title=weights_curve.get("title", "Weights / Exposure"),
                x_key="date",
                series=weights_curve.get("series", []),
                rows=weights_curve.get("rows", []),
            )
        )

    if confidence_curve:
        charts.append(
            build_line_chart(
                key="confidence_curve",
                title="Signal Confidence",
                x_key="date",
                series=[{"key": "confidence", "label": "Confidence"}],
                rows=confidence_curve,
            )
        )

    return {
        "header": {
            "panel_name": trading.get("panel_name"),
            "strategy_name": trading.get("strategy_name"),
            "test_mode": trading.get("test_mode"),
            "tickers": trading.get("tickers", []),
        },
        "summary_cards": cards,
        "charts": {
            "all": charts,
        }
    }

def build_dashboard_ui(response: dict) -> dict:
    forecasting_ui = build_forecasting_ui(response.get("forecasting", {}))
    trading_ui = build_trading_ui(response.get("trading"))

    selection = response.get("selection", {})
    overview_cards = [
        build_kpi_card(
            key="selected_panels_count",
            label="Selected Panels",
            value=len(selection.get("panels", [])),
            kind="integer",
        ),
        build_kpi_card(
            key="selected_models_count",
            label="Selected Models",
            value=len(selection.get("models", [])),
            kind="integer",
        ),
        build_kpi_card(
            key="selected_feature_sets_count",
            label="Selected Feature Sets",
            value=len(selection.get("feature_sets", [])),
            kind="integer",
        ),
    ]

    return {
        "overview": {
            "summary_cards": overview_cards,
        },
        "forecasting": forecasting_ui,
        "trading": trading_ui,
    }