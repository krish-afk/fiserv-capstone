from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.data.panel import build_panel
from src.data.store import read_fsbi, read_master
from src.models.experiment import build_trial_grid, run_experiment_with_details

from src.utils.config import config as base_config
from src.backend.presenters import build_dashboard_ui

from src.trading.pipeline import run_trading_pipeline_from_frames
from src.trading.strategy import (
    _STRATEGY_REGISTRY,
    build_strategy,
    list_available_strategies,
    load_configured_strategy,
    load_strategy_from_file,
)
import json



def _available_model_classes(cfg: dict) -> list[str]:
    models = []
    for family in cfg["experiment"]["models"].values():
        for variant in family.get("variants", []):
            cls_name = variant.get("class")
            if cls_name and cls_name not in models:
                models.append(cls_name)
    return models


def _available_feature_sets(cfg: dict) -> list[str]:
    return list(cfg["features"]["feature_sets"].keys())


def _available_panels(cfg: dict) -> list[str]:
    return list(cfg["data"]["panels"].keys())


def _normalize_requested(
    value: Any,
    *,
    available: list[str],
    default: list[str],
    label: str,
) -> list[str]:
    if value is None or value == "":
        return default

    if isinstance(value, str):
        requested = [value]
    else:
        requested = list(value)

    if any(str(v).lower() == "all" for v in requested):
        return available

    unknown = [v for v in requested if v not in available]
    if unknown:
        raise ValueError(f"Unknown {label}: {unknown}. Available: {available}")

    return requested


def _apply_request_to_config(cfg: dict, payload: dict) -> tuple[dict, dict]:
    available_panels = _available_panels(cfg)
    available_models = _available_model_classes(cfg)
    available_feature_sets = _available_feature_sets(cfg)

    selected_panels = _normalize_requested(
        payload.get("panels"),
        available=available_panels,
        default=list(cfg["experiment"]["panels"]),
        label="panels",
    )
    selected_models = _normalize_requested(
        payload.get("models"),
        available=available_models,
        default=available_models,
        label="models",
    )
    selected_feature_sets = _normalize_requested(
        payload.get("feature_sets"),
        available=available_feature_sets,
        default=available_feature_sets,
        label="feature_sets",
    )

    cfg["experiment"]["panels"] = selected_panels

    for family_name, family_cfg in cfg["experiment"]["models"].items():
        filtered_variants = []

        for variant in family_cfg.get("variants", []):
            if variant.get("class") not in selected_models:
                continue

            variant_copy = deepcopy(variant)
            current_feature_sets = variant_copy.get("feature_sets")

            if current_feature_sets:
                allowed_feature_sets = [
                    fs for fs in current_feature_sets if fs in selected_feature_sets
                ]
            else:
                allowed_feature_sets = list(selected_feature_sets)

            if not allowed_feature_sets:
                continue

            variant_copy["feature_sets"] = allowed_feature_sets
            filtered_variants.append(variant_copy)

        family_cfg["variants"] = filtered_variants
        family_cfg["enabled"] = bool(filtered_variants)

    selection = {
        "panels": selected_panels,
        "models": selected_models,
        "feature_sets": selected_feature_sets,
    }
    return cfg, selection


def _build_panels_data(cfg: dict) -> dict[str, tuple[pd.Series, pd.DataFrame]]:
    master = read_master()
    fsbi_long = read_fsbi()

    panels_data: dict[str, tuple[pd.Series, pd.DataFrame]] = {}
    for panel_name in cfg["experiment"]["panels"]:
        y, X, _ = build_panel(panel_name, master=master, fsbi_long=fsbi_long)
        panels_data[panel_name] = (y, X)

    return panels_data

def _rank_metrics_df(metrics_df: pd.DataFrame, rank_metric: str = "directional_accuracy_mape") -> pd.DataFrame:
    """
    Ranking logic for dashboard model selection.

    Default ranking:
      1. Higher directional accuracy is better.
      2. Lower MAPE is better.
      3. Lower RMSE is better.
      4. Lower MAE is better.
    """
    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    df = metrics_df.copy()

    for col in ["dir_acc", "mape", "rmse", "mae", "r2"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    composite_metrics = {
        "directional_accuracy_mape",
        "dir_acc_mape",
        "directional_accuracy",
        "dir_acc",
    }

    if rank_metric in composite_metrics:
        required = [c for c in ["dir_acc", "mape"] if c in df.columns]
        if required:
            df = df.dropna(subset=required)

        sort_cols = [c for c in ["dir_acc", "mape", "rmse", "mae"] if c in df.columns]
        ascending = [False if c == "dir_acc" else True for c in sort_cols]

        if not sort_cols:
            return df

        return df.sort_values(sort_cols, ascending=ascending, na_position="last")

    metric = rank_metric if rank_metric in df.columns else "rmse"
    df = df.dropna(subset=[metric])

    # Metrics where higher is better.
    higher_is_better = {"dir_acc", "directional_accuracy", "r2"}
    ascending = metric not in higher_is_better

    return df.sort_values(metric, ascending=ascending, na_position="last")

def _best_row(metrics_df: pd.DataFrame, panel_name: str, rank_metric: str) -> Optional[pd.Series]:
    subset = metrics_df.copy()

    if "panel_name" in subset.columns:
        subset = subset[subset["panel_name"] == panel_name].copy()

    if subset.empty:
        return None

    ranked = _rank_metrics_df(subset, rank_metric=rank_metric)

    if ranked.empty:
        return None

    return ranked.iloc[0]

def _top_model_metric_bar_payload(panel_name: str, rows: list[dict], metric: str = "rmse") -> dict:
    chart_rows = [
        {
            "label": f"{row.get('model_name')} | {row.get('feature_set')}",
            metric: row.get(metric),
        }
        for row in rows
        if row.get(metric) is not None
    ]
    return {
        "panel_name": panel_name,
        "metric": metric,
        "rows": chart_rows,
    }


def _forecast_error_plot_payload(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    panel_name: str,
    rank_metric: str = "rmse",
) -> dict:
    base = _forecast_plot_payload(
        forecasts_df=forecasts_df,
        metrics_df=metrics_df,
        panel_name=panel_name,
        rank_metric=rank_metric,
    )

    rows = []
    for dt, actual, model_fc, naive_fc in zip(
        base.get("dates", []),
        base.get("actual", []),
        base.get("model_forecast", []),
        base.get("naive_forecast", []),
    ):
        model_abs_error = None if actual is None or model_fc is None else abs(float(actual) - float(model_fc))
        naive_abs_error = None if actual is None or naive_fc is None else abs(float(actual) - float(naive_fc))
        rows.append(
            {
                "date": dt,
                "model_abs_error": model_abs_error,
                "naive_abs_error": naive_abs_error,
            }
        )

    return {
        "panel_name": panel_name,
        "model_name": base.get("model_name"),
        "feature_set": base.get("feature_set"),
        "rows": rows,
    }


def _drawdown_curve_payload(equity_curve_df: pd.DataFrame) -> list[dict]:
    if equity_curve_df is None or equity_curve_df.empty:
        return []

    df = equity_curve_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["rolling_max"] = df["equity"].cummax()
    df["drawdown"] = (df["equity"] / df["rolling_max"]) - 1.0

    return [
        {"date": d.strftime("%Y-%m-%d"), "drawdown": float(v)}
        for d, v in zip(df["date"], df["drawdown"])
    ]


def _period_return_curve_payload(trades_df: pd.DataFrame) -> list[dict]:
    if trades_df is None or trades_df.empty or "net_return" not in trades_df.columns:
        return []

    df = trades_df.copy()

    date_col = None
    for candidate in ["exit_date", "entry_date", "date"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        return []

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "net_return": float(r) if pd.notnull(r) else None,
        }
        for d, r in zip(df[date_col], pd.to_numeric(df["net_return"], errors="coerce"))
    ]


def _weights_curve_payload(signals_df: pd.DataFrame) -> Optional[dict]:
    if signals_df is None or signals_df.empty:
        return None

    df = signals_df.copy()
    if "date" not in df.columns:
        return None

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    weight_cols = [c for c in df.columns if str(c).startswith("weight__")]
    if not weight_cols and "cash_weight" not in df.columns:
        return None

    rows = []
    for _, row in df.iterrows():
        item = {"date": row["date"].strftime("%Y-%m-%d")}
        for col in weight_cols:
            item[col.replace("weight__", "")] = float(row[col]) if pd.notnull(row[col]) else None
        if "cash_weight" in df.columns:
            item["Cash"] = float(row["cash_weight"]) if pd.notnull(row["cash_weight"]) else None
        rows.append(item)

    if not rows:
        return None

    series = [{"key": key, "label": key} for key in rows[0].keys() if key != "date"]

    return {
        "title": "Strategy Weights / Exposure",
        "rows": rows,
        "series": series,
    }


def _confidence_curve_payload(signals_df: pd.DataFrame) -> Optional[list[dict]]:
    if signals_df is None or signals_df.empty or "confidence" not in signals_df.columns or "date" not in signals_df.columns:
        return None

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    return [
        {
            "date": d.strftime("%Y-%m-%d"),
            "confidence": float(v) if pd.notnull(v) else None,
        }
        for d, v in zip(df["date"], pd.to_numeric(df["confidence"], errors="coerce"))
    ]


def _metadata_numeric_curves(signals_df: pd.DataFrame) -> list[dict]:
    if signals_df is None or signals_df.empty or "metadata" not in signals_df.columns or "date" not in signals_df.columns:
        return []

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    rows = []
    for _, row in df.iterrows():
        meta = row.get("metadata")
        parsed = {}
        if isinstance(meta, str) and meta.strip():
            try:
                parsed = json.loads(meta)
            except Exception:
                parsed = {}
        entry = {"date": row["date"].strftime("%Y-%m-%d")}
        for key, value in parsed.items():
            if isinstance(value, (int, float)):
                entry[key] = float(value)
        rows.append(entry)

    if not rows:
        return []

    numeric_keys = sorted({k for row in rows for k in row.keys() if k != "date"})
    charts = []

    for key in numeric_keys:
        chart_rows = [{"date": row["date"], key: row.get(key)} for row in rows if key in row]
        if chart_rows:
            charts.append(
                {
                    "title": key.replace("_", " ").title(),
                    "key": key,
                    "rows": chart_rows,
                    "series": [{"key": key, "label": key.replace("_", " ").title()}],
                }
            )

    return charts


def _forecast_plot_payload(
    forecasts_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    panel_name: str,
    rank_metric: str = "rmse",
) -> dict:
    best = _best_row(metrics_df, panel_name, rank_metric)

    if best is None:
        return {
            "panel_name": panel_name,
            "model_name": None,
            "feature_set": None,
            "dates": [],
            "actual": [],
            "model_forecast": [],
            "naive_forecast": [],
        }

    selected = forecasts_df[
        (forecasts_df["panel_name"] == panel_name)
        & (forecasts_df["model_name"] == best["model_name"])
        & (forecasts_df["feature_set"] == best["feature_set"])
    ].copy()

    naive = forecasts_df[
        (forecasts_df["panel_name"] == panel_name)
        & (forecasts_df["model_name"].astype(str).str.contains("naive", case=False, na=False))
    ].copy()

    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.sort_values("date")

    naive["date"] = pd.to_datetime(naive["date"])
    naive = naive.sort_values("date")

    merged = selected[["date", "y_true", "y_pred"]].rename(columns={"y_pred": "model_forecast"})
    if not naive.empty:
        naive_small = naive[["date", "y_pred"]].rename(columns={"y_pred": "naive_forecast"})
        merged = merged.merge(naive_small, on="date", how="left")
    else:
        merged["naive_forecast"] = np.nan

    return {
        "panel_name": panel_name,
        "model_name": best["model_name"],
        "feature_set": best["feature_set"],
        "dates": merged["date"].dt.strftime("%Y-%m-%d").tolist(),
        "actual": merged["y_true"].tolist(),
        "model_forecast": merged["model_forecast"].tolist(),
        "naive_forecast": merged["naive_forecast"].tolist(),
    }


def _top_models_by_panel(
    metrics_df: pd.DataFrame,
    top_k: int = 5,
    rank_metric: str = "directional_accuracy_mape",
) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    for panel_name, panel_df in metrics_df.groupby("panel_name"):
        ranked = _rank_metrics_df(panel_df, rank_metric=rank_metric).head(top_k).copy()
        out[panel_name] = _records(ranked)

    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if value is None:
        return None

    return value


def _records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []

    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")

    return _json_safe(out.where(pd.notnull(out), None).to_dict(orient="records"))


def _risk_metrics_from_trades(trades_df: pd.DataFrame, initial_cash: float, final_value: float) -> dict:
    if trades_df is None or trades_df.empty or "net_return" not in trades_df.columns:
        cumulative_return = None
        if initial_cash and final_value is not None:
            cumulative_return = (float(final_value) / float(initial_cash)) - 1.0

        return {
            "value_at_risk_95": None,
            "conditional_loss_95": None,
            "cumulative_return": cumulative_return,
        }

    returns = pd.to_numeric(trades_df["net_return"], errors="coerce").dropna()
    if returns.empty:
        return {
            "value_at_risk_95": None,
            "conditional_loss_95": None,
            "cumulative_return": (float(final_value) / float(initial_cash)) - 1.0
            if initial_cash and final_value is not None
            else None,
        }

    var_95 = float(returns.quantile(0.05))
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95

    return {
        "value_at_risk_95": var_95,
        "conditional_loss_95": cvar_95,
        "cumulative_return": (float(final_value) / float(initial_cash)) - 1.0
        if initial_cash and final_value is not None
        else None,
    }

def _metric_from_results(results: dict, key: str, stat: str = "median") -> Optional[float]:
    value = results.get(key)

    if value is None:
        metric_summary = (results.get("metrics") or {}).get(key)
        if isinstance(metric_summary, dict):
            value = metric_summary.get(stat)

    try:
        num = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(num):
        return None

    return num


def _pct_points_to_decimal(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / 100.0

def _risk_metrics_from_monte_carlo_paths(
    path_results_df: pd.DataFrame,
    initial_cash: float,
    final_value: Optional[float],
) -> dict:
    if path_results_df is None or path_results_df.empty or "return_pct" not in path_results_df.columns:
        cumulative_return = None
        if initial_cash and final_value is not None:
            cumulative_return = (float(final_value) / float(initial_cash)) - 1.0

        return {
            "value_at_risk_95": None,
            "conditional_loss_95": None,
            "cumulative_return": cumulative_return,
        }

    returns_pct = pd.to_numeric(path_results_df["return_pct"], errors="coerce").dropna()
    if returns_pct.empty:
        cumulative_return = None
        if initial_cash and final_value is not None:
            cumulative_return = (float(final_value) / float(initial_cash)) - 1.0

        return {
            "value_at_risk_95": None,
            "conditional_loss_95": None,
            "cumulative_return": cumulative_return,
        }

    returns = returns_pct / 100.0
    var_95 = float(returns.quantile(0.05))
    tail = returns[returns <= var_95]
    cvar_95 = float(tail.mean()) if not tail.empty else var_95

    return {
        "value_at_risk_95": var_95,
        "conditional_loss_95": cvar_95,
        "cumulative_return": float(returns.median()),
    }


def _equity_and_pnl_payload(equity_curve_df: pd.DataFrame, initial_cash: float) -> tuple[list[dict], list[dict]]:
    if equity_curve_df is None or equity_curve_df.empty:
        return [], []

    df = equity_curve_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    equity_curve = [
        {"date": d.strftime("%Y-%m-%d"), "equity": float(v)}
        for d, v in zip(df["date"], df["equity"])
    ]

    pnl_curve = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "cumulative_return": (float(v) / float(initial_cash)) - 1.0,
        }
        for d, v in zip(df["date"], df["equity"])
    ]

    return equity_curve, pnl_curve


def _build_strategy_from_payload(payload: dict, cfg: dict):
    strategy_spec = payload.get("strategy") or {}
    if not strategy_spec:
        return load_configured_strategy(cfg=cfg)

    params = strategy_spec.get("params", {}) or {}
    source = strategy_spec.get("source")
    strategy_file = strategy_spec.get("file")
    class_name = strategy_spec.get("class_name") or strategy_spec.get("name")

    if source == "file" or strategy_file:
        return load_strategy_from_file(
            strategy_file=strategy_file,
            class_name=class_name,
            params=params,
        )

    if class_name:
        return build_strategy(
            name=class_name,
            **params,
        )

    return load_configured_strategy(cfg=cfg)


def get_dashboard_options(cfg: Optional[dict] = None) -> dict:
    cfg = deepcopy(cfg or base_config)
    strategy_catalog = list_available_strategies()

    return {
        "panels": _available_panels(cfg),
        "models": _available_model_classes(cfg),
        "feature_sets": _available_feature_sets(cfg),
        "strategies": {
            "catalog": strategy_catalog,
            "default_id": strategy_catalog[0]["id"] if strategy_catalog else None,
        },
        "trading_modes": ["backtest", "monte_carlo"],
        "defaults": {
            "panels": list(cfg["experiment"]["panels"]),
            "ranking_metric": "directional_accuracy_mape",
            "top_k": 5,
            "run_trading": True,
        },
    }


def run_dashboard_pipeline(payload: dict, cfg: Optional[dict] = None) -> dict:
    cfg = deepcopy(cfg or base_config)
    payload = payload or {}

    cfg, selection = _apply_request_to_config(cfg, payload)

    panels_data = _build_panels_data(cfg)
    trials = build_trial_grid(cfg, panels_data)

    if not trials:
        raise ValueError("No experiment trials were generated from the selected panels/models/feature sets.")

    horizon = int(cfg["forecasting"]["horizons"][0])
    min_train_size = int(cfg["forecasting"]["walk_forward_min_train"])
    rank_metric = str(payload.get("ranking_metric", "directional_accuracy_mape"))
    top_k = int(payload.get("top_k", 5))

    exp_result = run_experiment_with_details(
        trials=trials,
        min_train_size=min_train_size,
        horizon=horizon,
    )

    metrics_df = exp_result.metrics_df.copy()
    forecasts_df = exp_result.forecasts_df.copy()

    top_models_by_panel = _top_models_by_panel(
        metrics_df=metrics_df,
        top_k=top_k,
        rank_metric=rank_metric,
    )

    top_model_chart_metric = "dir_acc" if rank_metric in {
        "directional_accuracy_mape",
        "dir_acc_mape",
        "directional_accuracy",
        "dir_acc",
    } else rank_metric

    forecasting = {
        "metrics_table": _records(metrics_df),
        "top_models_by_panel": top_models_by_panel,
        "plots": {
            "forecast_comparison": {
                panel_name: _forecast_plot_payload(
                    forecasts_df=forecasts_df,
                    metrics_df=metrics_df,
                    panel_name=panel_name,
                    rank_metric=rank_metric,
                )
                for panel_name in selection["panels"]
            },
            "forecast_error": {
                panel_name: _forecast_error_plot_payload(
                    forecasts_df=forecasts_df,
                    metrics_df=metrics_df,
                    panel_name=panel_name,
                    rank_metric=rank_metric,
                )
                for panel_name in selection["panels"]
            },
            "top_model_metric": {
                panel_name: _top_model_metric_bar_payload(
                    panel_name=panel_name,
                    rows=rows,
                    metric=top_model_chart_metric,
                )
                for panel_name, rows in top_models_by_panel.items()
            },
        },
    }

    response = {
        "pipeline_run_id": exp_result.run_dir.name,
        "run_dir": str(exp_result.run_dir),
        "selection": selection,
        "metadata": _json_safe(exp_result.metadata),
        "forecasting": forecasting,
        "trading": None,
    }
    trade_start_date = payload.get("trade_start_date")
    trade_end_date = payload.get("trade_end_date")

    run_trading = bool(payload.get("run_trading", True))
    if run_trading:
        strategy = _build_strategy_from_payload(payload, cfg)
        trading_panel = payload.get("trading_panel") or selection["panels"][0]
        trading_mode = str(payload.get("trading_mode") or cfg.get("trading", {}).get("test_mode", "backtest"))

        trading_run = run_trading_pipeline_from_frames(
            strategy=strategy,
            forecasts_df=forecasts_df,
            metrics_df=metrics_df,
            primary_panel=trading_panel,
            cfg=cfg,
            refresh_market_data=bool(payload.get("refresh_market_data", False)),
            output_dir=Path(exp_result.run_dir) / "trading",
            mrts_panel=payload.get("mrts_panel"),
            test_mode=trading_mode,
            trade_start_date=trade_start_date,
            trade_end_date=trade_end_date,
        )

        trades_df = trading_run["trades_df"]
        path_results_df = trading_run.get("path_results_df", pd.DataFrame())
        equity_curve_df = trading_run["equity_curve_df"]
        results = dict(trading_run["results"])

        signals_df = trading_run.get("signals_df")
        drawdown_curve = _drawdown_curve_payload(equity_curve_df)
        period_return_curve = _period_return_curve_payload(trades_df)
        weights_curve = _weights_curve_payload(signals_df)
        confidence_curve = _confidence_curve_payload(signals_df)
        metadata_curves = _metadata_numeric_curves(signals_df)

        initial_cash = float(
            results.get(
                "initial_cash",
                cfg.get("trading", {}).get("portfolio", {}).get("initial_cash", 100000),
            )
        )

        final_value = _metric_from_results(results, "final_value")
        is_monte_carlo = str(trading_run.get("test_mode", "")).lower() == "monte_carlo"

        if is_monte_carlo:
            extra_risk = _risk_metrics_from_monte_carlo_paths(
                path_results_df=path_results_df,
                initial_cash=initial_cash,
                final_value=final_value,
            )
        else:
            extra_risk = _risk_metrics_from_trades(
                trades_df=trades_df,
                initial_cash=initial_cash,
                final_value=final_value,
            )
        equity_curve, pnl_curve = _equity_and_pnl_payload(
            equity_curve_df=equity_curve_df,
            initial_cash=initial_cash,
        )

        response["trading"] = {
            "panel_name": trading_panel,
            "strategy_name": trading_run["strategy_name"],
            "test_mode": trading_run["test_mode"],
            "tickers": trading_run["tickers"],
            "trade_window_start": trading_run.get("trade_window_start"),
            "trade_window_end": trading_run.get("trade_window_end"),
            "metrics": _json_safe({
                "value_at_risk_95": extra_risk["value_at_risk_95"],
                "conditional_loss_95": extra_risk["conditional_loss_95"],
                "max_drawdown_pct": _pct_points_to_decimal(_metric_from_results(results, "max_drawdown_pct")),
                "win_rate": _metric_from_results(results, "win_rate"),
                "cumulative_return": extra_risk["cumulative_return"],
                "num_trades": _metric_from_results(results, "num_trades"),
                "final_value": final_value,
                "absolute_return": _metric_from_results(results, "absolute_return"),
                "annualized_return_pct": _pct_points_to_decimal(_metric_from_results(results, "annualized_return_pct")),
                "sharpe_ratio": _metric_from_results(results, "sharpe_ratio"),
            }),
            "equity_curve": equity_curve,
            "pnl_curve": pnl_curve,
            "drawdown_curve": drawdown_curve,
            "period_return_curve": period_return_curve,
            "weights_curve": weights_curve,
            "confidence_curve": confidence_curve,
            "metadata_curves": metadata_curves,
            "trades_table": _records(trades_df),
            "output_dir": trading_run["output_dir"],
        }

    response["ui"] = build_dashboard_ui(response)
    return _json_safe(response)