from flask import Flask, jsonify, request

from src.backend.run_store import (
    complete_run,
    create_run,
    fail_run,
    get_run,
    get_run_results,
    list_runs,
)
from src.backend.service import get_dashboard_options, run_dashboard_pipeline

app = Flask(__name__)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.get("/api/options")
def options():
    try:
        return jsonify(get_dashboard_options()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/runs")
def runs_index():
    try:
        limit = int(request.args.get("limit", 20))
        return jsonify({"runs": list_runs(limit=limit)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/runs")
def run_pipeline():
    payload = request.get_json(silent=True) or {}
    meta = create_run(payload)
    run_id = meta["run_id"]

    try:
        result = run_dashboard_pipeline(payload)
        complete_run(run_id, result)

        return jsonify(
            {
                "run_id": run_id,
                "status": "completed",
                "run_url": f"/api/runs/{run_id}",
                "results_url": f"/api/runs/{run_id}/results",
            }
        ), 200

    except Exception as e:
        fail_run(run_id, str(e))
        return jsonify(
            {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
                "run_url": f"/api/runs/{run_id}",
            }
        ), 400


@app.get("/api/runs/<run_id>")
def run_status(run_id: str):
    try:
        meta = get_run(run_id)
        if meta is None:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404
        return jsonify(meta), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.get("/api/runs/<run_id>/results")
def run_results(run_id: str):
    try:
        meta = get_run(run_id)
        if meta is None:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404

        if meta["status"] == "failed":
            return jsonify(
                {
                    "run_id": run_id,
                    "status": "failed",
                    "error": meta.get("error"),
                }
            ), 200

        results = get_run_results(run_id)
        if results is None:
            return jsonify(
                {
                    "run_id": run_id,
                    "status": meta["status"],
                    "message": "Results are not available yet.",
                }
            ), 202

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400