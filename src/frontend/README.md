# Frontend Dashboard

This is a Vite + React frontend for the macro forecasting and trading dashboard backend.

## Features

- Load dashboard options from `/api/options`
- Submit runs to `/api/runs`
- Poll run status and results from `/api/runs/<run_id>` and `/api/runs/<run_id>/results`
- Display:
  - overview KPI cards
  - forecasting KPI cards
  - forecast vs actual vs naive charts
  - forecasting metrics table
  - top-model tables by panel
  - trading KPI cards
  - equity curve chart
  - cumulative return chart
  - trades table
- Show recent runs and reload them
- Animated progress bar while a run is executing

## Run locally

Start the Flask backend first on port 5000.

Then in this `frontend/` directory:

```bash
npm install
npm run dev