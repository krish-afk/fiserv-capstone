import { useEffect, useMemo, useRef, useState } from 'react'
import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    Line,
    LineChart,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
} from 'recharts'
import { createRun, fetchOptions, fetchRunResults, fetchRunStatus, fetchRuns } from './api'


function formatLabel(value) {
  return String(value || '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function classNames(...parts) {
  return parts.filter(Boolean).join(' ')
}

const BRAND_LOGOS = [
  {
    name: 'Georgia Tech',
    fallback: 'GT',
    variant: 'gt',
    assets: ['public/gt-logo.png', '/logos/gt-logo.png', '/GT-logo.svg', '/GT-logo.png'],
  },
  {
    name: 'Fiserv',
    fallback: 'fiserv',
    variant: 'fiserv',
    assets: ['public/fiserv-logo.png', '/logos/fiserv-logo.png', '/Fiserv-logo.svg', '/Fiserv-logo.png'],
  },
]

function barColorForSeries(series, index) {
  const key = String(series?.key || '').toLowerCase()

  if (key.includes('rmse') || key.includes('mae') || key.includes('mape')) {
    return GT_NAVY
  }

  return CHART_PALETTE[index % CHART_PALETTE.length]
}

function BrandLogo({ logo }) {
  const [assetIndex, setAssetIndex] = useState(0)
  const src = logo.assets?.[assetIndex]
  const hasAsset = Boolean(src) && assetIndex < logo.assets.length

  return (
    <div
      className={classNames('brand-logo-card', `brand-logo-card--${logo.variant}`)}
      aria-label={`${logo.name} logo`}
    >
      {hasAsset ? (
        <img
          src={src}
          alt={`${logo.name} logo`}
          onError={() => setAssetIndex((current) => current + 1)}
        />
      ) : (
        <span>{logo.fallback}</span>
      )}
    </div>
  )
}

function buildDefaultStrategyParams(strategy) {
    const out = {}
    for (const param of strategy?.parameters || []) {
      if (param.type === 'boolean') {
        out[param.name] = param.default ?? false
      } else if (param.type === 'ticker_list') {
        out[param.name] = Array.isArray(param.default) ? param.default : []
      } else {
        out[param.name] = param.default ?? ''
      }
    }
    return out
  }
  
  function coerceStrategyParams(strategy, rawParams) {
    const out = {}
    for (const param of strategy?.parameters || []) {
      let value = rawParams?.[param.name]
  
      if (param.type === 'number') {
        value = value === '' || value === null || value === undefined ? null : Number(value)
      } else if (param.type === 'boolean') {
        value = Boolean(value)
      } else if (param.type === 'ticker') {
        value = value ? String(value).toUpperCase().trim() : ''
      } else if (param.type === 'ticker_list') {
        if (Array.isArray(value)) {
          value = value.map((v) => String(v).toUpperCase().trim()).filter(Boolean)
        } else if (typeof value === 'string') {
          value = value.split(',').map((v) => v.toUpperCase().trim()).filter(Boolean)
        } else {
          value = []
        }
      } else {
        value = value ?? ''
      }
  
      out[param.name] = value
    }
    return out
  }
  
  function StrategyParamField({ param, value, onChange, disabled }) {
    const commonProps = {
      disabled,
      placeholder: param.placeholder || '',
    }
  
    if (param.type === 'boolean') {
      return (
        <div className="checkbox-row">
          <input
            id={`param_${param.name}`}
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onChange(param.name, e.target.checked)}
            disabled={disabled}
          />
          <label htmlFor={`param_${param.name}`}>{param.label}</label>
        </div>
      )
    }
  
    if (param.type === 'ticker_list') {
      return (
        <div className="input-group">
          <label className="field-label">{param.label}</label>
          <input
            type="text"
            value={Array.isArray(value) ? value.join(', ') : ''}
            onChange={(e) => onChange(param.name, e.target.value.split(',').map((v) => v.trim()).filter(Boolean))}
            {...commonProps}
          />
        </div>
      )
    }
  
    if (param.type === 'number') {
      return (
        <div className="input-group">
          <label className="field-label">{param.label}</label>
          <input
            type="number"
            step={param.step || 'any'}
            value={value ?? ''}
            onChange={(e) => onChange(param.name, e.target.value)}
            {...commonProps}
          />
        </div>
      )
    }
  
    if (param.type === 'ticker') {
      return (
        <div className="input-group">
          <label className="field-label">{param.label}</label>
          <input
            type="text"
            value={value ?? ''}
            onChange={(e) => onChange(param.name, e.target.value.toUpperCase())}
            {...commonProps}
          />
        </div>
      )
    }
  
    return (
      <div className="input-group">
        <label className="field-label">{param.label}</label>
        <input
          type="text"
          value={value ?? ''}
          onChange={(e) => onChange(param.name, e.target.value)}
          {...commonProps}
        />
      </div>
    )
  }

  function makeInitialForm(options) {
    const defaults = options?.defaults || {}
    const defaultStrategy = options?.strategies?.catalog?.[0] || null
  
    return {
      panels: defaults.panels || [],
      models: [],
      feature_sets: [],
      ranking_metric: defaults.ranking_metric || 'mape',
      top_k: defaults.top_k || 5,
      run_trading: defaults.run_trading ?? true,
      trading_mode: 'backtest',
      strategy_id: defaultStrategy?.id || '',
      trade_start_date: '',
      trade_end_date: '',
      strategy_params: buildDefaultStrategyParams(defaultStrategy),
    }
  }

function looksPercentKey(key) {
  return [
    'win_rate',
    'cumulative_return',
    'annualized_return_pct',
    'max_drawdown_pct',
    'value_at_risk_95',
    'conditional_loss_95',
    'net_return',
    'dir_acc',
  ].includes(key)
}

function looksCurrencyKey(key) {
  return [
    'final_value',
    'absolute_return',
    'entry_price',
    'exit_price',
    'net_pnl',
    'equity',
  ].includes(key)
}

function looksIntegerKey(key) {
  return ['num_trades', 'size'].includes(key)
}

function formatNumericValue(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—'

  if (looksPercentKey(key)) {
    return `${(Number(value) * 100).toFixed(2)}%`
  }

  if (looksCurrencyKey(key)) {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency: 'USD',
      maximumFractionDigits: 2,
    }).format(Number(value))
  }

  if (looksIntegerKey(key)) {
    return String(Math.round(Number(value)))
  }

  const abs = Math.abs(Number(value))
  if (abs >= 1000) return Number(value).toLocaleString(undefined, { maximumFractionDigits: 2 })
  if (abs >= 1) return Number(value).toFixed(4)
  return Number(value).toFixed(6)
}

function valueTone(key, value) {
  if (value === null || value === undefined || Number.isNaN(value)) return ''
  if (
    ['cumulative_return', 'annualized_return_pct', 'absolute_return', 'net_pnl', 'sharpe_ratio', 'net_return'].includes(key)
  ) {
    return Number(value) > 0 ? 'positive' : Number(value) < 0 ? 'negative' : ''
  }
  if (['max_drawdown_pct', 'value_at_risk_95', 'conditional_loss_95'].includes(key)) {
    return Number(value) < 0 ? 'negative' : ''
  }
  return ''
}

function getProgressInfo({ submitting, runStatus, results, error }) {
  if (error) {
    return {
      progress: 100,
      phase: 'failed',
      message: error,
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: true },
        { label: 'Finalizing', done: false },
        { label: 'Failed', done: true },
      ],
    }
  }

  if (results) {
    return {
      progress: 100,
      phase: 'completed',
      message: `Run ${runStatus?.run_id || 'loaded'} completed successfully.`,
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: true },
        { label: 'Finalizing', done: true },
        { label: 'Completed', done: true },
      ],
    }
  }

  if (runStatus?.status === 'failed') {
    return {
      progress: 100,
      phase: 'failed',
      message: runStatus?.error || 'Run failed.',
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: true },
        { label: 'Finalizing', done: false },
        { label: 'Failed', done: true },
      ],
    }
  }

  if (runStatus?.status === 'completed') {
    return {
      progress: 92,
      phase: 'finalizing',
      message: 'Backend run completed. Preparing result payload...',
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: true },
        { label: 'Finalizing', done: true },
        { label: 'Completed', done: false },
      ],
    }
  }

  if (runStatus?.status === 'running') {
    return {
      progress: 58,
      phase: 'running',
      message: `Run ${runStatus.run_id} is executing forecasting and trading.`,
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: true },
        { label: 'Finalizing', done: false },
        { label: 'Completed', done: false },
      ],
    }
  }

  if (runStatus?.status === 'loading') {
    return {
      progress: 34,
      phase: 'loading',
      message: `Loading stored run ${runStatus.run_id}...`,
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: false },
        { label: 'Finalizing', done: false },
        { label: 'Completed', done: false },
      ],
    }
  }

  if (submitting) {
    return {
      progress: 12,
      phase: 'queued',
      message: 'Submitting run request...',
      steps: [
        { label: 'Submitted', done: true },
        { label: 'Running', done: false },
        { label: 'Finalizing', done: false },
        { label: 'Completed', done: false },
      ],
    }
  }

  return {
    progress: 0,
    phase: 'idle',
    message: 'Submit a run to see progress and results.',
    steps: [
      { label: 'Submitted', done: false },
      { label: 'Running', done: false },
      { label: 'Finalizing', done: false },
      { label: 'Completed', done: false },
    ],
  }
}

function renderCell(key, value) {
  if (value === null || value === undefined || value === '') {
    return <span className="cell-muted">—</span>
  }

  if (typeof value === 'number') {
    return (
      <span className={classNames('metric-cell', valueTone(key, value))}>
        {formatNumericValue(key, value)}
      </span>
    )
  }

  return <span>{String(value)}</span>
}

function MultiSelectField({ label, options, values, onChange, helper }) {
    const valueSet = new Set(values)
    const [open, setOpen] = useState(false)
  
    function toggle(option) {
      if (valueSet.has(option)) {
        onChange(values.filter((item) => item !== option))
        return
      }
      onChange([...values, option])
    }
  
    function selectAll() {
      onChange(options)
    }
  
    function clearAll() {
      onChange([])
    }
  
    const summaryText =
      values.length === 0
        ? 'None selected'
        : values.length <= 2
          ? values.map(formatLabel).join(', ')
          : `${values.length} selected`
  
    return (
      <div className="compact-select">
        <label className="field-label">{label}</label>
        {helper ? <p className="field-helper">{helper}</p> : null}
  
        <button
          type="button"
          className="compact-select-trigger"
          onClick={() => setOpen((current) => !current)}
        >
          <span className="compact-select-summary">{summaryText}</span>
          <span className="compact-select-caret">{open ? '▴' : '▾'}</span>
        </button>
  
        {open ? (
          <div className="compact-select-menu">
            <div className="compact-select-actions">
              <button type="button" className="ghost-button" onClick={selectAll}>All</button>
              <button type="button" className="ghost-button" onClick={clearAll}>Clear</button>
            </div>
  
            <div className="compact-select-options">
              {options.map((option) => (
                <label key={option} className="compact-select-option">
                  <input
                    type="checkbox"
                    checked={valueSet.has(option)}
                    onChange={() => toggle(option)}
                  />
                  <span>{formatLabel(option)}</span>
                </label>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    )
  }

function StatCard({ card }) {
  const tone = valueTone(card.key, card.value)
  return (
    <div className="stat-card">
      <div className="stat-label">{card.label}</div>
      <div className={classNames('stat-value', tone && `stat-value--${tone}`)}>
        {card.display}
      </div>
    </div>
  )
}

function ProgressBar({ info }) {
  return (
    <div className="progress-panel">
      <div className="progress-topline">
        <div>
          <div className="panel-title">Run Progress</div>
          <div className="panel-subtitle">{info.message}</div>
        </div>
        <div className={classNames('status-pill', `status-pill--${info.phase}`)}>
          {info.phase}
        </div>
      </div>

      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${info.progress}%` }} />
      </div>

      <div className="progress-steps">
        {info.steps.map((step) => (
          <div key={step.label} className={classNames('progress-step', step.done && 'progress-step--done')}>
            {step.label}
          </div>
        ))}
      </div>
    </div>
  )
}

function Tabs({ items, activeKey, onChange }) {
  return (
    <div className="tabs-row">
      {items.map((item) => (
        <button
          key={item.key}
          type="button"
          className={classNames('tab-button', activeKey === item.key && 'tab-button--active')}
          onClick={() => onChange(item.key)}
        >
          {item.label}
        </button>
      ))}
    </div>
  )
}

const GT_NAVY = '#003057'
const GT_GOLD = '#B3A369'
const GT_DARK_GOLD = '#857437'
const FISERV_ORANGE = '#FF6600'
const BRAND_INK = '#142033'
const BRAND_MUTED = '#667085'
const BRAND_SUCCESS = '#16833A'
const BRAND_RISK = '#C33A1F'

const CHART_PALETTE = [
  FISERV_ORANGE,
  GT_NAVY,
  GT_GOLD,
  '#7A542E',
  '#B94700',
  '#4C6A85',
  '#A28D5B',
  '#1E5B79',
]
  
function styleForSeries(series, index) {
  const key = String(series?.key || '').toLowerCase()
  const label = String(series?.label || '').toLowerCase()

  if (key === 'actual' || label === 'actual') {
    return {
      stroke: GT_NAVY,
      strokeWidth: 3.2,
      strokeDasharray: undefined,
    }
  }

  if (
    key === 'model_forecast' ||
    label.includes('lasso') ||
    label.includes('ridge') ||
    label.includes('elastic') ||
    label.includes('xgboost') ||
    label.includes('random forest')
  ) {
    return {
      stroke: FISERV_ORANGE,
      strokeWidth: 3,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('naive') || label.includes('naive')) {
    return {
      stroke: '#596B82',
      strokeWidth: 2.4,
      strokeDasharray: '8 5',
    }
  }

  if (key.includes('model_abs_error')) {
    return {
      stroke: BRAND_RISK,
      strokeWidth: 2.5,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('naive_abs_error')) {
    return {
      stroke: BRAND_MUTED,
      strokeWidth: 2.5,
      strokeDasharray: '6 4',
    }
  }

  if (key.includes('drawdown')) {
    return {
      stroke: BRAND_RISK,
      strokeWidth: 2.5,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('cumulative_return')) {
    return {
      stroke: BRAND_SUCCESS,
      strokeWidth: 2.5,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('equity')) {
    return {
      stroke: GT_NAVY,
      strokeWidth: 2.5,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('confidence')) {
    return {
      stroke: GT_DARK_GOLD,
      strokeWidth: 2.5,
      strokeDasharray: undefined,
    }
  }

  if (key.includes('net_return')) {
    return {
      stroke: FISERV_ORANGE,
      strokeWidth: 2,
      strokeDasharray: undefined,
    }
  }

  return {
    stroke: CHART_PALETTE[index % CHART_PALETTE.length],
    strokeWidth: 2,
    strokeDasharray: undefined,
  }
}
  
  function fillForSeries(series, index) {
    const key = String(series?.key || '').toLowerCase()
  
    if (key.includes('rmse') || key.includes('mae') || key.includes('mape')) {
      return '#2563eb'
    }
  
    return CHART_PALETTE[index % CHART_PALETTE.length]
  }

  const chartAxisStyle = { fontSize: 12, fill: BRAND_MUTED }

  const chartAxisLine = { stroke: '#D8D0B6' }
  
  const chartTooltipStyle = {
    borderRadius: 16,
    border: '1px solid #DED6BE',
    boxShadow: '0 18px 36px rgba(0, 48, 87, 0.12)',
  }

  const TRADING_CHART_KEYS = new Set([
    'equity_curve',
    'cumulative_return_curve',
    'drawdown_curve',
    'period_return_curve',
    'weights_curve',
    'confidence_curve',
  ])
  
  function isCoreTradingChart(chart) {
    return TRADING_CHART_KEYS.has(chart?.key)
  }

  function ChartCard({ chart }) {
    const hasRows = chart?.rows?.length > 0
    const isBar = chart?.chart_type === 'bar'
  
    return (
      <div className="content-card chart-card">
        <div className="panel-title">{chart?.title || 'Chart'}</div>
        {!hasRows ? (
          <div className="empty-state">No chart data available.</div>
        ) : (
          <div className="chart-wrapper">
            <ResponsiveContainer width="100%" height={340}>
              {isBar ? (
                <BarChart data={chart.rows}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E5DDC5" />
                <XAxis
                  dataKey={chart.x_key}
                  tick={chartAxisStyle}
                  axisLine={chartAxisLine}
                  tickLine={chartAxisLine}
                  minTickGap={20}
                />
                <YAxis
                  tick={chartAxisStyle}
                  axisLine={chartAxisLine}
                  tickLine={chartAxisLine}
                  width={72}
                />
                <Tooltip contentStyle={chartTooltipStyle} />
                <Legend />
                {chart.series.map((series, index) => (
                  <Bar
                    key={series.key}
                    dataKey={series.key}
                    name={series.label || formatLabel(series.key)}
                    fill={barColorForSeries(series, index)}
                    radius={[8, 8, 0, 0]}
                  />
                ))}
              </BarChart>
              ) : (
                <LineChart data={chart.rows}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E5DDC5" />
                  <XAxis
                    dataKey={chart.x_key}
                    tick={chartAxisStyle}
                    axisLine={chartAxisLine}
                    tickLine={chartAxisLine}
                    minTickGap={20}
                  />
                  <YAxis
                    tick={chartAxisStyle}
                    axisLine={chartAxisLine}
                    tickLine={chartAxisLine}
                    width={72}
                  />
                  <Tooltip contentStyle={chartTooltipStyle} />
                  <Legend />
                  {chart.series.map((series, index) => {
                    const style = styleForSeries(series, index)

                    return (
                      <Line
                        key={series.key}
                        type="monotone"
                        dataKey={series.key}
                        name={series.label || formatLabel(series.key)}
                        stroke={style.stroke}
                        strokeWidth={style.strokeWidth}
                        strokeDasharray={style.strokeDasharray}
                        dot={false}
                      />
                    )
                  })}
                </LineChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </div>
    )
  }

  function tableSemanticColumnKey(column) {
    const key = String(column?.key || '').toLowerCase()
    const label = String(column?.label || '').toLowerCase()
  
    if (key === 'panel' || key === 'panel_name' || label === 'panel') return 'panel'
    if (key === 'model' || key === 'model_name' || label === 'model') return 'model'
    if (key === 'feature_set' || key === 'features' || label === 'feature set') return 'feature_set'
    if (key === 'rmse' || label === 'rmse') return 'rmse'
    if (key === 'mae' || label === 'mae') return 'mae'
    if (key === 'mape' || label === 'mape') return 'mape'
    if (key === 'dir_acc' || key === 'directional_accuracy' || label.includes('directional')) return 'directional_accuracy'
    if (key === 'r2' || key === 'r_squared' || label === 'r²') return 'r2'
  
    return key
  }
  
  function tableColumnClassName(column) {
    const semanticKey = tableSemanticColumnKey(column)
  
    return classNames(
      'table-column',
      semanticKey && `table-column--${semanticKey}`,
    )
  }
  
  function shouldTruncateTableCell(column) {
    const semanticKey = tableSemanticColumnKey(column)
  
    return [
      'panel',
      'model',
      'feature_set',
      'strategy',
      'weights',
      'ticker_returns',
      'metadata',
    ].includes(semanticKey)
  }
  
  function renderTableCell(column, value) {
    const rendered = renderCell(column.key, value)
  
    if (!shouldTruncateTableCell(column)) {
      return rendered
    }
  
    const title = value === null || value === undefined ? '' : String(value)
  
    return (
      <span className="table-cell-truncate" title={title}>
        {rendered}
      </span>
    )
  }

  function DataTable({ table }) {
    if (!table) return null
  
    return (
      <div className="content-card">
        <div className="panel-title">{table.title}</div>
        {table.rows?.length ? (
          <div className="table-scroll">
            <table className="data-table">
            <colgroup>
              {table.columns.map((column) => (
                <col
                  key={column.key}
                  className={tableColumnClassName(column)}
                />
              ))}
            </colgroup>
              <thead>
                <tr>
                  {table.columns.map((column) => (
                    <th
                      key={column.key}
                      className={tableColumnClassName(column)}
                      title={column.label}
                    >
                      {column.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {table.rows.map((row, index) => (
                  <tr key={`${table.key}-${index}`}>
                    {table.columns.map((column) => (
                      <td
                        key={column.key}
                        className={tableColumnClassName(column)}
                      >
                        {renderTableCell(column, row[column.key])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">{table.empty_message || 'No data available.'}</div>
        )}
      </div>
    )
  }

function RunHistory({ runs, onSelect }) {
  return (
    <div className="content-card">
      <div className="panel-title">Recent Runs</div>
      {runs.length === 0 ? (
        <div className="empty-state">No runs yet.</div>
      ) : (
        <div className="run-history-list">
          {runs.map((run) => (
            <button key={run.run_id} type="button" className="run-history-item" onClick={() => onSelect(run.run_id)}>
              <div>
                <div className="run-id">{run.run_id}</div>
                <div className="run-meta-small">
                  {(run.selection?.panels || []).join(', ') || 'No panel selection'}
                </div>
              </div>
              <div className={classNames('status-pill', `status-pill--${run.status || 'idle'}`)}>
                {run.status}
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

function SelectionSummary({ selection }) {
  const groups = [
    { label: 'Panels', values: selection?.panels || [] },
    { label: 'Models', values: selection?.models || [] },
    { label: 'Feature Sets', values: selection?.feature_sets || [] },
  ]

  return (
    <div className="content-card">
      <div className="panel-title">Run Selection</div>
      <div className="selection-grid">
        {groups.map((group) => (
          <div key={group.label}>
            <div className="selection-label">{group.label}</div>
            <div className="selection-chips">
              {group.values.length ? (
                group.values.map((value) => (
                  <span key={value} className="selection-chip">
                    {formatLabel(value)}
                  </span>
                ))
              ) : (
                <span className="cell-muted">None</span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function RunMetadata({ runStatus, results }) {
  const meta = results?.metadata || {}
  const items = [
    { label: 'Run ID', value: runStatus?.run_id || '—' },
    { label: 'Pipeline Run ID', value: results?.pipeline_run_id || '—' },
    { label: 'Status', value: runStatus?.status || 'completed' },
    { label: 'Experiment Horizon', value: meta?.horizon ?? '—' },
    { label: 'Min Train Size', value: meta?.min_train_size ?? '—' },
    { label: 'Trials', value: meta?.n_trials ?? '—' },
  ]

  return (
    <div className="content-card">
      <div className="panel-title">Run Metadata</div>
      <div className="meta-grid">
        {items.map((item) => (
          <div key={item.label} className="meta-item">
            <div className="meta-label">{item.label}</div>
            <div className="meta-value">{String(item.value)}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [options, setOptions] = useState(null)
  const [form, setForm] = useState(null)
  const [results, setResults] = useState(null)
  const [runStatus, setRunStatus] = useState(null)
  const [recentRuns, setRecentRuns] = useState([])
  const [error, setError] = useState('')
  const [loadingOptions, setLoadingOptions] = useState(true)
  const [submitting, setSubmitting] = useState(false)

  const [activeMainTab, setActiveMainTab] = useState('overview')
  const [activeForecastTab, setActiveForecastTab] = useState('charts')
  const [activeTopModelPanel, setActiveTopModelPanel] = useState('')
  const strategyCatalog = useMemo(
    () => options?.strategies?.catalog || [],
    [options],
  )
  
  const selectedStrategy = useMemo(
    () => strategyCatalog.find((item) => item.id === form?.strategy_id) || strategyCatalog[0] || null,
    [strategyCatalog, form],
  )

  function updateStrategyParam(name, value) {
    setForm((current) => ({
      ...current,
      strategy_params: {
        ...(current.strategy_params || {}),
        [name]: value,
      },
    }))
  }

  const pollTimerRef = useRef(null)

  useEffect(() => {
    loadInitial()

    return () => {
      window.clearTimeout(pollTimerRef.current)
    }
  }, [])

  useEffect(() => {

    const firstTopModelPanel = results?.ui?.forecasting?.tables?.top_models_by_panel?.[0]?.key || ''
    if (firstTopModelPanel && !activeTopModelPanel) {
      setActiveTopModelPanel(firstTopModelPanel)
    }

    if (results) {
      setActiveMainTab('overview')
    }
  }, [results])

  const progressInfo = useMemo(
    () => getProgressInfo({ submitting, runStatus, results, error }),
    [submitting, runStatus, results, error],
  )

  const topModelTables = useMemo(
    () => results?.ui?.forecasting?.tables?.top_models_by_panel || [],
    [results],
  )

  const topModelTabs = useMemo(
    () =>
      topModelTables.map((table) => ({
        key: table.key,
        label: table.title.replace('Top Models — ', ''),
      })),
    [topModelTables],
  )

  const activeTopModelTable = useMemo(
    () => topModelTables.find((table) => table.key === activeTopModelPanel) || topModelTables[0] || null,
    [topModelTables, activeTopModelPanel],
  )

  async function loadInitial() {
    setLoadingOptions(true)
    setError('')
    try {
      const [optionsData, runsData] = await Promise.all([fetchOptions(), fetchRuns(12)])
      setOptions(optionsData)
      setForm(makeInitialForm(optionsData))
      setRecentRuns(runsData.runs || [])
    } catch (err) {
      setError(err.message || 'Failed to load dashboard options.')
    } finally {
      setLoadingOptions(false)
    }
  }

  function updateField(key, value) {
    setForm((current) => ({ ...current, [key]: value }))
  }

  async function refreshRunHistory() {
    try {
      const runsData = await fetchRuns(12)
      setRecentRuns(runsData.runs || [])
    } catch {
      // ignore
    }
  }

  async function handleSubmit(event) {
    event.preventDefault()
    if (!form) return

    setSubmitting(true)
    setError('')
    setResults(null)
    setRunStatus(null)
    setActiveTopModelPanel('')

    if (form.run_trading && selectedStrategy) {
        for (const param of selectedStrategy.parameters || []) {
          if (!param.required) continue
      
          const value = form.strategy_params?.[param.name]
      
          const missing =
            value === null ||
            value === undefined ||
            value === '' ||
            (Array.isArray(value) && value.length === 0)
      
          if (missing) {
            setSubmitting(false)
            setError(`Missing required strategy parameter: ${param.label}`)
            return
          }
        }
      }
      if (
        form.run_trading &&
        form.trade_start_date &&
        form.trade_end_date &&
        form.trade_start_date > form.trade_end_date
      ) {
        setSubmitting(false)
        setError('Trade start date must be earlier than or equal to trade end date.')
        return
      }
    const payload = {
        panels: form.panels,
        models: form.models.length ? form.models : ['all'],
        feature_sets: form.feature_sets.length ? form.feature_sets : ['all'],
        ranking_metric: form.ranking_metric,
        top_k: Number(form.top_k),
        run_trading: form.run_trading,
        trading_mode: form.trading_mode,
        trade_start_date: form.trade_start_date || null,
        trade_end_date: form.trade_end_date || null,
        strategy: selectedStrategy
          ? {
              source: selectedStrategy.source,
              file: selectedStrategy.file,
              class_name: selectedStrategy.class_name,
              name: selectedStrategy.class_name,
              params: coerceStrategyParams(selectedStrategy, form.strategy_params),
            }
          : undefined,
      }

    try {
      const created = await createRun(payload)
      const activeRunId = created.run_id

      setRunStatus({
        run_id: activeRunId,
        status: created.status || 'running',
      })

      await refreshRunHistory()
      await pollRun(activeRunId)
    } catch (err) {
      setError(err.message || 'Failed to start run.')
    } finally {
      setSubmitting(false)
    }
  }

  async function pollRun(runId) {
    const check = async () => {
      try {
        const status = await fetchRunStatus(runId)
        setRunStatus(status)

        if (status.status === 'failed') {
          setError(status.error || 'Run failed.')
          await refreshRunHistory()
          return
        }

        const result = await fetchRunResults(runId)
        if (!result.pending) {
          setResults(result)
          await refreshRunHistory()
          return
        }

        pollTimerRef.current = window.setTimeout(check, 1500)
      } catch (err) {
        setError(err.message || 'Polling failed.')
      }
    }

    await check()
  }

  async function handleLoadRun(runId) {
    setError('')
    setResults(null)
    setRunStatus({ run_id: runId, status: 'loading' })
    setActiveTopModelPanel('')

    try {
      const [status, result] = await Promise.all([fetchRunStatus(runId), fetchRunResults(runId)])
      setRunStatus(status)

      if (!result.pending) {
        setResults(result)
      } else {
        pollTimerRef.current = window.setTimeout(() => pollRun(runId), 1000)
      }
    } catch (err) {
      setError(err.message || 'Failed to load saved run.')
    }
  }

  if (loadingOptions || !form || !options) {
    return <div className="screen-state">Loading dashboard…</div>
  }

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="brand-bar">
          <div className="brand-lockup" aria-label="Georgia Tech and Fiserv partnership">
            {BRAND_LOGOS.map((logo, index) => (
              <div key={logo.name} className="brand-lockup-item">
                <BrandLogo logo={logo} />
                {index < BRAND_LOGOS.length - 1 ? <span className="brand-divider">×</span> : null}
              </div>
            ))}
          </div>

          <div className="brand-badge">GT × Fiserv analytics</div>
        </div>

        <div className="hero-grid">
          <div>
            <p className="eyebrow">Forecasting + Trading Dashboard</p>
            <h1>Macro Forecast Control Center</h1>
            <p className="hero-copy">
              Configure panels, models, feature sets, and trading strategy. Submit a run, watch progress,
              and review forecasting and trading outputs in charts and tables.
            </p>
          </div>

          <div className="hero-accent-card" aria-label="Dashboard capabilities">
            <div className="hero-accent-kicker">Pipeline</div>
            <div className="hero-accent-title">Forecast → Backtest → Risk</div>
            <div className="hero-accent-grid">
              <span>Macro panels</span>
              <span>Model ranking</span>
              <span>Strategy runs</span>
            </div>
          </div>
        </div>
      </header>
      <div className="page-stack">
  <section className="horizontal-section">
    <form className="content-card form-card config-strip" onSubmit={handleSubmit}>
      <div className="panel-title">Run Configuration</div>

      <div className="config-grid">
      <div className="config-block config-block--panels">
        <MultiSelectField
            label="Panels"
            options={options.panels || []}
            values={form.panels}
            onChange={(value) => updateField('panels', value)}
            helper="Choose one or more data panels to evaluate."
        />
        </div>

        <div className="config-block config-block--models">
        <MultiSelectField
            label="Models"
            options={options.models || []}
            values={form.models}
            onChange={(value) => updateField('models', value)}
            helper="Leave empty to run all available models."
        />
        </div>

        <div className="config-block config-block--features">
        <MultiSelectField
            label="Feature Sets"
            options={options.feature_sets || []}
            values={form.feature_sets}
            onChange={(value) => updateField('feature_sets', value)}
            helper="Leave empty to run all available feature sets."
        />
        </div>

        <div className="config-block config-block--ranking">
        <div className="input-group">
            <label className="field-label">Ranking Metric</label>
            <select
            value={form.ranking_metric}
            onChange={(e) => updateField('ranking_metric', e.target.value)}
            >
            <option value="mape">MAPE</option>
            <option value="rmse">RMSE</option>
            <option value="mae">MAE</option>
            </select>
        </div>

        <div className="input-group">
            <label className="field-label">Top K</label>
            <input
            type="number"
            min="1"
            max="20"
            value={form.top_k}
            onChange={(e) => updateField('top_k', e.target.value)}
            />
        </div>
        </div>

        <div className="config-block config-block--trading">
        <div className="panel-title panel-title--small">Trading Setup</div>

          <div className="checkbox-row">
            <input
              id="run_trading"
              type="checkbox"
              checked={form.run_trading}
              onChange={(e) => updateField('run_trading', e.target.checked)}
            />
            <label htmlFor="run_trading">Run trading strategy after forecasting</label>
          </div>

          <div className="trading-controls-grid">
            <div className="input-group">
              <label className="field-label">Strategy</label>
              <select
                value={form.strategy_id}
                onChange={(e) => {
                  const next = strategyCatalog.find((item) => item.id === e.target.value)
                  setForm((current) => ({
                    ...current,
                    strategy_id: e.target.value,
                    strategy_params: buildDefaultStrategyParams(next),
                  }))
                }}
                disabled={!form.run_trading}
              >
                {strategyCatalog.map((strategy) => (
                  <option key={strategy.id} value={strategy.id}>
                    {strategy.label}
                  </option>
                ))}
              </select>
            </div>

            <div className="input-group">
              <label className="field-label">Trading Mode</label>
              <select
                value={form.trading_mode}
                onChange={(e) => updateField('trading_mode', e.target.value)}
                disabled={!form.run_trading}
              >
                {(options.trading_modes || []).map((mode) => (
                  <option key={mode} value={mode}>{formatLabel(mode)}</option>
                ))}
              </select>
            </div>
          </div>
          <p className="field-helper trading-date-helper">
            Trading dates define the strategy evaluation window only. Forecasting still uses the full available FSBI history.
          </p>
          <div className="trading-dates-grid">
            <div className="input-group">
                <label className="field-label">Trade Start Date</label>
                <input
                type="date"
                value={form.trade_start_date}
                onChange={(e) => updateField('trade_start_date', e.target.value)}
                disabled={!form.run_trading}
                />
            </div>

            <div className="input-group">
                <label className="field-label">Trade End Date</label>
                <input
                type="date"
                value={form.trade_end_date}
                onChange={(e) => updateField('trade_end_date', e.target.value)}
                disabled={!form.run_trading}
                />
            </div>
            </div>

          {form.run_trading && selectedStrategy ? (
            <div className="strategy-params-card">
              <div className="panel-title panel-title--small">{selectedStrategy.label}</div>
              {selectedStrategy.description ? (
                <p className="panel-subtitle">{selectedStrategy.description}</p>
              ) : null}

              <div className="two-col-grid strategy-params-grid">
                {(selectedStrategy.parameters || []).map((param) => (
                  <StrategyParamField
                    key={param.name}
                    param={param}
                    value={form.strategy_params?.[param.name]}
                    onChange={updateStrategyParam}
                    disabled={!form.run_trading}
                  />
                ))}
              </div>
            </div>
          ) : null}
        </div>

        <div className="config-block config-block--actions">
          <button className="primary-button" type="submit" disabled={submitting || !form.panels.length}>
            {submitting ? 'Starting Run…' : 'Run Pipeline'}
          </button>
          <RunHistory runs={recentRuns} onSelect={handleLoadRun} />
        </div>
      </div>
    </form>
  </section>

  <section className="horizontal-section">
    <ProgressBar info={progressInfo} />
    {error ? <div className="error-banner">{error}</div> : null}
  </section>

  <section className="horizontal-section">
    {!results ? (
      <div className="content-card empty-results-card">
        <div className="panel-title">No results yet</div>
        <p className="panel-subtitle">
          Select panels, models, and feature sets, then start a run. Once it completes, the
          forecasting and trading outputs will render here.
        </p>
      </div>
    ) : (
      <div className="results-shell">
        <Tabs
          items={[
            { key: 'overview', label: 'Overview' },
            { key: 'forecasting', label: 'Forecasting' },
            ...(results.ui?.trading ? [{ key: 'trading', label: 'Trading' }] : []),
          ]}
          activeKey={activeMainTab}
          onChange={setActiveMainTab}
        />

        {activeMainTab === 'overview' && (
          <section className="section-stack">
            <div className="section-header">
              <div>
                <h2>Overview</h2>
                <p>High-level summary of the current run and selected configuration.</p>
              </div>
            </div>

            <div className="stats-grid">
              {(results.ui?.overview?.summary_cards || []).map((card) => (
                <StatCard key={card.key} card={card} />
              ))}
            </div>

            <SelectionSummary selection={results.selection} />
            <RunMetadata runStatus={runStatus} results={results} />
          </section>
        )}

        {activeMainTab === 'forecasting' && (
          <section className="section-stack">
            <div className="section-header">
              <div>
                <h2>Forecasting</h2>
                <p>Model ranking, evaluation metrics, and forecast comparison charts.</p>
              </div>
            </div>

            <div className="stats-grid">
              {(results.ui?.forecasting?.summary_cards || []).map((card) => (
                <StatCard key={card.key} card={card} />
              ))}
            </div>

            <Tabs
              items={[
                { key: 'charts', label: 'Charts' },
                { key: 'metrics', label: 'Metrics Table' },
                { key: 'top_models', label: 'Top Models' },
              ]}
              activeKey={activeForecastTab}
              onChange={setActiveForecastTab}
            />

            {activeForecastTab === 'charts' && (
              <>
                {(results.ui?.forecasting?.charts?.all || []).map((item, index) => (
                  <ChartCard
                    key={`${item.group}-${item.panel_name}-${index}`}
                    chart={item.chart}
                  />
                ))}
              </>
            )}

            {activeForecastTab === 'metrics' && (
              <DataTable table={results.ui?.forecasting?.tables?.metrics} />
            )}

            {activeForecastTab === 'top_models' && (
              <>
                {topModelTabs.length > 0 && (
                  <Tabs
                    items={topModelTabs}
                    activeKey={activeTopModelTable?.key || topModelTabs[0]?.key}
                    onChange={setActiveTopModelPanel}
                  />
                )}

                <DataTable table={activeTopModelTable} />
              </>
            )}
          </section>
        )}

        {activeMainTab === 'trading' && results.ui?.trading && (
        <section className="section-stack">
            <div className="section-header">
            <div>
                <h2>Trading</h2>
                <p>
                Strategy {results.ui.trading.header.strategy_name} on{' '}
                {(results.ui.trading.header.tickers || []).join(', ') || 'selected tickers'}
                </p>

                {results.ui?.trading?.header?.trade_window_start || results.ui?.trading?.header?.trade_window_end ? (
                <p className="panel-subtitle">
                    Trade window: {results.ui.trading.header.trade_window_start || '—'} to{' '}
                    {results.ui.trading.header.trade_window_end || '—'}
                </p>
                ) : null}
            </div>
            </div>

            <div className="stats-grid">
            {(results.ui.trading.summary_cards || []).map((card) => (
                <StatCard key={card.key} card={card} />
            ))}
            </div>

            {(results.ui?.trading?.charts?.all || [])
              .filter(isCoreTradingChart)
              .map((chart) => (
                <ChartCard key={chart.key} chart={chart} />
            ))}

          </section>
        )}
      </div>
    )}
  </section>
</div>
    </div>
  )
}