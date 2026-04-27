const JSON_HEADERS = { 'Content-Type': 'application/json' }

async function parseJson(response) {
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    throw new Error(data?.error || `Request failed with status ${response.status}`)
  }
  return data
}

export async function fetchOptions() {
  const response = await fetch('/api/options')
  return parseJson(response)
}

export async function createRun(payload) {
  const response = await fetch('/api/runs', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: JSON.stringify(payload),
  })
  return parseJson(response)
}

export async function fetchRunStatus(runId) {
  const response = await fetch(`/api/runs/${runId}`)
  return parseJson(response)
}

export async function fetchRunResults(runId) {
  const response = await fetch(`/api/runs/${runId}/results`)
  if (response.status === 202) {
    const data = await response.json().catch(() => ({}))
    return { pending: true, ...data }
  }
  return parseJson(response)
}

export async function fetchRuns(limit = 10) {
  const response = await fetch(`/api/runs?limit=${limit}`)
  return parseJson(response)
}