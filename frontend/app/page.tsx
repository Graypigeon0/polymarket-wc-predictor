'use client'

import useSWR from 'swr'
import { supabase } from '@/lib/supabase'
import clsx from 'clsx'

type Pnl = {
  n_bets: number
  n_resolved: number
  n_pending: number
  n_won: number | null
  hit_rate: number | null
  total_profit: number
  roi: number | null
}

async function fetchPnl(): Promise<Pnl> {
  // First alerts per market
  const { data: edges, error } = await supabase
    .from('edges')
    .select('market_id, pm_prob, alerted_at')
    .eq('alerted', true)
    .order('alerted_at', { ascending: true })
  if (error) throw error
  const first: Record<string, { pm_prob: number }> = {}
  for (const e of edges ?? []) {
    if (!(e.market_id in first)) first[e.market_id] = { pm_prob: e.pm_prob }
  }
  const ids = Object.keys(first)
  if (ids.length === 0) {
    return { n_bets: 0, n_resolved: 0, n_pending: 0, n_won: 0,
             hit_rate: null, total_profit: 0, roi: null }
  }
  const { data: mkts } = await supabase
    .from('polymarket_markets')
    .select('id, resolved, resolution_outcome')
    .in('id', ids)
  let resolved = 0, won = 0, profit = 0, staked = 0
  for (const m of mkts ?? []) {
    if (!m.resolved) continue
    resolved++; staked += 1
    const pm = first[m.id].pm_prob
    if (m.resolution_outcome === 'Yes' && pm > 0) {
      profit += (1 - pm) / pm; won++
    } else {
      profit -= 1
    }
  }
  return {
    n_bets: ids.length,
    n_resolved: resolved,
    n_pending: ids.length - resolved,
    n_won: resolved > 0 ? won : null,
    hit_rate: resolved > 0 ? won / resolved : null,
    total_profit: Math.round(profit * 100) / 100,
    roi: staked > 0 ? Math.round((profit / staked) * 1000) / 1000 : null,
  }
}

type EdgeRow = {
  market_id: string
  market_type: string
  market_label: string
  description: string
  event_slug: string | null
  model_prob: number
  pm_prob: number
  edge: number
  computed_at: string
}

// Pull recent edges and keep the latest one per market.
async function fetchEdges(): Promise<EdgeRow[]> {
  // Only fetch edges computed in the last 2 hours so stale rows don't pollute.
  const cutoff = new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()

  const { data, error } = await supabase
    .from('edges')
    .select(`
      market_id, model_prob, pm_prob, edge, computed_at,
      polymarket_markets!inner ( market_type, outcome_label, description, event_slug )
    `)
    .gte('computed_at', cutoff)
    .order('computed_at', { ascending: false })
    .limit(5000)
  if (error) throw error

  // Dedupe to latest row per market (already ordered DESC, first wins).
  const seen = new Set<string>()
  const out: EdgeRow[] = []
  for (const r of data ?? []) {
    if (seen.has(r.market_id)) continue
    seen.add(r.market_id)
    const pm = (r as any).polymarket_markets
    out.push({
      market_id: r.market_id,
      market_type: pm?.market_type ?? 'other',
      market_label: pm?.outcome_label ?? r.market_id,
      description: pm?.description ?? '',
      event_slug: pm?.event_slug ?? null,
      model_prob: r.model_prob,
      pm_prob: r.pm_prob,
      edge: r.edge,
      computed_at: r.computed_at,
    })
  }
  // Keep only positive edges, sort by edge DESC.
  return out.filter(e => e.edge > 0).sort((a, b) => b.edge - a.edge)
}

const TYPE_LABELS: Record<string, string> = {
  group_winner: 'Group Winners',
  outright:     'World Cup Winner',
  stage_advance:'Stage Advancement',
  match_1x2:    'Match Markets',
}

// Extract just the team name from "FIFA World Cup Group X Winner — Team"
function shortLabel(label: string): string {
  const m = label.match(/—\s*(.+)$/)
  return m ? m[1].trim() : label
}

// Group letter from "Group X" in the label, if any.
function groupOf(label: string): string | null {
  const m = label.match(/Group\s+([A-L])/)
  return m ? m[1] : null
}

function PnlCard() {
  const { data } = useSWR('pnl', fetchPnl, { refreshInterval: 5 * 60_000 })
  if (!data || data.n_bets === 0) return null
  const profitColor =
    data.total_profit > 0 ? 'text-green-400'
    : data.total_profit < 0 ? 'text-red-400'
    : 'text-neutral-300'
  return (
    <section className="mb-6 rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-400 mb-2">
        Model P&L <span className="text-neutral-600 font-normal">(flat $1 stakes)</span>
      </h2>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div><span className="text-neutral-500">Bets placed:</span> {data.n_bets}</div>
        <div><span className="text-neutral-500">Resolved:</span> {data.n_resolved} / {data.n_pending} pending</div>
        <div><span className="text-neutral-500">Hit rate:</span> {data.hit_rate !== null ? `${(data.hit_rate * 100).toFixed(0)}%` : '—'}</div>
        <div className={profitColor}>
          <span className="text-neutral-500">Profit:</span> ${data.total_profit.toFixed(2)}
          {data.roi !== null && ` (${(data.roi * 100).toFixed(1)}% ROI)`}
        </div>
      </div>
    </section>
  )
}

type DriftRow = {
  market: string
  alert_pm: number
  latest_pm: number
  drift_pp: number
  direction: 'toward' | 'away' | 'flat'
}

type DriftSummary = {
  n_markets: number
  n_toward: number
  n_away: number
  pct_toward: number
  avg_signed_drift_pp: number
  per_market: DriftRow[]
}

async function fetchDrift(): Promise<DriftSummary> {
  // Re-implements the Python drift computation in the browser.
  const { data: edges } = await supabase
    .from('edges').select('market_id, model_prob, pm_prob, alerted_at')
    .eq('alerted', true).order('alerted_at', { ascending: true })
  const first: Record<string, { model: number; alert_pm: number }> = {}
  for (const e of edges ?? []) {
    if (!(e.market_id in first))
      first[e.market_id] = { model: e.model_prob, alert_pm: e.pm_prob }
  }
  const ids = Object.keys(first)
  if (ids.length === 0)
    return { n_markets: 0, n_toward: 0, n_away: 0, pct_toward: 0,
             avg_signed_drift_pp: 0, per_market: [] }

  // For each, fetch latest price + label
  const { data: mkts } = await supabase
    .from('polymarket_markets').select('id, outcome_label').in('id', ids)
  const labels: Record<string, string> = {}
  for (const m of mkts ?? []) labels[m.id] = m.outcome_label ?? m.id

  const rows: DriftRow[] = []
  let toward = 0, away = 0, totalSigned = 0
  for (const id of ids) {
    const { data: pr } = await supabase
      .from('polymarket_prices').select('price')
      .eq('market_id', id).order('captured_at', { ascending: false }).limit(1)
    const latest = pr?.[0]?.price
    if (latest == null) continue
    const { model, alert_pm } = first[id]
    const drift = latest - alert_pm
    const signTowards = model > alert_pm ? 1 : -1
    const signed = drift * signTowards
    const direction: 'toward'|'away'|'flat' =
      Math.abs(signed) < 0.005 ? 'flat' : signed > 0 ? 'toward' : 'away'
    if (direction === 'toward') toward++
    else if (direction === 'away') away++
    totalSigned += signed
    rows.push({
      market: labels[id] ?? id.slice(0, 24),
      alert_pm: Math.round(alert_pm * 1000) / 1000,
      latest_pm: Math.round(latest * 1000) / 1000,
      drift_pp: Math.round(drift * 1000) / 10,
      direction,
    })
  }
  const n = rows.length
  rows.sort((a, b) => b.drift_pp - a.drift_pp)
  return {
    n_markets: n,
    n_toward: toward, n_away: away,
    pct_toward: n > 0 ? toward / n : 0,
    avg_signed_drift_pp: n > 0 ? Math.round((totalSigned / n * 100) * 10) / 10 : 0,
    per_market: rows,
  }
}

function DriftCard() {
  const { data } = useSWR('drift', fetchDrift, { refreshInterval: 10 * 60_000 })
  if (!data || data.n_markets === 0) return null
  const towardColor =
    data.pct_toward > 0.55 ? 'text-green-400' :
    data.pct_toward < 0.45 ? 'text-red-400' : 'text-neutral-300'
  return (
    <section className="mb-6 rounded-lg border border-neutral-800 bg-neutral-900 p-3">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-400 mb-2">
        Line drift since alert
      </h2>
      <div className="grid grid-cols-2 gap-2 text-sm mb-2">
        <div className={towardColor}>
          <span className="text-neutral-500">Toward us:</span> {(data.pct_toward * 100).toFixed(0)}%
          <span className="text-neutral-600 text-xs"> ({data.n_toward}/{data.n_markets})</span>
        </div>
        <div className={data.avg_signed_drift_pp > 0 ? 'text-green-400' : data.avg_signed_drift_pp < 0 ? 'text-red-400' : ''}>
          <span className="text-neutral-500">Avg drift:</span> {data.avg_signed_drift_pp > 0 ? '+' : ''}{data.avg_signed_drift_pp.toFixed(1)} pp
        </div>
      </div>
      <p className="text-xs text-neutral-500">
        Market moving toward our model = good (we found mispricing early).
        Moving away = we may have been wrong.
      </p>
    </section>
  )
}

export default function Page() {
  const { data, error, isLoading } = useSWR('edges', fetchEdges, {
    refreshInterval: 60_000,
  })

  // Bucket by market type, preserving the global edge-DESC order within each.
  const bucketed: Record<string, EdgeRow[]> = {}
  for (const row of data ?? []) {
    const key = row.market_type in TYPE_LABELS ? row.market_type : 'other'
    if (!bucketed[key]) bucketed[key] = []
    bucketed[key].push(row)
  }

  const sectionOrder = ['group_winner', 'outright', 'match_1x2', 'stage_advance', 'other']

  return (
    <>
      <header className="mb-4">
        <h1 className="text-2xl font-bold">WC 2026 Edges</h1>
        <p className="text-sm text-neutral-400">
          Live model vs. Polymarket — auto-refreshes every minute
        </p>
      </header>
      <PnlCard />
      <DriftCard />

      {isLoading && <p className="text-neutral-400">Loading…</p>}
      {error && (
        <p className="text-red-400 text-sm">Failed to load: {String(error)}</p>
      )}

      {sectionOrder.map(key => {
        const rows = bucketed[key]
        if (!rows || rows.length === 0) return null
        return (
          <section key={key} className="mb-6">
            <h2 className="text-sm font-semibold uppercase tracking-wide text-neutral-400 mb-2">
              {TYPE_LABELS[key] ?? key}{' '}
              <span className="text-neutral-600 font-normal">({rows.length})</span>
            </h2>
            <ul className="space-y-2">
              {rows.map(row => {
                const grp = groupOf(row.market_label)
                const polymarketUrl = row.event_slug
                  ? `https://polymarket.com/event/${row.event_slug}`
                  : 'https://polymarket.com/sports/fifa-world-cup/games'
                return (
                  <li
                    key={row.market_id}
                    className="rounded-lg border border-neutral-800 bg-neutral-900 p-3"
                  >
                    <a
                      href={polymarketUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block"
                    >
                      <div className="flex justify-between items-start gap-2">
                        <span className="font-medium leading-tight">
                          {grp && (
                            <span className="text-neutral-500 mr-2">Group {grp}</span>
                          )}
                          {shortLabel(row.market_label)}
                        </span>
                        <span
                          className={clsx(
                            'text-sm font-bold whitespace-nowrap',
                            row.edge >= 0.15
                              ? 'text-green-400'
                              : row.edge >= 0.08
                                ? 'text-yellow-300'
                                : 'text-neutral-300'
                          )}
                        >
                          +{(row.edge * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="mt-1 text-xs text-neutral-400 flex gap-3">
                        <span>Model {(row.model_prob * 100).toFixed(1)}%</span>
                        <span>Market {(row.pm_prob * 100).toFixed(1)}%</span>
                      </div>
                    </a>
                  </li>
                )
              })}
            </ul>
          </section>
        )
      })}

      {data?.length === 0 && (
        <p className="text-neutral-500 text-sm">No positive edges right now.</p>
      )}
    </>
  )
}
