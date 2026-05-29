'use client'

import useSWR from 'swr'
import { supabase } from '@/lib/supabase'
import clsx from 'clsx'

type EdgeRow = {
  market_id: string
  market_type: string
  market_label: string
  description: string
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
      polymarket_markets!inner ( market_type, outcome_label, description )
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
                const polymarketUrl = `https://polymarket.com/market/${row.market_id}`
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
