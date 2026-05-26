'use client'

import useSWR from 'swr'
import { supabase } from '@/lib/supabase'
import clsx from 'clsx'

type EdgeRow = {
  id: number
  market_id: string
  market_label: string
  model_prob: number
  pm_prob: number
  edge: number
  computed_at: string
}

async function fetchEdges(): Promise<EdgeRow[]> {
  const { data, error } = await supabase
    .from('edges')
    .select(`
      id, market_id, model_prob, pm_prob, edge, computed_at,
      polymarket_markets!inner ( outcome_label, description )
    `)
    .gt('edge', 0)
    .order('edge', { ascending: false })
    .limit(50)
  if (error) throw error
  return (data ?? []).map((r: any) => ({
    id: r.id,
    market_id: r.market_id,
    market_label: r.polymarket_markets?.outcome_label ?? r.market_id,
    model_prob: r.model_prob,
    pm_prob: r.pm_prob,
    edge: r.edge,
    computed_at: r.computed_at
  }))
}

export default function Page() {
  const { data, error, isLoading } = useSWR('edges', fetchEdges, {
    refreshInterval: 60_000
  })

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

      <ul className="space-y-2">
        {(data ?? []).map((row) => (
          <li
            key={row.id}
            className="rounded-lg border border-neutral-800 bg-neutral-900 p-3"
          >
            <div className="flex justify-between items-start gap-2">
              <span className="font-medium leading-tight">{row.market_label}</span>
              <span
                className={clsx(
                  'text-sm font-bold whitespace-nowrap',
                  row.edge >= 0.05
                    ? 'text-green-400'
                    : row.edge >= 0.02
                      ? 'text-yellow-300'
                      : 'text-neutral-300'
                )}
              >
                +{(row.edge * 100).toFixed(1)}%
              </span>
            </div>
            <div className="mt-1 text-xs text-neutral-400 flex gap-3">
              <span>Model {(row.model_prob * 100).toFixed(1)}%</span>
              <span>PM {(row.pm_prob * 100).toFixed(1)}%</span>
            </div>
          </li>
        ))}
        {data?.length === 0 && (
          <li className="text-neutral-500 text-sm">No positive edges right now.</li>
        )}
      </ul>
    </>
  )
}
