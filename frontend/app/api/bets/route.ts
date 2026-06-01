import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

function db() {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) return null;
  return createClient(url, key);
}

function sidePrice(side: string, yes: number): number {
  return side.toUpperCase() === "YES" ? yes : 1 - yes;
}

export async function GET() {
  const supabase = db();
  if (!supabase) {
    return NextResponse.json(
      { error: "Server missing SUPABASE_SERVICE_KEY" },
      { status: 500 }
    );
  }

  const { data: positions, error } = await supabase
    .from("positions")
    .select("*")
    .order("opened_at", { ascending: false });
  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const open = (positions || []).filter((p) => p.status === "open");
  const ids = [...new Set(open.map((p) => p.market_id))];

  // latest pm_prob per market from the edges table
  const priceMap: Record<string, number> = {};
  if (ids.length) {
    const { data: edges } = await supabase
      .from("edges")
      .select("market_id, pm_prob, computed_at")
      .in("market_id", ids)
      .order("computed_at", { ascending: false });
    for (const e of edges || []) {
      if (e.market_id && priceMap[e.market_id] === undefined && e.pm_prob != null) {
        priceMap[e.market_id] = Number(e.pm_prob);
      }
    }
  }

  let realized = 0;
  let unrealized = 0;
  let staked = 0;
  let openN = 0;
  let closedN = 0;
  let wins = 0;

  const enriched = (positions || []).map((p) => {
    const stake = Number(p.stake);
    const entry = Number(p.entry_price);
    staked += stake;
    let current: number | null = null;
    let retPct: number | null = null;
    let upnl: number | null = null;

    if (p.status === "closed") {
      closedN += 1;
      const r = Number(p.realized_pnl || 0);
      realized += r;
      if (r > 0) wins += 1;
    } else {
      openN += 1;
      const yes = priceMap[p.market_id];
      if (yes !== undefined) {
        current = sidePrice(p.side, yes);
        retPct = (current / entry - 1) * 100;
        upnl = stake * (current / entry - 1);
        unrealized += upnl;
      }
    }
    return { ...p, current, retPct, upnl };
  });

  const total = realized + unrealized;
  const summary = {
    bets: (positions || []).length,
    open: openN,
    closed: closedN,
    staked,
    realized,
    unrealized,
    total,
    roi: staked ? (total / staked) * 100 : 0,
    winRate: closedN ? (wins / closedN) * 100 : 0,
    wins,
  };

  return NextResponse.json({ positions: enriched, summary });
}

export async function POST(req: Request) {
  const supabase = db();
  if (!supabase) {
    return NextResponse.json(
      { error: "Server missing SUPABASE_SERVICE_KEY" },
      { status: 500 }
    );
  }
  let body: any;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON" }, { status: 400 });
  }

  const entry = Number(body.entry_price);
  const stake = Number(body.stake);
  const side = (body.side || "YES").toUpperCase();

  if (!body.market_id) {
    return NextResponse.json({ error: "Pick a market first" }, { status: 400 });
  }
  if (!(entry > 0 && entry < 1)) {
    return NextResponse.json(
      { error: "Entry price must be between 0 and 1" },
      { status: 400 }
    );
  }
  if (!(stake > 0)) {
    return NextResponse.json({ error: "Stake must be positive" }, { status: 400 });
  }
  if (side !== "YES" && side !== "NO") {
    return NextResponse.json({ error: "Side must be YES or NO" }, { status: 400 });
  }

  const { data, error } = await supabase
    .from("positions")
    .insert({
      market_id: body.market_id,
      label: (body.label || "").slice(0, 80),
      side,
      entry_price: entry,
      stake,
      status: "open",
      last_alert_band: 0,
      notes: (body.notes || "").slice(0, 300),
    })
    .select()
    .single();

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });
  return NextResponse.json({ position: data });
}
