import { createClient } from "@supabase/supabase-js";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(req: Request) {
  const url = process.env.SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_KEY;
  if (!url || !key) {
    return NextResponse.json(
      { error: "Server missing SUPABASE_SERVICE_KEY" },
      { status: 500 }
    );
  }
  const q = (new URL(req.url).searchParams.get("q") || "").trim();
  if (q.length < 2) return NextResponse.json({ markets: [] });

  const supabase = createClient(url, key);
  const { data, error } = await supabase
    .from("polymarket_markets")
    .select("id, description, outcome_label, active")
    .or(`outcome_label.ilike.%${q}%,description.ilike.%${q}%`)
    .limit(25);

  if (error) return NextResponse.json({ error: error.message }, { status: 500 });

  const markets = (data || []).map((m) => ({
    id: m.id,
    label: (m.outcome_label || m.description || m.id).slice(0, 90),
    active: m.active,
  }));
  return NextResponse.json({ markets });
}
