"use client";

import { useEffect, useState, useCallback, type CSSProperties } from "react";

type Market = { id: string; label: string; active?: boolean };
type Position = {
  id: string;
  market_id: string;
  label: string;
  side: string;
  entry_price: number;
  stake: number;
  status: string;
  realized_pnl: number | null;
  current: number | null;
  retPct: number | null;
  upnl: number | null;
};
type Summary = {
  bets: number;
  open: number;
  closed: number;
  staked: number;
  realized: number;
  unrealized: number;
  total: number;
  roi: number;
  winRate: number;
  wins: number;
};

const C = {
  bg: "#0b0e14",
  card: "#151a24",
  cardAlt: "#1b2230",
  line: "#252d3d",
  text: "#e6e9ef",
  muted: "#8b93a7",
  green: "#34d399",
  red: "#f87171",
  accent: "#5b8cff",
};

const money = (n: number) => `${n >= 0 ? "+" : "-"}$${Math.abs(n).toFixed(2)}`;
const pct = (n: number) => `${n >= 0 ? "+" : ""}${n.toFixed(1)}%`;

export default function BetsPage() {
  const [positions, setPositions] = useState<Position[]>([]);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [loading, setLoading] = useState(true);

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Market[]>([]);
  const [picked, setPicked] = useState<Market | null>(null);
  const [side, setSide] = useState("YES");
  const [price, setPrice] = useState("");
  const [stake, setStake] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [msg, setMsg] = useState<{ t: string; ok: boolean } | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const r = await fetch("/api/bets", { cache: "no-store" });
      const j = await r.json();
      if (j.positions) setPositions(j.positions);
      if (j.summary) setSummary(j.summary);
    } catch {
      /* ignore */
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // debounced market search
  useEffect(() => {
    if (picked) return;
    if (query.trim().length < 2) {
      setResults([]);
      return;
    }
    const t = setTimeout(async () => {
      try {
        const r = await fetch(`/api/markets?q=${encodeURIComponent(query)}`, {
          cache: "no-store",
        });
        const j = await r.json();
        setResults(j.markets || []);
      } catch {
        setResults([]);
      }
    }, 300);
    return () => clearTimeout(t);
  }, [query, picked]);

  async function submit() {
    setMsg(null);
    if (!picked) return setMsg({ t: "Pick a market first", ok: false });
    const p = Number(price);
    const s = Number(stake);
    if (!(p > 0 && p < 1))
      return setMsg({ t: "Price must be between 0 and 1", ok: false });
    if (!(s > 0)) return setMsg({ t: "Stake must be positive", ok: false });

    setSubmitting(true);
    try {
      const r = await fetch("/api/bets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          market_id: picked.id,
          label: picked.label,
          side,
          entry_price: p,
          stake: s,
        }),
      });
      const j = await r.json();
      if (!r.ok) {
        setMsg({ t: j.error || "Failed", ok: false });
      } else {
        setMsg({ t: "Bet logged", ok: true });
        setPicked(null);
        setQuery("");
        setPrice("");
        setStake("");
        setSide("YES");
        load();
      }
    } catch {
      setMsg({ t: "Network error", ok: false });
    }
    setSubmitting(false);
  }

  const input: CSSProperties = {
    width: "100%",
    boxSizing: "border-box",
    background: C.cardAlt,
    border: `1px solid ${C.line}`,
    borderRadius: 10,
    color: C.text,
    fontSize: 16, // keeps iOS from zooming on focus
    padding: "13px 14px",
    outline: "none",
  };
  const label: CSSProperties = {
    fontSize: 12,
    color: C.muted,
    marginBottom: 6,
    display: "block",
    letterSpacing: 0.3,
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: C.bg,
        color: C.text,
        fontFamily:
          "ui-sans-serif, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        padding: "20px 16px 64px",
        maxWidth: 520,
        margin: "0 auto",
      }}
    >
      <h1 style={{ fontSize: 22, fontWeight: 700, margin: "4px 0 18px" }}>
        My Bets
      </h1>

      {/* summary card */}
      <div
        style={{
          background: C.card,
          border: `1px solid ${C.line}`,
          borderRadius: 14,
          padding: 16,
          marginBottom: 22,
        }}
      >
        {summary ? (
          <>
            <div
              style={{
                fontSize: 30,
                fontWeight: 800,
                color: summary.total >= 0 ? C.green : C.red,
              }}
            >
              {money(summary.total)}
            </div>
            <div style={{ color: C.muted, fontSize: 13, marginBottom: 14 }}>
              total earnings · ROI {pct(summary.roi)}
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px 22px" }}>
              <Stat k="Staked" v={`$${summary.staked.toFixed(0)}`} />
              <Stat
                k="Realized"
                v={money(summary.realized)}
                c={summary.realized >= 0 ? C.green : C.red}
              />
              <Stat
                k="Unrealized"
                v={money(summary.unrealized)}
                c={summary.unrealized >= 0 ? C.green : C.red}
              />
              <Stat
                k="Win rate"
                v={`${summary.winRate.toFixed(0)}% (${summary.wins}/${summary.closed})`}
              />
              <Stat k="Open" v={`${summary.open}`} />
            </div>
          </>
        ) : (
          <div style={{ color: C.muted }}>{loading ? "Loading…" : "No bets yet."}</div>
        )}
      </div>

      {/* add bet form */}
      <div
        style={{
          background: C.card,
          border: `1px solid ${C.line}`,
          borderRadius: 14,
          padding: 16,
          marginBottom: 22,
        }}
      >
        <div style={{ fontWeight: 600, marginBottom: 14 }}>Log a bet</div>

        <label style={label}>MARKET</label>
        {picked ? (
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              background: C.cardAlt,
              border: `1px solid ${C.accent}`,
              borderRadius: 10,
              padding: "12px 14px",
              marginBottom: 14,
            }}
          >
            <span style={{ fontSize: 14 }}>{picked.label}</span>
            <button
              onClick={() => setPicked(null)}
              style={{
                background: "none",
                border: "none",
                color: C.muted,
                fontSize: 18,
                cursor: "pointer",
              }}
            >
              ✕
            </button>
          </div>
        ) : (
          <>
            <input
              style={{ ...input, marginBottom: results.length ? 0 : 14 }}
              placeholder="Search… e.g. Senegal"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            {results.length > 0 && (
              <div
                style={{
                  border: `1px solid ${C.line}`,
                  borderTop: "none",
                  borderRadius: "0 0 10px 10px",
                  marginBottom: 14,
                  maxHeight: 220,
                  overflowY: "auto",
                }}
              >
                {results.map((m) => (
                  <div
                    key={m.id}
                    onClick={() => {
                      setPicked(m);
                      setResults([]);
                    }}
                    style={{
                      padding: "12px 14px",
                      borderTop: `1px solid ${C.line}`,
                      fontSize: 14,
                      cursor: "pointer",
                      background: C.cardAlt,
                    }}
                  >
                    {m.label}
                  </div>
                ))}
              </div>
            )}
          </>
        )}

        <div style={{ display: "flex", gap: 10, marginBottom: 14 }}>
          <div style={{ flex: 1 }}>
            <label style={label}>SIDE</label>
            <div style={{ display: "flex", gap: 8 }}>
              {["YES", "NO"].map((s) => (
                <button
                  key={s}
                  onClick={() => setSide(s)}
                  style={{
                    flex: 1,
                    padding: "12px 0",
                    borderRadius: 10,
                    border: `1px solid ${side === s ? C.accent : C.line}`,
                    background: side === s ? C.accent : C.cardAlt,
                    color: side === s ? "#fff" : C.text,
                    fontSize: 15,
                    fontWeight: 600,
                    cursor: "pointer",
                  }}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
          <div style={{ flex: 1 }}>
            <label style={label}>ENTRY PRICE (0–1)</label>
            <input
              style={input}
              inputMode="decimal"
              placeholder="0.32"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
            />
          </div>
          <div style={{ flex: 1 }}>
            <label style={label}>STAKE ($)</label>
            <input
              style={input}
              inputMode="decimal"
              placeholder="20"
              value={stake}
              onChange={(e) => setStake(e.target.value)}
            />
          </div>
        </div>

        <button
          onClick={submit}
          disabled={submitting}
          style={{
            width: "100%",
            padding: "14px 0",
            borderRadius: 10,
            border: "none",
            background: submitting ? C.line : C.accent,
            color: "#fff",
            fontSize: 16,
            fontWeight: 700,
            cursor: "pointer",
          }}
        >
          {submitting ? "Saving…" : "Log bet"}
        </button>

        {msg && (
          <div
            style={{
              marginTop: 12,
              fontSize: 14,
              color: msg.ok ? C.green : C.red,
            }}
          >
            {msg.t}
          </div>
        )}
      </div>

      {/* positions list */}
      <div style={{ fontWeight: 600, margin: "0 4px 12px" }}>
        Positions {positions.length ? `(${positions.length})` : ""}
      </div>
      {positions.map((p) => {
        const up = p.status === "open";
        const ret = up ? p.retPct : null;
        const pnl = up ? p.upnl : p.realized_pnl;
        const color =
          pnl == null ? C.muted : pnl >= 0 ? C.green : C.red;
        return (
          <div
            key={p.id}
            style={{
              background: C.card,
              border: `1px solid ${C.line}`,
              borderRadius: 12,
              padding: "13px 14px",
              marginBottom: 10,
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              opacity: up ? 1 : 0.7,
            }}
          >
            <div style={{ minWidth: 0 }}>
              <div
                style={{
                  fontSize: 14,
                  fontWeight: 600,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {p.side} · {p.label}
              </div>
              <div style={{ fontSize: 12, color: C.muted, marginTop: 3 }}>
                ${Number(p.stake).toFixed(0)} @ {Number(p.entry_price).toFixed(3)}
                {up && p.current != null ? ` → ${p.current.toFixed(3)}` : ""}
                {!up ? " · settled" : ""}
              </div>
            </div>
            <div style={{ textAlign: "right", marginLeft: 12 }}>
              <div style={{ fontSize: 15, fontWeight: 700, color }}>
                {pnl == null ? "—" : money(pnl)}
              </div>
              {ret != null && (
                <div style={{ fontSize: 12, color }}>{pct(ret)}</div>
              )}
            </div>
          </div>
        );
      })}
      {!loading && positions.length === 0 && (
        <div style={{ color: C.muted, fontSize: 14, padding: "8px 4px" }}>
          No positions yet — log your first bet above.
        </div>
      )}
    </div>
  );
}

function Stat({ k, v, c }: { k: string; v: string; c?: string }) {
  return (
    <div>
      <div style={{ fontSize: 11, color: C.muted, letterSpacing: 0.3 }}>{k}</div>
      <div style={{ fontSize: 15, fontWeight: 600, color: c || C.text }}>{v}</div>
    </div>
  );
}
