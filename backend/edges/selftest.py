"""
Self-test for the match_1x2 edge calculation path.

Since real match-1X2 markets don't exist on Polymarket until close to kickoff,
this synthetically inserts a test market + price + cleans up, exercising the
exact code path that will fire when real match markets appear.

Run with:
  python -m backend.main selftest-match-1x2

Returns:
  - PASS if a sensible edge row gets computed and matches the input
  - FAIL with a diagnostic message otherwise
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from backend.db.client import get_client
from backend.edges import calculator

log = structlog.get_logger()


async def run_match_1x2_selftest() -> dict[str, Any]:
    """
    Exercise the match_1x2 edge path end-to-end with synthetic data.

    Steps:
      1. Find an upcoming WC2026 fixture between two teams that both have
         match_predictions rows (so the calculator can look up a model_prob).
      2. Insert a fake polymarket_markets row for the home team's "Yes" outcome,
         and a polymarket_prices row at 0.40 price.
      3. Run the edge calculator.
      4. Read back the resulting edges row; verify it matches expectations.
      5. Clean up all rows we created.
    """
    db = get_client()
    test_token = f"selftest-{uuid.uuid4().hex[:16]}"
    summary: dict[str, Any] = {"test_token": test_token}

    try:
        # 1. Find a fixture with a model prediction available
        preds = (db.table("match_predictions")
                 .select("match_id,p_home,p_away,model_version")
                 .limit(1)
                 .execute()).data
        if not preds:
            return {"status": "SKIP",
                    "reason": "no match_predictions rows yet (run match-model workflow first)"}
        pred = preds[0]
        match = (db.table("matches")
                 .select("id,home_id,away_id,competition")
                 .eq("id", pred["match_id"])
                 .single()
                 .execute()).data
        if not match:
            return {"status": "FAIL", "reason": "linked match row missing"}

        home_id = match["home_id"]
        expected_model_prob = float(pred["p_home"])
        # Choose a price that creates a known edge sign for testing
        synthetic_pm_price = max(0.01, expected_model_prob - 0.10)
        summary.update({
            "match_id": match["id"], "home_id": home_id,
            "expected_model_prob": round(expected_model_prob, 3),
            "synthetic_pm_price":  round(synthetic_pm_price, 3),
            "expected_edge":       round(expected_model_prob - synthetic_pm_price, 3),
        })

        # 2. Insert synthetic market + price
        db.table("polymarket_markets").insert({
            "id":            test_token,
            "market_type":   "match_1x2",
            "description":  f"[SELFTEST] {match['competition']} home win",
            "outcome_label": "[SELFTEST] Match 1X2 home win",
            "team_id":       home_id,
            "active":        True,
        }).execute()
        db.table("polymarket_prices").insert({
            "market_id":  test_token,
            "price":      synthetic_pm_price,
            "bid":        synthetic_pm_price,
            "ask":        synthetic_pm_price,
            "book_depth": 1000.0,
        }).execute()

        # 3. Run calculator (it will compute edges for all market types
        # including our synthetic match_1x2)
        await calculator.recompute_all()

        # 4. Read back the newest edge row for our synthetic token
        edge_rows = (db.table("edges")
                     .select("model_prob,pm_prob,edge,model_version,alerted")
                     .eq("market_id", test_token)
                     .order("computed_at", desc=True)
                     .limit(1)
                     .execute()).data

        if not edge_rows:
            return {**summary, "status": "FAIL",
                    "reason": "calculator did not produce an edge row for the synthetic match_1x2 market"}
        edge = edge_rows[0]
        summary["actual_model_prob"] = round(float(edge["model_prob"]), 3)
        summary["actual_pm_prob"]    = round(float(edge["pm_prob"]), 3)
        summary["actual_edge"]       = round(float(edge["edge"]), 3)
        summary["model_version"]     = edge["model_version"]

        # 5. Validation: model_prob should match what we expected
        if abs(edge["model_prob"] - expected_model_prob) > 0.001:
            return {**summary, "status": "FAIL",
                    "reason": f"model_prob mismatch: got {edge['model_prob']}, expected {expected_model_prob}"}
        # match_1x2 has no overround adjustment, so pm_prob should equal synthetic_pm_price
        if abs(edge["pm_prob"] - synthetic_pm_price) > 0.001:
            return {**summary, "status": "FAIL",
                    "reason": f"pm_prob mismatch: got {edge['pm_prob']}, expected {synthetic_pm_price}"}
        expected_edge = expected_model_prob - synthetic_pm_price
        if abs(edge["edge"] - expected_edge) > 0.001:
            return {**summary, "status": "FAIL",
                    "reason": f"edge mismatch: got {edge['edge']}, expected {expected_edge}"}

        return {**summary, "status": "PASS",
                "message": "match_1x2 path computes correct edge end-to-end"}

    finally:
        # Cleanup — always run, even on failure
        try:
            db.table("edges").delete().eq("market_id", test_token).execute()
            db.table("polymarket_prices").delete().eq("market_id", test_token).execute()
            db.table("polymarket_markets").delete().eq("id", test_token).execute()
        except Exception as e:
            log.warning("selftest.cleanup_failed", error=str(e))
