"""Fact-grounded explanations for replenishment actions.

This module intentionally uses deterministic text, so the demo remains usable
without an LLM API key. A future LLM UI can receive the exact same facts and
must be instructed not to add unsupported operational claims.
"""

import pandas as pd


def explain_actions(plan: pd.DataFrame) -> list[dict]:
    explanations = []
    for row in plan.to_dict("records"):
        action = "Place an order today" if row["action"] == "order_today" else "Monitor inventory"
        reason = (
            f"Forecast demand is {row['forecast_7d']:.0f} over the next 7 days "
            f"({row['forecast_daily']:.0f}/day). Simulated current stock is "
            f"{row['current_stock_simulated']:.0f}, versus a reorder point of "
            f"{row['reorder_point']:.0f}; supplier lead time is {row['lead_time_days']} day(s)."
        )
        tradeoff = (
            f"Shelf-life scenario: {row['shelf_life_days']} day(s). "
            f"Waste risk is {row['waste_risk']}."
        )
        explanations.append({
            "store_nbr": row["store_nbr"],
            "family": row["family"],
            "action": row["action"],
            "recommended_order_qty": row["recommended_order_qty"],
            "summary": action,
            "reason": reason,
            "tradeoff": tradeoff,
            "evidence_boundary": row["assumption_note"],
        })
    return explanations
