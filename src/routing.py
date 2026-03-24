from __future__ import annotations


def classify_question(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["forecast", "store", "department", "brand"]):
        return "forecast"
    if any(k in q for k in ["inventory", "reorder", "stock"]):
        return "inventory"
    if any(k in q for k in ["site", "location", "expansion"]):
        return "site"
    if any(k in q for k in ["drift", "monitor", "retrain", "watch"]):
        return "monitoring"
    return "docs"


def route_question(query: str, storage, retriever):
    qtype = classify_question(query)
    if qtype == "forecast":
        df = storage.read_sql("SELECT * FROM dashboard_department_forecast ORDER BY forecast_units DESC LIMIT 5")
        return {"route": "structured_forecast", "payload": df}
    if qtype == "inventory":
        df = storage.read_sql("SELECT * FROM inventory_recommendations ORDER BY recommended_reorder_qty DESC LIMIT 10")
        return {"route": "structured_inventory", "payload": df}
    if qtype == "site":
        df = storage.read_sql("SELECT * FROM optimized_site_selection ORDER BY projected_value_index DESC LIMIT 10")
        return {"route": "structured_site", "payload": df}
    if qtype == "monitoring":
        df = storage.read_sql("SELECT * FROM drift_monitor ORDER BY ks_stat DESC")
        return {"route": "structured_monitoring", "payload": df}
    return {"route": "document_retrieval", "payload": retriever.search(query, top_k=3)}
