"""Scoped, bilingual rule-based answers backed by the existing business tools."""
from __future__ import annotations

import re

from src.api import service
from src.api.tools import TOOL_REGISTRY

FAMILIES = {
    "BEVERAGES": ("饮料", ["beverage", "drink", "饮料", "饮品"]),
    "BREAD/BAKERY": ("面包烘焙", ["bread", "bakery", "面包", "烘焙"]),
    "DAIRY": ("乳制品", ["dairy", "milk", "乳制品", "牛奶", "奶"]),
    "MEATS": ("肉类", ["meat", "肉类", "牛肉", "猪肉", "肉"]),
    "POULTRY": ("禽类", ["poultry", "chicken", "鸡肉", "禽类", "鸡"]),
    "PRODUCE": ("果蔬", ["produce", "vegetable", "fruit", "蔬菜", "水果", "果蔬", "生鲜"]),
    "SEAFOOD": ("海鲜", ["seafood", "fish", "海鲜", "鱼"]),
    "FROZEN FOODS": ("冷冻食品", ["frozen", "冷冻"]),
}


def _store(text):
    match = re.search(r"(?:\bstore\b|\bshop\b|门店|店)\s*#?\s*(\d+)|(?<!\d)(\d+)\s*号?店", text, re.I)
    return int(next(v for v in match.groups() if v is not None)) if match else None


def _family(text, available):
    for family in sorted(available, key=len, reverse=True):
        if family in text.upper():
            return family
    aliases = sorted(((alias, family) for family, (_, items) in FAMILIES.items()
                      if family in available for alias in items), key=lambda pair: len(pair[0]), reverse=True)
    return next((family for alias, family in aliases if alias.lower() in text.lower()), None)


def _scope(message, history, families):
    store, family = _store(message), _family(message, families)
    all_stores = bool(re.search(r"所有门店|全部门店|all stores", message, re.I))
    all_families = bool(re.search(r"所有品类|全部品类|all families", message, re.I))
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        previous = turn.get("content", "")
        if store is None and not all_stores:
            store = _store(previous)
        if family is None and not all_families:
            family = _family(previous, families)
    return None if all_stores else store, None if all_families else family


def run_fallback(message: str, reason: str, history: list[dict] | None = None) -> dict:
    scope = service.get_scope()
    store, family = _scope(message, history or [], scope["families"])
    text = message.lower()
    zh = bool(re.search(r"[\u4e00-\u9fff]", message))
    trace = []
    note = ("库存、保质期、交货期和采购倍数为模拟设定；销量沿用原始数据单位。"
            if zh else "Inventory, shelf-life, lead-time and order multiples are simulated. Sales retain source-data units.")

    def t(chinese, english):
        return chinese if zh else english

    def call(name, **args):
        result = TOOL_REGISTRY[name](**args)
        trace.append({"tool": name, "args": args, "result": result})
        return result

    def finish(reply):
        return {"reply": reply, "tool_trace": trace, "mode": "fallback", "fallback_reason": reason}

    def label(s, f):
        return f"门店 {s} · {FAMILIES.get(f, (f, []))[0]}（{f}）" if zh else f"Store {s} {f}"

    def actions(rows):
        if not rows:
            return t("当前范围没有符合条件的记录。", "No matching actions in this scope.")
        lines = []
        for row in rows:
            name = label(row['store_nbr'], row['family'])
            action = t("建议补货", "Reorder") if row['action'] == 'order_today' else t("建议观察", "Monitor")
            risk = t("高", "high") if row['waste_risk'] == 'high' else t("低", "low")
            lines.append(t(
                f"{name}\n{action}；计划采购量 {row['recommended_order_qty']:,.0f}。7 天预测 {row['forecast_7d']:,.1f}，模拟库存 {row['current_stock_simulated']:,.1f}，补货点 {row['reorder_point']:,.1f}。交货期 {row['lead_time_days']} 天，报废风险{risk}。",
                f"{name}: {action}; planned order {row['recommended_order_qty']:,.0f}. 7-day forecast {row['forecast_7d']:,.1f}, simulated stock {row['current_stock_simulated']:,.1f}, reorder point {row['reorder_point']:,.1f}, lead time {row['lead_time_days']}d, waste risk {risk}."))
        return "\n\n".join(lines)

    if store is not None and store not in scope['stores']:
        call('list_scope')
        return finish(t(f"目前没有门店 {store} 的数据。可选门店：{scope['stores']}。请指定其中一家。",
                        f"Store {store} is not available. Available stores: {scope['stores']}."))

    promotion = any(w in text for w in ['uplift', 'promotion', 'promo', 'what if', 'what-if', '促销', '需求增加', '需求下降', '需求减少', '需求提升'])
    promotion = promotion or ('%' in text and any(w in text for w in ['增加', '下降', '减少', '提升', 'increase', 'decrease']))
    if promotion:
        # A discount is not a measured demand uplift.
        if any(w in text for w in ['打折', '折扣', 'discount', '打八折', '打九折']):
            return finish(t("折扣比例不能直接换算为需求变化。请给出假设，例如“需求增加 20%”，再比较补货量。",
                            "A discount is not a demand uplift. Specify an assumed demand change, such as +20%."))
        match = re.search(r'([+-]?\d+(?:\.\d+)?)\s*[%％]', text)
        uplift = float(match.group(1)) if match else 20.0
        if any(w in text for w in ['下降', '减少', '降低', 'decrease', 'decreases', 'drop']):
            uplift = -abs(uplift)
        if not -100 <= uplift <= 500:
            return finish(t("演示支持 -100% 至 +500% 的需求变化，请调整假设。", "Use a demand change from -100% to +500%."))
        targets = service.get_replenishment(store_nbr=store, family=family, limit=1)
        if not targets:
            return finish(t("当前范围没有可计算的数据。", "No data in this scope."))
        target = targets[0]
        sim = call('simulate_promotion', store_nbr=target['store_nbr'], family=target['family'], uplift_pct=uplift)['simulation']
        title = label(sim['store_nbr'], sim['family'])
        return finish(t(
            f"{title} · 假设需求变化 {uplift:+g}%\n\n7 天需求：{sim['baseline_forecast_7d']:,.1f} → {sim['simulated_forecast_7d']:,.1f}\n建议采购量：{sim['baseline_order_qty']:,} → {sim['simulated_order_qty']:,}（变化 {sim['order_delta']:+,}）\n\n这是按输入比例计算的情景，不是模型测得的促销效果。\n{note}",
            f"{title}: demand scenario {uplift:+g}%\n7-day demand: {sim['baseline_forecast_7d']:,.1f} -> {sim['simulated_forecast_7d']:,.1f}\nRecommended order: {sim['baseline_order_qty']:,} -> {sim['simulated_order_qty']:,}\nThis is an assumed scenario. {note}"))

    if any(w in text for w in ['metric', 'mae', 'wape', 'accuracy', 'backtest', '指标', '准确', '误差']):
        metrics = call('get_forecast_metrics')['metrics']
        wape = metrics.get('wape')
        ratio = f"{wape:.2%}" if wape is not None else t('不可用', 'unavailable')
        return finish(t(f"全局基线回测：WAPE {ratio}，MAE {metrics.get('mae')}，共 {metrics.get('n_predictions')} 个预测点。\nWAPE 是绝对误差总和 / 实际销量总和，并非准确率或缺货率。结果来自一次 7 天的历史回测。",
                        f"Global baseline backtest: WAPE {ratio}, MAE {metrics.get('mae')}, {metrics.get('n_predictions')} predictions. WAPE is total absolute error divided by total actual sales; it is not an accuracy or stockout rate."))

    is_risk = any(w in text for w in ['risk', 'stockout', 'waste', 'shortage', 'expire', '风险', '缺货', '报废', '过期'])
    is_order = any(w in text for w in ['reorder', 'order', 'replenish', '补货', '补多少', '订货', '采购', '下单', '为什么', 'why'])
    if is_risk or is_order:
        result = call('get_replenishment_plan', store_nbr=store, family=family, limit=24)
        rows = result['actions']
        if is_risk:
            risk_field = 'waste_risk' if any(w in text for w in ['waste', 'expire', '报废', '过期']) else 'stockout_risk'
            rows = [r for r in rows if r[risk_field] == 'high']
        return finish(actions(rows) + '\n\n' + note)

    if any(w in text for w in ['inventory', 'on hand', '库存', '存货']):
        targets = service.get_replenishment(store_nbr=store, family=family)
        if len(targets) != 1:
            call('list_scope')
            return finish(t("请选择一家门店和一个品类，再查看具体库存。", "Select one store and family to inspect inventory."))
        row = targets[0]
        inv = call('get_inventory_status', store_nbr=row['store_nbr'], family=row['family'])['inventory']
        return finish(t(f"{label(row['store_nbr'], row['family'])}\n模拟库存 {inv['current_stock_simulated']:,.1f}，补货点 {inv['reorder_point']:,.1f}，安全库存 {inv['safety_stock']:,.1f}。交货期 {inv['lead_time_days']} 天，保质期 {inv['shelf_life_days']} 天。\n\n{note}",
                        f"{label(row['store_nbr'], row['family'])}: stock {inv['current_stock_simulated']}, reorder point {inv['reorder_point']}, safety stock {inv['safety_stock']}. {note}"))

    if any(w in text for w in ['forecast', 'demand', 'predict', 'sales', '预测', '需求', '销量']):
        match = re.search(r'(\d+)\s*(?:天|days?)', text)
        horizon = int(match.group(1)) if match else 7
        if not 1 <= horizon <= 7:
            return finish(t("当前结果只覆盖 7 天，请选择 1–7 天。", "Current artifacts cover only 7 days; choose 1–7."))
        result = call('get_demand_forecast', store_nbr=store, family=family, horizon_days=horizon)
        rows = result['forecast']
        if not rows:
            return finish(t("当前范围没有预测结果。", "No forecast in this scope."))
        if len({(r['store_nbr'], r['family']) for r in rows}) != 1:
            return finish(t(f"已查询 {len(rows)} 条预测。请选择一个门店与品类查看每日明细，避免混合不同品类的销量单位。",
                            f"Retrieved {len(rows)} forecast rows. Select one store-family for a daily breakdown."))
        daily = '\n'.join(f"{str(r['date'])[:10]}：{r['forecast_sales']:,.1f}" for r in rows)
        return finish(t(f"{label(rows[0]['store_nbr'], rows[0]['family'])} · {horizon} 天预测\n\n{daily}\n\n季节朴素基线，采用上一周销量模式。{note}",
                        f"{label(rows[0]['store_nbr'], rows[0]['family'])}: {horizon}-day forecast\n{daily}\n{note}"))

    if any(w in text for w in ['scope', 'coverage', 'which store', '范围', '覆盖', '哪些门店', '有哪些品类']):
        result = call('list_scope')['scope']
        return finish(t(f"可用门店：{result['stores']}\n食品品类：{', '.join(FAMILIES.get(f, (f, []))[0] for f in result['families'])}",
                        f"Stores: {result['stores']}\nFamilies: {', '.join(result['families'])}"))

    return finish(t("当前为规则备用模式。我可以查询预测、解释补货、查看库存与风险，或按指定百分比比较需求变化。试试“为什么需要补货？”或“需求增加 20% 会怎样？”。",
                    "Rule-based mode supports forecast, replenishment, inventory, risks, metrics and demand-change scenarios. Try: What should I reorder?"))
