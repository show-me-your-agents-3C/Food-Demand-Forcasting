"""Regressions for the interactive dashboard's real API and scoped questions."""
import pytest
from fastapi.testclient import TestClient

from src.api import agent, service
from src.api.main import app

client = TestClient(app)


def test_dashboard_assets_and_forecast_data():
    page = client.get('/')
    assert page.status_code == 200
    assert '需求与补货计划' in page.text
    assert client.get('/assets/app.js').status_code == 200
    assert client.get('/assets/styles.css').status_code == 200
    rows = client.get('/api/forecast', params={'store_nbr': 1, 'family': 'DAIRY'}).json()
    assert len(rows) == 7
    assert sum(row['forecast_sales'] for row in rows) == 3961


@pytest.mark.parametrize('question', [
    '今天门店1牛奶补货', '今天门店1牛奶需要补多少货？',
    '门店1牛奶有哪些缺货风险？', 'Risk for store 1 DAIRY',
])
def test_questions_preserve_scope(question):
    result = agent.run_fallback(question, 'test')
    trace = result['tool_trace'][0]
    assert trace['tool'] == 'get_replenishment_plan'
    assert trace['args']['store_nbr'] == 1
    assert trace['args']['family'] == 'DAIRY'
    assert all(row['store_nbr'] == 1 and row['family'] == 'DAIRY' for row in trace['result']['actions'])


def test_unknown_store_is_not_silently_replaced():
    result = agent.run_fallback('Forecast demand for store 999 DAIRY', 'test')
    assert 'not available' in result['reply']
    assert [t['tool'] for t in result['tool_trace']] == ['list_scope']


def test_page_context_and_follow_up_reach_chat_endpoint(monkeypatch):
    monkeypatch.setattr(agent, 'llm_configured', lambda: False)
    result = client.post('/api/chat', json={
        'message': '如果需求增加20%，补货量会怎样变化？',
        'history': [{'role': 'user', 'content': '当前页面选中：门店 1，品类 DAIRY。'}],
    }).json()
    simulation = result['tool_trace'][0]['result']['simulation']
    assert simulation['store_nbr'] == 1
    assert simulation['family'] == 'DAIRY'
    assert simulation['simulated_order_qty'] == 4220


def test_demand_decrease_and_chinese_horizon():
    result = agent.run_fallback('What if store 1 DAIRY demand decreases 20%?', 'test')
    assert result['tool_trace'][0]['result']['simulation']['uplift_pct'] == -20
    forecast = agent.run_fallback('门店1牛奶未来3天销量预测', 'test')
    assert forecast['tool_trace'][0]['result']['row_count'] == 3


def test_percent_does_not_select_a_store():
    result = agent.run_fallback('What if demand increases 2%?', 'test', history=[
        {'role': 'user', 'content': 'Store 1 DAIRY forecast'},
    ])
    assert result['tool_trace'][0]['args']['store_nbr'] == 1


def test_metrics_answer_does_not_claim_accuracy():
    result = agent.run_fallback('预测误差是多少？', 'test')
    assert '绝对误差总和' in result['reply']
    assert '17.42%' in result['reply']


def test_partial_artifacts_do_not_pass_health_check(tmp_path, monkeypatch):
    (tmp_path / 'replenishment_plan.csv').write_bytes((service.output_dir() / 'replenishment_plan.csv').read_bytes())
    monkeypatch.setattr(service, 'OUTPUT_DIR', tmp_path)
    assert client.get('/health').json()['data_available'] is False


def test_placeholder_gateway_is_not_ready(monkeypatch):
    monkeypatch.setenv('LLM_GATEWAY_URL', 'https://replace-with-your-llm-gateway')
    monkeypatch.setenv('LLM_GATEWAY_API_KEY', 'replace-with-your-api-key')
    monkeypatch.setenv('LLM_MODEL', 'replace-with-your-model-name')
    assert not agent.llm_configured()
