from pathlib import Path

from swift.ui.web_ui_restful.app import FRONTEND_DIR, create_app


def test_frontend_has_local_chartjs():
    chart_js = FRONTEND_DIR / 'chart.umd.min.js'
    index_html = (FRONTEND_DIR / 'index.html').read_text(encoding='utf-8')

    assert chart_js.is_file(), 'chart.umd.min.js must be vendored for offline deployments'
    assert chart_js.stat().st_size > 100_000
    assert 'cdn.jsdelivr.net' not in index_html
    assert './chart.umd.min.js' in index_html


def test_frontend_serves_vendored_chartjs():
    from fastapi.testclient import TestClient

    client = TestClient(create_app())
    response = client.get('/chart.umd.min.js')

    assert response.status_code == 200
    assert 'Chart' in response.text
