import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import os
import sqlite3
import base64
import io
from PIL import Image

# 1. Mock heavy libraries AND OpenAI class
with patch.dict('sys.modules', {
    'vllm': MagicMock(),
    'mlx_vlm': MagicMock(),
    'mlx_lm': MagicMock(),
    'torch': MagicMock(),
    'transformers': MagicMock(),
    'peft': MagicMock(),
    'bitsandbytes': MagicMock(),
    'openai': MagicMock(),
}):
    import vlm_server

client = TestClient(vlm_server.app)

@pytest.fixture
def test_db():
    db_name = "test_rlhf.db"
    if os.path.exists(db_name):
        os.remove(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS tuples (id INTEGER PRIMARY KEY, timestamp TEXT, image_base64 TEXT, a11y_tree TEXT, bad_action TEXT, good_action TEXT, processed INTEGER DEFAULT 0)")
    conn.commit()
    conn.close()
    old_db = vlm_server.DB_PATH
    vlm_server.DB_PATH = db_name
    yield db_name
    vlm_server.DB_PATH = old_db
    if os.path.exists(db_name):
        os.remove(db_name)

def test_vlm_act_endpoint(test_db):
    img = Image.new('RGB', (1, 1), color='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    payload = {"image_base64": img_b64, "a11y_tree": "{}"}
    vlm_server.model_engine = "vllm"
    vlm_server.llm = MagicMock()
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text='{"action": "click", "target_bbox": [1, 2, 3, 4]}')]
    vlm_server.llm.generate.return_value = [mock_output]
    response = client.post("/vlm/act", json=payload)
    assert response.status_code == 200
    assert response.json()["action"] == "click"
    vlm_server.model_engine = None

def test_rlhf_log_endpoint(test_db):
    payload = {"timestamp": "123", "state_image_base64": "img", "state_a11y_tree": "tree", "vlm_bad_action": "bad", "human_good_action": "good"}
    response = client.post("/vlm/rlhf_log", json=payload)
    assert response.status_code == 200
    conn = sqlite3.connect(test_db)
    assert conn.execute("SELECT good_action FROM tuples").fetchone()[0] == "good"
    conn.close()

def test_osint_analyze_endpoint():
    payload = {"objective": "obj", "raw_data": "raw"}
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="Brief Content"))]
    vlm_server.client.chat.completions.create.return_value = mock_resp
    response = client.post("/osint/analyze", json=payload)
    assert response.status_code == 200
    assert "report_path" in response.json()
    if os.path.exists(response.json()["report_path"]):
        os.remove(response.json()["report_path"])
