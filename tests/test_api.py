import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def client(container):
    with TestClient(create_app(container=container)) as test_client:
        yield test_client


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["nlu"] == "rules" and body["records"] == 1000


def test_chat_roundtrip(client):
    payload = {"text": "Поставь задачу купить молоко", "session_id": "api-test", "user_id": "api", "timezone": "Europe/Moscow"}
    body = client.post("/api/chat", json=payload).json()
    assert body["intent"] == "create_task" and body["items"][0]["title"] == "Купить молоко"
    assert body["items"][0]["when"] == "без срока"


def test_chat_validation(client):
    assert client.post("/api/chat", json={"text": "x", "session_id": "bad id!"}).status_code == 422
    assert client.post("/api/chat", json={"text": "x" * 1001, "session_id": "ok"}).status_code == 422


def test_agenda_and_search(client):
    agenda = client.get("/api/agenda", params={"start": "2026-10-12", "end": "2026-10-18"}).json()
    assert agenda["items"]
    assert client.get("/api/agenda", params={"start": "2026-10-18", "end": "2026-10-12"}).status_code == 422
    assert client.get("/api/search", params={"q": "презентация"}).json()


def test_calendar_export(client):
    response = client.get("/api/calendar.ics")
    assert response.headers["content-type"].startswith("text/calendar")
    assert response.text.startswith("BEGIN:VCALENDAR") and response.text.count("BEGIN:VEVENT") == 1000


def test_index_page(client):
    assert "Голосовой планировщик" in client.get("/").text


def agenda_ids(client, user_id):
    items = client.get("/api/agenda", params={"user_id": user_id, "start": "2026-10-01", "end": "2026-12-31"}).json()["items"]
    return {item["id"]: item for item in items}


def create(client, user_id, text):
    body = client.post("/api/chat", json={"text": text, "session_id": f"s-{user_id}", "user_id": user_id}).json()
    return body["items"][0]["id"]


def test_delete_record(client):
    record_id = create(client, "api", "Каждый понедельник в 10 планёрка")
    assert client.delete(f"/api/records/{record_id}", params={"user_id": "other"}).status_code == 404
    response = client.delete(f"/api/records/{record_id}", params={"user_id": "api"})
    assert response.status_code == 204 and record_id not in agenda_ids(client, "api")
    assert client.delete(f"/api/records/{record_id}", params={"user_id": "api"}).status_code == 404


def test_complete_record(client):
    record_id = create(client, "api", "Поставь задачу купить молоко до конца декабря")
    assert client.post(f"/api/records/{record_id}/complete", params={"user_id": "other"}).status_code == 404
    body = client.post(f"/api/records/{record_id}/complete", params={"user_id": "api"}).json()
    assert body["id"] == record_id and body["status"] == "done"
    assert client.post("/api/records/999999/complete", params={"user_id": "api"}).status_code == 404
