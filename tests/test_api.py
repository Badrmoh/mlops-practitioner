def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_happy_path(client, sample_features):
    resp = client.post("/predict", json={"data": sample_features[0]})
    assert resp.status_code == 200
    assert resp.json()["Prediction"] is not None


def test_invalid_payload_returns_422(client):
    payload = {"data": {"PU_DO": "260_193"}}  # missing Trip_Distance
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 422


def test_response_schema_matches(client, sample_features):
    resp = client.post("/predict", json={"data": sample_features[0]})
    body = resp.json()
    assert set(body.keys()) == {"PU_DO", "Trip_Distance", "Prediction"}