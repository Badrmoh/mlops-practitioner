# Module 1 Report — Notebook → Production Service

## 1. Baseline metrics
- **Validation MAE:**  4.17 min
- **Validation RMSE:** 6.31 min
- **RMSE/MAE ratio:** 1.51 (Gaussian baseline ≈ 1.25)
- **Error distribution:** p50 2.8 min, p95 13.1 min
- **Split:** trained Nov 2024, evaluated Dec 2024 (temporal — future data)

## 2. Serialization decision *(Step 4)*
| Format | Human-readable | Cross-language | Schema-enforced | Safe from untrusted |
|---|---|---|---|---|
| JSON | | | | |
| Protobuf | | | | |
| Pickle | | | | |
| ONNX | | | | |
Decision: __

## 3. Latency benchmark *(Step 4)*
pickle vs ONNX — mean, p95 on 500 rows

## 4. Image size *(Step 7)*
single-stage vs multi-stage; .dockerignore on/off

## 5. Maturity self-assessment *(Step 8)*
Level: __ / Missing to reach next level: __