# Module 1 Report — Notebook → Production Service

## 1. Baseline metrics
- **Validation MAE:**  3.80 min
- **Validation RMSE:** 5.94 min
- **RMSE/MAE ratio:** 1.56 (Gaussian baseline ≈ 1.25)
- **Error distribution:** p50 2.5 min, p95 12.1 min
- **Split:** trained Nov 2024, evaluated Dec 2024 (temporal — future data)

## 2. Serialization decision
| Format | Human-readable | Cross-language | Schema-enforced | Safe from untrusted |
|---|---|---|---|---|
| **JSON** | ✅ text | ✅ every language | ❌ no schema in the format | ✅ no code exec on parse |
| **Protobuf** | ❌ binary | ✅ generated code, many languages | ✅ `.proto` definition | ✅ bounded, validated decode |
| **Pickle** | ❌ binary | ❌ Python-only | ❌ arbitrary | ❌ **executes code on load** |
| **ONNX** | ❌ binary (protobuf) | ✅ runtime ecosystem (ONNX Runtime: C++, Python, Java, JS…) | ✅ schema + versioned | ✅ no code exec on load |



## 3. Latency benchmark
pickle vs ONNX — mean, p95 on 500 rows

```
[INFO]: 20:09:53+0200 - benchmark_pk_onnx_latency - pickle  mean 0.036 ms | p95 0.041 ms
[INFO]: 20:09:53+0200 - benchmark_pk_onnx_latency - onnx    mean 0.014 ms | p95 0.017 ms
```

## 4. Image size
single-stage vs multi-stage

I am using docker bake to target individual stages. If only a single stage exists it would be the builder, so the comparison is between builder and runtime stages

```
IMAGE                                               ID             DISK USAGE   CONTENT SIZE   EXTRA
ghcr.io/badrmoh/mlops-practitioner/prodml:builder   2816d4cc2109        648MB          142MB        
ghcr.io/badrmoh/mlops-practitioner/prodml:runtime   a5a745daad59        561MB          117MB   
```


## 5. Maturity self-assessment
Level: 1