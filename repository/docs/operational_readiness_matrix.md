# Operational Readiness Matrix

| Area | Status | Laptop | Server | Remaining work |
|---|---:|---:|---:|---|
| Repo hygiene | Ready | yes | yes | none critical |
| Requirements/build contract | Ready | yes | yes | clean server build |
| Stage 1/2 | Ready scaffold | yes | yes | real video validation |
| Stage 3/4/5 COLMAP/dense | Partial | limited | required | real heavy run |
| Stage 6 DA3 | Optional scaffold | skip-safe | optional | real model integration |
| Stage 7 cleanup | Ready in core env | yes | yes | tune thresholds |
| Stage 7.5 VLM QA | Mock/local scaffold | mock | real Qwen | endpoint validation |
| Stage 7.6 viewer export | Package-ready | yes | yes | optional Potree/Cesium |
| Stage 7.7 CAMS-GS | Prepared-only | yes | optional | real 3DGS training |
| Stage 8 metric alignment | Hardened scaffold | yes | yes | real anchors/visual obs |
| Stage 8 registration | Partial | limited | required | real IFC/scan validation |
| Stage 9 progress | Hardened scaffold | yes | yes | real mini-case metrics |
| Stage 10 copilot | Hardened scaffold | mock | real Qwen | inspect audit JSON |
| Strict server gate | Ready | non-strict | strict | GPU/model/input checks |
| Scientific evaluation | Not complete | no | required | baselines, ablation, uncertainty |

## Key policy

Metric truth remains Stage 8/9. VLM and 3DGS are evidence/explanation layers only.
