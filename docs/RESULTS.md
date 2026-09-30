# Verified results

## Experiment 1 — higher-data Elo estimation

| Model | MAE | RMSE | R² |
| --- | ---: | ---: | ---: |
| CatBoost | 109.2 | 143.3 | 0.802 |
| XGBoost tuned | 112.7 | 142.7 | 0.803 |
| Ridge | 117.4 | 146.8 | 0.792 |
| OOF blend | 102.7 | 131.8 | 0.832 |
| Stacking Ridge / ElasticNet | **100.1** | 128.8–128.9 | **0.840** |

The report treats stacking as a high-performing comparison rather than a single-model baseline.

## Experiment 3 — learned embeddings, N = 10 games

| Time control | Approach | MAE | R² | Runs |
| --- | --- | ---: | ---: | ---: |
| Blitz | V8-B + mean embedding + LightGBM | 159.7 ± 4.7 | 0.583 | 3 |
| Bullet | V8-B + mean embedding + LightGBM | 174.3 ± 7.6 | 0.551 | 3 |
| Rapid | V8-B + attention embedding + LightGBM | **137.4 ± 11.0** | 0.578 | 3 |

## Player-profile finding

The report's central modeling distinction is:

> Elo represents overall playing strength; the semantic profile describes how the player tends to play relative to players of similar estimated level.

This is why the project is framed as a foundation for adaptation rather than as a completed adaptive engine.