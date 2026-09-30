# Player Modeling for Adaptive Systems

Machine-learning study of **chess player modeling from Lichess games**, developed as a foundation for an adaptive and personalized chess engine.

The original goal was broader than what the project ultimately implemented: a complete adaptive engine was **not** built. The completed work addresses the prerequisite modeling problem — estimating player strength from a limited number of games and learning an interpretable behavioral profile that separates **how strong a player is** from **how the player tends to play**.

## Research questions

1. Can player Elo be estimated reliably from analyzed games?
2. How quickly does performance degrade when only a few games are available?
3. Do learned game embeddings add information beyond handcrafted behavioral features?
4. Can we construct an interpretable player profile that is complementary to Elo?

## Main contributions

- Lichess-oriented data pipeline structured at **move, game and player** levels.
- Behavioral features covering move quality, time management, stability and game phases.
- Comparison of Ridge, CatBoost, XGBoost, LightGBM and stacking.
- Few-game evaluation by time control: **bullet, blitz and rapid**.
- DeepSets mean/attention embeddings combined with tabular features.
- Player-level semantic profiling corrected relative to predicted Elo.
- Explicit distinction between **skill estimation** and **playing-style representation**.

## Selected results

### High-data setting

On the initial setting with up to 100 games per player, the strongest stacking configuration achieved:

- **MAE: 100.1 Elo**
- **R²: 0.840**

Among individual models, CatBoost reached a MAE of 109.2 Elo.

### Few-game setting — 10 games per player

The learned embeddings were most useful for rapid games:

| Time control | Best retained approach | MAE | R² |
| --- | --- | ---: | ---: |
| Blitz | V8-B + mean embedding + LightGBM | 159.7 ± 4.7 | 0.583 |
| Bullet | V8-B + mean embedding + LightGBM | 174.3 ± 7.6 | 0.551 |
| Rapid | V8-B + attention embedding + LightGBM | **137.4 ± 11.0** | 0.578 |

The project report concludes that embeddings provide complementary signal rather than replacing the handcrafted behavioral variables.

## Why this is an adaptive-systems project

The final project stops **before** the decision/control layer of an adaptive chess engine. Its output is instead the player representation that such a system would need:

```text
Lichess games
    ↓
move/game behavioral features
    ↓
Elo estimation + learned embeddings
    ↓
semantic behavioral profile
    ↓
[future work] adaptive training / engine behavior
```

This distinction is intentional: the repository does not claim that the adaptive engine itself was implemented.

## Repository structure

```text
.
├── experiments/
│   ├── 01_eda.py
│   ├── 02_hierarchical_feature_engineering.py
│   ├── 03_stacking_high_data.py
│   ├── 04_few_game_by_time_control.py
│   ├── 05_feature_selection_ablation.py
│   ├── 06_deepsets_player_embeddings_a.py
│   ├── 07_deepsets_player_embeddings_b.py
│   ├── 08_game_level_late_aggregation.py
│   └── 09_neural_hierarchical_extensions.py
├── docs/
│   ├── RESULTS.md
│   └── REPRODUCIBILITY.md
├── requirements.txt
└── README.md
```

## Limitations

- The project builds the **player model**, not the complete adaptive chess engine.
- Elo prediction becomes less reliable as the number of available games decreases.
- Behavioral profiles are continuous, interpretable descriptors; they should not be treated as perfectly separated natural player classes.
- The profiling dimensions depend on the engineered features and engine-analysis signals available in the dataset.
- Some neural variants are exploratory and are compared against strong tabular baselines rather than presented as universally superior.

## Academic context

**Sorbonne Université — Master 1 MIND, 2025–2026**

Authors:
- **Wafaa Berrais**
- **Zineddine Mohammedi**

Supervision documented in the academic report: Olivier Schwander and Nicolas Baskiotis (in memoriam).

This repository presents the joint academic work without inventing an undocumented individual contribution split.
