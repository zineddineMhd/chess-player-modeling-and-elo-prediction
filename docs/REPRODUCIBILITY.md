# Reproducibility notes

The experiments were originally implemented as Jupyter/Colab notebooks. This repository stores cleaned `# %%` text exports so that the research code remains diffable and readable in GitHub, VS Code and compatible notebook workflows.

## Expected data organization

The later experiments expect Lichess-derived tables such as:

```text
Bullet/
  moves.csv
  games.csv
  players.csv
Blitz/
  moves.csv
  games.csv
  players.csv
Rapid/
  moves.csv
  games.csv
  players.csv
```

The data itself is not redistributed here.

## Suggested reading order

Start with `01_eda.py`, then follow the numbered experiments. The sequence mirrors the research progression from feature engineering and high-data Elo estimation to few-game modeling, learned embeddings and hierarchical alternatives.

Several scripts still contain environment-specific path configuration inherited from the original Colab experiments. Adapt those configuration blocks before execution.
