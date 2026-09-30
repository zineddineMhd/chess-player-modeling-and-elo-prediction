# PMind experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # V10 corrigée — comparaison V9 best vs V10 game-level
#
# Cette version corrige le protocole expérimental :
#
# - comparaison sur `N = [3, 5, 10, 30]` pour chaque cadence ;
# - split **par joueur stratifié par Elo** commun à V9 et V10 ;
# - exclusion de `game_month` et des colonnes temporelles non comportementales ;
# - V10 entraîné avec exactement `N` parties par joueur train pour chaque expérience ;
# - pondération game-level `1 / nb_parties_du_joueur` ;
# - V9 embeddings réentraînés par seed par défaut, pour mesurer aussi une partie de la variance neuronale ;
# - analyse par buckets Elo pour détecter la régression vers la moyenne ;
# - V10 reste une expérience d’agrégation tardive/interprétabilité, comparée à la vraie baseline V9 best.

# %% [markdown]
# # V10 — Game-level hierarchical model vs meilleure baseline V9
#
# Ce notebook corrige la comparaison : **V10 n’est plus comparée à V8 seulement**, mais à la meilleure approche V9 retenue après les expériences précédentes.
#
# Choix V9 utilisés directement, sans refaire toute la grille :
#
# - **Bullet** : `N=5`, `V8 + embedding mean + LightGBM`
# - **Blitz** : `N=10`, `V8 + embedding mean + LightGBM`
# - **Rapid** : `N=10`, `V8 + embedding attention + LightGBM`
#
# Objectif V10 : tester une approche hiérarchique au niveau partie :
#
# ```text
# Chaque partie → Elo local estimé → agrégation des Elo locaux → Elo joueur
# ```
#
# On compare donc :
#
# ```text
# V9 best: features joueur V8 + embedding DeepSets + LightGBM
# vs
# V10: prédictions game-level + agrégation mean/median/trimmed/weighted
# ```
#
# Puis on analyse aussi l’incertitude via la dispersion des prédictions game-level.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %%
# NOTEBOOK: !pip install catboost

# %%
# Optionnel : montage Google Drive si vous êtes sur Colab.
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    print('Pas dans Google Colab : montage Drive ignoré.')

# %%
# ============================================================
# 0. Imports & configuration V10 corrigée
# ============================================================

import os, sys, random, warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM non installé. Installez-le avec: pip install lightgbm")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch non installé. Installez-le avec: pip install torch")

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 140)
pd.set_option('display.float_format', '{:.3f}'.format)

RANDOM_STATE = 42
DEVICE = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
print('Device:', DEVICE)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(RANDOM_STATE)

# ============================================================
# Chemins : adaptez BASE_DIR si nécessaire
# ============================================================
IN_COLAB = 'google.colab' in sys.modules
BASE_DIR = Path('/content/drive/MyDrive/Data_Pmind_700-30') if IN_COLAB else Path(r'G:\Mon Drive\Data_Pmind_700-30')

CADENCES = ['bullet', 'blitz', 'rapid']
CADENCE_DATA_DIRS = {c: BASE_DIR / c for c in CADENCES}

# Vraie comparaison faible-N + référence haute.
N_VALUES = [3, 5, 10, 30]
MAX_N = 30

# Seeds : 5 par défaut pour une conclusion plus stable.
V10_VS_V9_SEEDS = [0, 1, 2, 3, 4]

# Si True : réentraîne le DeepSets/attention V9 pour chaque seed.
# Plus rigoureux, mais plus long. Si vous voulez aller vite, mettez False.
RETRAIN_V9_NEURAL_PER_SEED = True

# Entraînement neural V9 : réglages modestes.
BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 12
LR = 1e-3
WEIGHT_DECAY = 1e-4
EMB_DIM = 48
HIDDEN_DIM = 96
DROPOUT = 0.20

FAST_DEBUG = False
DEBUG_MAX_PLAYERS_PER_CADENCE = 180

# Seuils V8 repris de vos notebooks.
EVAL_CAP = 900
LOSS_CAP = 600
THRESHOLD_WIN = 150
THRESHOLD_LOSE = -150
LOW_CLOCK_RATIO = 0.20

print('N_VALUES =', N_VALUES)
print('Seeds =', V10_VS_V9_SEEDS)
print('RETRAIN_V9_NEURAL_PER_SEED =', RETRAIN_V9_NEURAL_PER_SEED)

# %% [markdown]
# ## 1. Fonctions V8 reprises du notebook V9/V8

# %%
def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Lit un CSV s'il existe, sinon retourne None."""
    if path.exists():
        return pd.read_csv(path)
    return None


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _build_games_from_available(
    cadence: str,
    moves_cad: pd.DataFrame,
    summary_cad: Optional[pd.DataFrame],
    players_cad: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fallback pour une cadence sans fichier games.csv.
    On reconstruit uniquement la table game-level minimale attendue par le notebook,
    à partir de summary.csv puis moves.csv si les colonnes existent.
    """
    required_cols = [
        'game_id', 'target_name', 'speed', 'time_control', 'target_elo',
        'target_result', 'target_score', 'opening_eco', 'opening_ply',
        'ply_count', 'target_color', 'date_utc'
    ]

    sources = [df for df in [summary_cad, moves_cad] if df is not None and 'game_id' in df.columns]
    if not sources:
        raise FileNotFoundError(
            f"{cadence}: games.csv absent et impossible de reconstruire le game-level "
            "car aucun game_id n'a été trouvé dans summary.csv ou moves.csv."
        )

    base = sources[0].copy()
    if base['game_id'].duplicated().any():
        # Une ligne par partie ; first() suffit pour les métadonnées constantes.
        base = base.groupby('game_id', as_index=False).first()

    games_cad = pd.DataFrame({'game_id': base['game_id']})

    column_map = {
        'target_name':   ['target_name', 'username', 'player', 'target_username'],
        'time_control':  ['time_control', 'tc', 'clock'],
        'target_elo':    ['target_elo', 'elo', 'player_elo'],
        'target_result': ['target_result', 'result'],
        'target_score':  ['target_score', 'score'],
        'opening_eco':   ['opening_eco', 'eco'],
        'opening_ply':   ['opening_ply'],
        'ply_count':     ['ply_count', 'plies', 'num_plies'],
        'target_color':  ['target_color', 'player_color', 'color'],
        'date_utc':      ['date_utc', 'created_at', 'date'],
    }

    for out_col, candidates in column_map.items():
        found = _first_existing_column(base, candidates)
        games_cad[out_col] = base[found] if found is not None else np.nan

    # Si ply_count n'existe pas côté summary, le reconstruire depuis moves.
    if games_cad['ply_count'].isna().all() and 'ply' in moves_cad.columns:
        ply_count = moves_cad.groupby('game_id')['ply'].max().add(1).rename('ply_count_rebuilt')
        games_cad = games_cad.merge(ply_count, on='game_id', how='left')
        games_cad['ply_count'] = games_cad['ply_count_rebuilt']
        games_cad = games_cad.drop(columns=['ply_count_rebuilt'])

    games_cad['speed'] = cadence

    # Valeurs par défaut non analytiques, uniquement pour garder le pipeline inchangé.
    default_tc = {'bullet': '60+0', 'blitz': '180+0', 'rapid': '600+0'}
    games_cad['time_control'] = games_cad['time_control'].fillna(default_tc.get(cadence, ''))
    games_cad['opening_eco'] = games_cad['opening_eco'].fillna('?')
    games_cad['opening_ply'] = pd.to_numeric(games_cad['opening_ply'], errors='coerce')
    games_cad['ply_count'] = pd.to_numeric(games_cad['ply_count'], errors='coerce')
    games_cad['target_score'] = pd.to_numeric(games_cad['target_score'], errors='coerce')
    games_cad['target_elo'] = pd.to_numeric(games_cad['target_elo'], errors='coerce')

    missing_core = [c for c in ['target_name'] if games_cad[c].isna().all()]
    if missing_core:
        raise ValueError(
            f"{cadence}: games.csv absent et colonne(s) impossible(s) à reconstruire : {missing_core}. "
            "Ajoute ces colonnes dans summary.csv ou fournis games.csv."
        )

    return games_cad[required_cols]


def _force_cadence(df: pd.DataFrame, cadence: str) -> pd.DataFrame:
    df = df.copy()
    df['speed'] = cadence
    return df


def load_cadence_data(cadence_dirs: dict[str, Path]):
    """Charge bullet/blitz/rapid depuis des dossiers séparés et concatène les tables."""
    all_moves, all_games, all_summary, all_players = [], [], [], []

    for cadence in CADENCES:
        data_dir = cadence_dirs[cadence]
        moves_path   = data_dir / 'moves.csv'
        games_path   = data_dir / 'games.csv'
        summary_path = data_dir / 'summary.csv'
        players_path = data_dir / 'players.csv'

        moves_cad = _read_csv_if_exists(moves_path)
        summary_cad = _read_csv_if_exists(summary_path)
        players_cad = _read_csv_if_exists(players_path)

        if moves_cad is None:
            raise FileNotFoundError(f"{cadence}: fichier introuvable -> {moves_path}")
        if players_cad is None:
            raise FileNotFoundError(f"{cadence}: fichier introuvable -> {players_path}")
        if summary_cad is None:
            summary_cad = pd.DataFrame()

        games_cad = _read_csv_if_exists(games_path)
        if games_cad is None:
            print(f"⚠ {cadence}: games.csv absent — reconstruction minimale depuis summary/moves")
            games_cad = _build_games_from_available(cadence, moves_cad, summary_cad, players_cad)

        moves_cad   = _force_cadence(moves_cad, cadence)
        games_cad   = _force_cadence(games_cad, cadence)
        summary_cad = _force_cadence(summary_cad, cadence) if len(summary_cad) else summary_cad
        players_cad = _force_cadence(players_cad, cadence)

        all_moves.append(moves_cad)
        all_games.append(games_cad)
        all_summary.append(summary_cad)
        all_players.append(players_cad)

        print(f"\n[{cadence.upper()}] {data_dir}")
        print(f"  moves   : {moves_cad.shape[0]:>8,} × {moves_cad.shape[1]}")
        print(f"  games   : {games_cad.shape[0]:>8,} × {games_cad.shape[1]}")
        print(f"  players : {players_cad.shape[0]:>8,} × {players_cad.shape[1]}")

    moves   = pd.concat(all_moves, ignore_index=True)
    games   = pd.concat(all_games, ignore_index=True)
    summary = pd.concat([df for df in all_summary if len(df)], ignore_index=True) if any(len(df) for df in all_summary) else pd.DataFrame()
    players = pd.concat(all_players, ignore_index=True)

    print(f"\n✓ Chargement concaténé OK")
    print(f"moves   : {moves.shape[0]:>8,} × {moves.shape[1]}")
    print(f"games   : {games.shape[0]:>8,} × {games.shape[1]}")
    print(f"players : {players.shape[0]:>8,} × {players.shape[1]}")
    return moves, games, summary, players

# %%
def clean_moves(moves: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage moves — taxonomie recalculée sur cp_loss_clean.
    CORRECTION : is_good / is_blunder / etc. utilisent cp_loss_clean (pas loss_capped).
    """
    df = moves.copy()

    df['eval_before_capped'] = df['eval_white_before'].clip(-EVAL_CAP, EVAL_CAP)
    df['eval_after_capped']  = df['eval_white_after'].clip(-EVAL_CAP, EVAL_CAP)

    # Perte reconstruite (diagnostic uniquement)
    df['loss_signed'] = np.where(
        df['player_color'] == 'white',
        df['eval_before_capped'] - df['eval_after_capped'],
        df['eval_after_capped']  - df['eval_before_capped']
    )
    df['loss_capped']   = df['loss_signed'].clip(0, LOSS_CAP)
    df['cp_loss_clean'] = df['cp_loss'].abs().clip(0, LOSS_CAP)
    df['eval_abs']      = df['eval_before_capped'].abs()

    eval_for_player = np.where(
        df['player_color'] == 'white',
        df['eval_before_capped'], -df['eval_before_capped']
    )
    df['is_winning'] = (eval_for_player > THRESHOLD_WIN).astype(int)
    df['is_equal']   = ((eval_for_player >= THRESHOLD_LOSE) &
                        (eval_for_player <= THRESHOLD_WIN)).astype(int)
    df['is_losing']  = (eval_for_player < THRESHOLD_LOSE).astype(int)

    # Taxonomie sur cp_loss_clean
    df['is_good']       = (df['cp_loss_clean'] < 50).astype(int)
    df['is_inaccuracy'] = ((df['cp_loss_clean'] >= 50)  & (df['cp_loss_clean'] < 100)).astype(int)
    df['is_mistake']    = ((df['cp_loss_clean'] >= 100) & (df['cp_loss_clean'] < 200)).astype(int)
    df['is_blunder']    = (df['cp_loss_clean'] >= 200).astype(int)
    df['is_error']      = (df['cp_loss_clean'] >= 50).astype(int)

    mask_not_target = ~df['is_target_move']
    for col in ['is_good', 'is_inaccuracy', 'is_mistake', 'is_blunder',
                'is_error', 'cp_loss_clean', 'loss_capped']:
        df.loc[mask_not_target, col] = np.nan

    df['is_opening']    = (df['phase'] == 'opening').astype(int)
    df['is_middlegame'] = (df['phase'] == 'middlegame').astype(int)
    df['is_endgame']    = (df['phase'] == 'endgame').astype(int)

    df['time_spent_s'] = df['time_spent_cs'].fillna(0) / 100
    df['clock_s']      = df['clock_before_cs'].fillna(df['clock_before_cs'].median()) / 100
    df['time_log']     = np.log1p(df['time_spent_s'])

    df['is_capture']   = df['is_capture'].astype(int)
    df['is_check']     = df['is_check'].astype(int)
    df['is_castle']    = df['is_castle'].astype(int)
    df['is_promotion'] = df['is_promotion'].astype(int)
    df['move_number']  = (df['ply'] + 1) // 2

    print(f"✓ Moves nettoyés : {len(df):,} lignes")
    print(f"  Blunder rate (cp_loss_clean) : "
          f"{df[df['is_target_move']]['is_blunder'].mean()*100:.1f}%")
    return df

# %%
def laplace_smooth(k, n, alpha=1.0):
    """Lissage Laplace : (k + alpha) / (n + 2*alpha). Robuste aux petits n."""
    return (k + alpha) / (n + 2 * alpha)


def build_game_features_v8(moves_df: pd.DataFrame) -> pd.DataFrame:
    m = moves_df[moves_df['is_target_move']].copy()
    m['loss_main']     = m['cp_loss_clean']
    m['loss_log_main'] = np.log1p(m['loss_main'])

    max_clock_per_game = m.groupby('game_id')['clock_s'].transform('max')
    m['is_low_clock']  = (m['clock_s'] < LOW_CLOCK_RATIO * max_clock_per_game).astype(int)
    m['is_critical']   = ((m['is_error'] == 1) | (m['eval_abs'] > 300)).astype(int)

    agg = {}

    # ── Qualité de base ───────────────────────────────────────────────────────
    agg['n_moves']         = ('loss_main', 'count')
    agg['loss_mean']       = ('loss_main', 'mean')
    agg['loss_median']     = ('loss_main', 'median')
    agg['loss_std']        = ('loss_main', 'std')
    agg['loss_p25']        = ('loss_main', lambda x: x.quantile(0.25))
    agg['loss_p75']        = ('loss_main', lambda x: x.quantile(0.75))
    agg['loss_p90']        = ('loss_main', lambda x: x.quantile(0.90))
    agg['loss_iqr']        = ('loss_main', lambda x: x.quantile(0.75) - x.quantile(0.25))
    agg['loss_max']        = ('loss_main', 'max')
    agg['loss_log_mean']   = ('loss_log_main', 'mean')
    agg['rate_good']       = ('is_good',       'mean')
    agg['rate_inaccuracy'] = ('is_inaccuracy',  'mean')
    agg['rate_mistake']    = ('is_mistake',     'mean')
    agg['rate_blunder']    = ('is_blunder',     'mean')
    agg['rate_error']      = ('is_error',       'mean')
    agg['n_blunders']      = ('is_blunder',     'sum')
    agg['n_mistakes']      = ('is_mistake',     'sum')
    agg['n_errors']        = ('is_error',       'sum')

    # Taux lissés Laplace
    agg['rate_blunder_smooth'] = ('is_blunder', lambda x: laplace_smooth(x.sum(), len(x)))
    agg['rate_mistake_smooth'] = ('is_mistake', lambda x: laplace_smooth(x.sum(), len(x)))
    agg['rate_error_smooth']   = ('is_error',   lambda x: laplace_smooth(x.sum(), len(x)))
    agg['rate_loss_gt100']     = ('loss_main', lambda x: (x > 100).mean())
    agg['rate_loss_gt200']     = ('loss_main', lambda x: (x > 200).mean())

    # ── Temps ─────────────────────────────────────────────────────────────────
    agg['time_mean_s']   = ('time_spent_s', 'mean')
    agg['time_median_s'] = ('time_spent_s', 'median')
    agg['time_std_s']    = ('time_spent_s', 'std')
    agg['time_iqr_s']    = ('time_spent_s', lambda x: x.quantile(0.75) - x.quantile(0.25))
    agg['time_p90_s']    = ('time_spent_s', lambda x: x.quantile(0.90))
    agg['time_log_mean'] = ('time_log', 'mean')
    agg['clock_mean_s']  = ('clock_s', 'mean')

    agg['time_mean_on_error']   = ('time_spent_s', lambda x: x[m.loc[x.index, 'is_error'] == 1].mean())
    agg['time_mean_on_blunder'] = ('time_spent_s', lambda x: x[m.loc[x.index, 'is_blunder'] == 1].mean())
    agg['time_mean_on_good']    = ('time_spent_s', lambda x: x[m.loc[x.index, 'is_good'] == 1].mean())

    agg['time_pressure_by_phase_opening']    = ('is_low_clock', lambda x: x[m.loc[x.index, 'is_opening'] == 1].mean())
    agg['time_pressure_by_phase_middlegame'] = ('is_low_clock', lambda x: x[m.loc[x.index, 'is_middlegame'] == 1].mean())
    agg['time_pressure_by_phase_endgame']    = ('is_low_clock', lambda x: x[m.loc[x.index, 'is_endgame'] == 1].mean())

    agg['good_fast_move_rate']        = ('is_good', lambda x: (x[m.loc[x.index, 'time_spent_s'] <= 2] == 1).sum() / (len(x) + 1e-9))
    agg['good_fast_move_rate_smooth'] = ('is_good', lambda x: laplace_smooth((x[m.loc[x.index, 'time_spent_s'] <= 2] == 1).sum(), len(x)))

    agg['blunder_low_time_rate']        = ('is_blunder', lambda x: (x[m.loc[x.index, 'is_low_clock'] == 1] == 1).sum() / (len(x) + 1e-9))
    agg['blunder_low_time_rate_smooth'] = ('is_blunder', lambda x: laplace_smooth((x[m.loc[x.index, 'is_low_clock'] == 1] == 1).sum(), len(x)))

    agg['time_used_when_equal']       = ('time_spent_s', lambda x: x[m.loc[x.index, 'is_equal'] == 1].mean())
    agg['avg_time_on_critical_moves'] = ('time_spent_s', lambda x: x[m.loc[x.index, 'is_critical'] == 1].mean())

    # ── Contexte positionnel ──────────────────────────────────────────────────
    agg['loss_when_winning'] = ('loss_main', lambda x: x[m.loc[x.index, 'is_winning'] == 1].mean())
    agg['loss_when_equal']   = ('loss_main', lambda x: x[m.loc[x.index, 'is_equal'] == 1].mean())
    agg['loss_when_losing']  = ('loss_main', lambda x: x[m.loc[x.index, 'is_losing'] == 1].mean())

    agg['rate_blunder_when_equal_smooth']   = ('is_blunder', lambda x: laplace_smooth(x[m.loc[x.index, 'is_equal'] == 1].sum(),   (m.loc[x.index, 'is_equal'] == 1).sum()))
    agg['rate_blunder_when_winning_smooth'] = ('is_blunder', lambda x: laplace_smooth(x[m.loc[x.index, 'is_winning'] == 1].sum(), (m.loc[x.index, 'is_winning'] == 1).sum()))
    agg['rate_blunder_when_losing_smooth']  = ('is_blunder', lambda x: laplace_smooth(x[m.loc[x.index, 'is_losing'] == 1].sum(),  (m.loc[x.index, 'is_losing'] == 1).sum()))

    # Counts (dénominateurs)
    agg['n_moves_winning']   = ('is_winning',   lambda x: (m.loc[x.index, 'is_winning'] == 1).sum())
    agg['n_moves_equal']     = ('is_equal',     lambda x: (m.loc[x.index, 'is_equal'] == 1).sum())
    agg['n_moves_losing']    = ('is_losing',    lambda x: (m.loc[x.index, 'is_losing'] == 1).sum())
    agg['n_moves_opening']   = ('is_opening',   lambda x: (m.loc[x.index, 'is_opening'] == 1).sum())
    agg['n_moves_low_clock'] = ('is_low_clock', lambda x: (m.loc[x.index, 'is_low_clock'] == 1).sum())

    # Taux blunder par phase — lissés
    agg['rate_blunder_opening_smooth']    = ('is_blunder', lambda x: laplace_smooth(x[m.loc[x.index, 'is_opening'] == 1].sum(),    (m.loc[x.index, 'is_opening'] == 1).sum()))
    agg['rate_blunder_middlegame_smooth'] = ('is_blunder', lambda x: laplace_smooth(x[m.loc[x.index, 'is_middlegame'] == 1].sum(), (m.loc[x.index, 'is_middlegame'] == 1).sum()))
    agg['rate_mistake_opening_smooth']    = ('is_mistake', lambda x: laplace_smooth(x[m.loc[x.index, 'is_opening'] == 1].sum(),    (m.loc[x.index, 'is_opening'] == 1).sum()))
    agg['rate_mistake_middlegame_smooth'] = ('is_mistake', lambda x: laplace_smooth(x[m.loc[x.index, 'is_middlegame'] == 1].sum(), (m.loc[x.index, 'is_middlegame'] == 1).sum()))

    # ── Style ─────────────────────────────────────────────────────────────────
    agg['rate_capture']   = ('is_capture',   'mean')
    agg['rate_check']     = ('is_check',     'mean')
    agg['rate_castle']    = ('is_castle',    'sum')
    agg['rate_promotion'] = ('is_promotion', 'mean')

    game_features = m.groupby('game_id').agg(**agg).reset_index()

    # ── Qualité par phase (mean + median + std + IQR + p90) ──────────────────
    for phase in ['opening', 'middlegame', 'endgame']:
        sub = (
            m[m['phase'] == phase]
            .groupby('game_id')
            .agg(
                **{
                    f'loss_{phase}_mean':   pd.NamedAgg('loss_main', 'mean'),
                    f'loss_{phase}_median': pd.NamedAgg('loss_main', 'median'),
                    f'loss_{phase}_std':    pd.NamedAgg('loss_main', 'std'),
                    f'loss_{phase}_iqr':    pd.NamedAgg('loss_main', lambda x: x.quantile(0.75) - x.quantile(0.25)),
                    f'loss_{phase}_p90':    pd.NamedAgg('loss_main', lambda x: x.quantile(0.90)),
                }
            )
            .reset_index()
        )
        game_features = game_features.merge(sub, on='game_id', how='left')

        sub_t = (
            m[m['phase'] == phase]
            .groupby('game_id')
            .agg(
                **{
                    f'time_{phase}_mean_s':   pd.NamedAgg('time_spent_s', 'mean'),
                    f'time_{phase}_median_s': pd.NamedAgg('time_spent_s', 'median'),
                }
            )
            .reset_index()
        )
        game_features = game_features.merge(sub_t, on='game_id', how='left')

    # ── Drift intra-partie ────────────────────────────────────────────────────
    def half_loss_diff(g):
        mid    = len(g) // 2
        first  = g.iloc[:mid]['loss_main'].mean() if mid > 0      else np.nan
        second = g.iloc[mid:]['loss_main'].mean() if mid < len(g) else np.nan
        return second - first

    drift = m.groupby('game_id').apply(half_loss_diff).rename('loss_drift_end_vs_start')
    game_features = game_features.merge(drift, on='game_id', how='left')

    # ── Temps avant blunder ───────────────────────────────────────────────────
    def time_before_blunder(g):
        times = []
        for idx in g[g['is_blunder'] == 1].index:
            pos = g.index.get_loc(idx)
            if pos > 0:
                times.append(g['time_spent_s'].iloc[pos - 1])
        return np.mean(times) if times else np.nan

    tbb = m.groupby('game_id').apply(time_before_blunder).rename('time_used_before_blunder')
    game_features = game_features.merge(tbb, on='game_id', how='left')

    # ── Interactions Temps × Qualité ──────────────────────────────────────────
    game_features['loss_per_second']        = game_features['loss_mean'] / (game_features['time_mean_s'] + 1e-3)
    game_features['blunder_pressure_ratio'] = game_features['blunder_low_time_rate_smooth'] / (game_features['rate_blunder_smooth'] + 1e-3)
    game_features['time_on_error_vs_mean']  = game_features['time_mean_on_error'] / (game_features['time_mean_s'] + 1e-3)
    game_features['fast_good_vs_error']     = game_features['good_fast_move_rate_smooth'] / (game_features['rate_error_smooth'] + 1e-3)

    print(f"✓ game_features V8 : {game_features.shape[0]:,} parties × {game_features.shape[1]} features")
    return game_features

# %%
def enrich_game_features(game_features: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    meta = games[[
        'game_id', 'target_name', 'speed', 'time_control',
        'target_elo', 'target_result', 'target_score',
        'opening_eco', 'opening_ply', 'ply_count', 'target_color', 'date_utc'
    ]].copy()
    tc_split = meta['time_control'].astype(str).str.split('+', expand=True)
    meta['tc_base']      = pd.to_numeric(tc_split[0], errors='coerce')
    meta['tc_increment'] = pd.to_numeric(tc_split[1], errors='coerce')
    meta['is_white']     = (meta['target_color'] == 'white').astype(int)
    meta['date_utc']     = pd.to_datetime(meta['date_utc'], errors='coerce', utc=True)
    meta['game_month']   = meta['date_utc'].dt.month
    meta['eco_family']   = meta['opening_eco'].astype(str).str[0].replace('n', '?').fillna('?')
    meta['speed_enc']    = meta['speed'].map({'ultraBullet': 0, 'bullet': 1, 'blitz': 2, 'rapid': 3, 'classical': 4})
    meta = meta.rename(columns={'target_elo': 'elo'})
    result = game_features.merge(meta.drop(columns=['time_control', 'target_color']), on='game_id', how='left')
    print(f"✓ game_rich : {result.shape} — {result['target_name'].nunique()} joueurs")
    return result

# %%
def build_player_features_v8(game_rich: pd.DataFrame) -> pd.DataFrame:
    g = game_rich.copy()

    # Colonnes à agréger avec mean + median + IQR
    rich_agg_cols = [
        'loss_mean', 'loss_median', 'loss_p75', 'loss_p90', 'loss_iqr',
        'loss_opening_mean', 'loss_opening_median', 'loss_opening_iqr',
        'loss_middlegame_mean', 'loss_middlegame_median',
        'loss_endgame_mean',
        'rate_blunder_smooth', 'rate_mistake_smooth', 'rate_error_smooth',
        'rate_blunder_opening_smooth', 'rate_blunder_middlegame_smooth',
        'rate_mistake_opening_smooth', 'rate_mistake_middlegame_smooth',
        'rate_blunder_when_equal_smooth', 'rate_blunder_when_winning_smooth',
        'rate_blunder_when_losing_smooth',
        'time_mean_s', 'time_median_s', 'time_iqr_s', 'time_p90_s',
        'time_opening_mean_s', 'time_opening_median_s',
        'time_middlegame_mean_s', 'time_middlegame_median_s',
        'time_endgame_mean_s',
        'good_fast_move_rate_smooth', 'blunder_low_time_rate_smooth',
        'time_mean_on_blunder', 'time_mean_on_good', 'time_used_when_equal',
        'time_pressure_by_phase_opening', 'time_pressure_by_phase_middlegame',
        'time_pressure_by_phase_endgame',
        'loss_per_second', 'blunder_pressure_ratio',
        'time_on_error_vs_mean', 'fast_good_vs_error',
        'ply_count', 'loss_drift_end_vs_start',
        'n_moves_winning', 'n_moves_equal', 'n_moves_losing',
        'n_moves_opening', 'n_moves_low_clock',
    ]
    rich_agg_cols = [c for c in rich_agg_cols if c in g.columns]

    # Colonnes à agréger avec mean + std seulement
    std_agg_cols = [
        'loss_std', 'loss_max', 'loss_log_mean',
        'rate_good', 'rate_inaccuracy', 'rate_mistake', 'rate_blunder', 'rate_error',
        'n_blunders', 'n_mistakes', 'n_errors', 'rate_loss_gt100', 'rate_loss_gt200',
        'loss_when_winning', 'loss_when_equal', 'loss_when_losing',
        'time_std_s', 'time_log_mean', 'clock_mean_s',
        'time_mean_on_error', 'avg_time_on_critical_moves',
        'rate_capture', 'rate_check', 'rate_castle', 'rate_promotion', 'n_moves',
    ]
    std_agg_cols = [c for c in std_agg_cols if c in g.columns]

    agg_dict = {}
    for col in rich_agg_cols:
        agg_dict[f'{col}_mean']   = (col, 'mean')
        agg_dict[f'{col}_median'] = (col, 'median')
        agg_dict[f'{col}_iqr']    = (col, lambda x: x.quantile(0.75) - x.quantile(0.25))
    for col in std_agg_cols:
        agg_dict[f'{col}_mean'] = (col, 'mean')
        agg_dict[f'{col}_std']  = (col, 'std')

    agg_dict['n_games'] = ('game_id', 'count')
    if 'target_score' in g.columns:
        agg_dict['win_rate'] = ('target_score', 'mean')
    # Important : la cible Elo doit exister dans les datasets exact-N pour LightGBM.
    # Elle est agrégée ici mais sera exclue automatiquement des features.
    if 'elo' in g.columns:
        agg_dict['elo'] = ('elo', 'median')

    player_feat = g.groupby('target_name').agg(**agg_dict).reset_index()

    eco_div = g.groupby('target_name')['eco_family'].nunique().rename('eco_diversity')
    player_feat = player_feat.merge(eco_div, on='target_name', how='left')

    speed_mode = g.groupby('target_name')['speed'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
    ).rename('dominant_speed')
    player_feat = player_feat.merge(speed_mode, on='target_name', how='left')
    player_feat['dominant_speed_enc'] = player_feat['dominant_speed'].map(
        {'ultraBullet': 0, 'bullet': 1, 'blitz': 2, 'rapid': 3, 'classical': 4}
    )

    # Stabilité inter-parties enrichie
    stability = g.groupby('target_name').agg(
        between_game_loss_variance=('loss_mean', 'var'),
        between_game_error_variance=('rate_error_smooth', 'var'),
        loss_mean_cv=('loss_mean', lambda x: x.std() / (x.mean() + 1e-9)),
        rate_error_cv=('rate_error_smooth', lambda x: x.std() / (x.mean() + 1e-9)),
        prop_games_with_blunder=('n_blunders', lambda x: (x > 0).mean()),
        prop_games_with_mistake=('n_mistakes', lambda x: (x > 0).mean()),
        loss_mean_p25=('loss_mean', lambda x: x.quantile(0.25)),
        loss_mean_p75=('loss_mean', lambda x: x.quantile(0.75)),
        loss_mean_iqr=('loss_mean', lambda x: x.quantile(0.75) - x.quantile(0.25)),
    ).reset_index()
    player_feat = player_feat.merge(stability, on='target_name', how='left')

    if {'loss_endgame_mean', 'loss_opening_mean'}.issubset(g.columns):
        gap = g.groupby('target_name').apply(
            lambda x: (x['loss_endgame_mean'] - x['loss_opening_mean']).mean()
        ).rename('opening_vs_endgame_loss_gap').reset_index()
        player_feat = player_feat.merge(gap, on='target_name', how='left')

    print(f"✓ player_features V8 : {player_feat.shape[0]} joueurs × {player_feat.shape[1]} features")
    return player_feat


def prepare_target(player_feat: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    pf = player_feat.copy()
    cols_to_drop = [c for c in ['elo', 'elo_bucket'] if c in pf.columns]
    if cols_to_drop:
        pf = pf.drop(columns=cols_to_drop)
    target = players[['username', 'elo', 'elo_bucket']].rename(columns={'username': 'target_name'})
    df = pf.merge(target, on='target_name', how='inner')
    print(f"✓ Dataset final : {df.shape[0]} joueurs × {df.shape[1]} colonnes")
    return df

# %%

# ============================================================
# 2. Chargement / construction game_rich V8 avec cache sûr
# ============================================================

# Cette cellule charge les CSV cadence par cadence, construit les features game-level V8,
# puis crée `game_rich`, qui est la table centrale du notebook V10.
# Le cache est écrit dans un dossier temporaire/local pour éviter les problèmes Drive en lecture seule.

def _choose_writable_cache_dir():
    candidates = []
    if IN_COLAB:
        candidates.append(Path('/content/_v10_cache'))
    candidates.append(Path.cwd() / '_v10_cache')
    candidates.append(Path('/tmp/_v10_cache'))
    for p in candidates:
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / '.write_test'
            test.write_text('ok')
            test.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    raise RuntimeError('Aucun dossier cache écrivable trouvé.')

CACHE_DIR = _choose_writable_cache_dir()
GAME_RICH_CACHE = CACHE_DIR / 'game_rich_v8_all_cadences.csv'
REBUILD_GAME_RICH = False  # mettez True pour forcer la reconstruction

print('BASE_DIR =', BASE_DIR)
print('CACHE_DIR =', CACHE_DIR)

if GAME_RICH_CACHE.exists() and not REBUILD_GAME_RICH:
    print(f'Chargement du cache existant : {GAME_RICH_CACHE}')
    game_rich = pd.read_csv(GAME_RICH_CACHE)
else:
    print('Chargement des CSV cadence par cadence...')
    moves_raw, games_raw, summary_raw, players_raw = load_cadence_data(CADENCE_DATA_DIRS)

    print('Nettoyage des coups avec cp_loss_clean...')
    moves_v8 = clean_moves(moves_raw)
    print('moves_v8 :', moves_v8.shape)

    print('Construction des features game-level V8...')
    game_features = build_game_features_v8(moves_v8)

    print('Ajout des métadonnées de partie...')
    game_rich = enrich_game_features(game_features, games_raw)

    game_rich.to_csv(GAME_RICH_CACHE, index=False)
    print(f'Cache sauvegardé : {GAME_RICH_CACHE}')

print('\ngame_rich:', game_rich.shape)
display(game_rich.head())

# %%
# ============================================================
# 3. Nettoyage anti-leakage + choix des colonnes game-level
# ============================================================

# On retire explicitement les colonnes qui donnent directement/indirectement le Elo.
# On retire aussi game_month/date/timestamp : ce ne sont pas des signaux comportementaux.
LEAKAGE_PATTERNS = [
    'elo', 'rating', 'opponent', 'advers', 'bucket',
]
LEAKAGE_EXACT = {
    'target_elo', 'elo', 'opponent_elo', 'elo_diff', 'rating_diff',
    'white_elo', 'black_elo', 'target_name', 'username', 'player',
    'game_id', 'date_utc', 'opening_eco', 'time_control', 'speed',
}

EXCLUDE_EXTRA = {
    'game_month', 'month', 'created_at', 'timestamp', 'utc_date', 'date',
}


def get_game_feature_cols(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cols = []
    for c in numeric_cols:
        cl = c.lower()
        if c in LEAKAGE_EXACT or c in EXCLUDE_EXTRA:
            continue
        if any(p in cl for p in LEAKAGE_PATTERNS):
            continue
        # Évite les colonnes temporelles numériques qui auraient été générées.
        if any(p in cl for p in ['month', 'timestamp', 'date']):
            continue
        cols.append(c)
    return cols

GAME_FEATURE_COLS = get_game_feature_cols(game_rich)
print(f'{len(GAME_FEATURE_COLS)} colonnes game-level utilisées')
print(GAME_FEATURE_COLS[:40])

# Contrôle rapide : aucune colonne Elo/adversaire/date ne doit passer.
bad_cols = [c for c in GAME_FEATURE_COLS if any(p in c.lower() for p in ['elo','opponent','rating','month','date','timestamp'])]
print('Colonnes suspectes restantes :', bad_cols)

# %%
# ============================================================
# 4. Split par joueur stratifié par Elo
# ============================================================

PLAYER_COL = 'target_name'
TARGET_COL = 'elo'
CADENCE_COL = 'speed'


def prepare_cadence_df(game_rich: pd.DataFrame, cadence: str, min_games: int = MAX_N) -> pd.DataFrame:
    df = game_rich[game_rich[CADENCE_COL] == cadence].copy()
    if TARGET_COL not in df.columns and 'target_elo' in df.columns:
        df = df.rename(columns={'target_elo': TARGET_COL})
    df = df.dropna(subset=[PLAYER_COL, TARGET_COL])

    counts = df.groupby(PLAYER_COL)['game_id'].nunique()
    keep_players = counts[counts >= min_games].index
    df = df[df[PLAYER_COL].isin(keep_players)].copy()

    # Cible joueur stable : médiane du Elo si petites variations.
    player_elo = df.groupby(PLAYER_COL)[TARGET_COL].median().rename('elo_player_median')
    df = df.drop(columns=[TARGET_COL]).merge(player_elo, on=PLAYER_COL, how='left')
    df = df.rename(columns={'elo_player_median': TARGET_COL})
    return df


def split_players_stratified_by_elo(df: pd.DataFrame, seed=42, n_bins=5):
    """Split joueur train/val/test stratifié par buckets Elo."""
    player_elo = df.groupby(PLAYER_COL)[TARGET_COL].median().reset_index()
    # qcut robuste : si duplicates, réduit automatiquement les bins.
    player_elo['elo_bucket'] = pd.qcut(
        player_elo[TARGET_COL].rank(method='first'),
        q=min(n_bins, len(player_elo)),
        labels=False,
        duplicates='drop'
    )

    players = player_elo[PLAYER_COL].values
    buckets = player_elo['elo_bucket'].values

    try:
        train_val, test, y_train_val, y_test = train_test_split(
            players, buckets, test_size=0.20, random_state=seed, stratify=buckets
        )
        train_val_df = pd.DataFrame({PLAYER_COL: train_val, 'bucket': y_train_val})
        train, val, _, _ = train_test_split(
            train_val_df[PLAYER_COL].values,
            train_val_df['bucket'].values,
            test_size=0.20,
            random_state=seed,
            stratify=train_val_df['bucket'].values
        )
    except ValueError as e:
        print('⚠️ Stratification impossible, fallback split aléatoire :', e)
        train_val, test = train_test_split(players, test_size=0.20, random_state=seed)
        train, val = train_test_split(train_val, test_size=0.20, random_state=seed)

    return np.array(train), np.array(val), np.array(test), player_elo


cadence_data = {}
for cadence in CADENCES:
    try:
        df_c = prepare_cadence_df(game_rich, cadence, min_games=MAX_N)
        if FAST_DEBUG:
            rng = np.random.default_rng(RANDOM_STATE)
            players_dbg = rng.choice(df_c[PLAYER_COL].unique(), size=min(DEBUG_MAX_PLAYERS_PER_CADENCE, df_c[PLAYER_COL].nunique()), replace=False)
            df_c = df_c[df_c[PLAYER_COL].isin(players_dbg)].copy()
        tr, va, te, player_elo_table = split_players_stratified_by_elo(df_c, RANDOM_STATE, n_bins=5)
        cadence_data[cadence] = {
            'df': df_c,
            'train_players': tr,
            'val_players': va,
            'test_players': te,
            'player_elo_table': player_elo_table,
        }
        print(f'{cadence:7s}: {df_c[PLAYER_COL].nunique():4d} joueurs, {len(df_c):6d} parties | train={len(tr)}, val={len(va)}, test={len(te)}')
    except Exception as e:
        print(f'⚠️ {cadence}: ignorée -> {e}')

# %%
# ============================================================
# Échantillonnage exact-N : même seed = mêmes parties pour V8 et embedding
# ============================================================

def sample_exact_n_games(df: pd.DataFrame, players: np.ndarray, n_games: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in players:
        g = df[df[PLAYER_COL] == p]
        # sécurité : normalement tous ont >= n_games
        if len(g) < n_games:
            continue
        idx = rng.choice(g.index.to_numpy(), size=n_games, replace=False)
        rows.append(df.loc[idx])
    if not rows:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(rows, ignore_index=True)


def build_player_dataset_exact_n(df_c: pd.DataFrame, players: np.ndarray, n_games: int, seed: int = 42) -> pd.DataFrame:
    sampled = sample_exact_n_games(df_c, players, n_games, seed)
    # réutilise votre agrégation V8 au niveau joueur
    player_df = build_player_features_v8(sampled)
    return player_df

# %%
# ============================================================
# 7. Fonctions LightGBM baseline/hybride
# ============================================================

def get_player_feature_cols_safe(dataset: pd.DataFrame) -> List[str]:
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    banned_exact = {'elo'}
    cols = []
    for c in numeric_cols:
        cl = c.lower()
        if c in banned_exact:
            continue
        if any(p in cl for p in ['elo', 'rating', 'opponent', 'advers', 'bucket']):
            continue
        if any(p in cl for p in ['month', 'timestamp', 'date']):
            continue
        cols.append(c)
    return cols


def fit_eval_lgbm(train_df, test_df, target='elo', random_state=42):
    if not LIGHTGBM_AVAILABLE:
        raise ImportError('LightGBM non disponible')
    if target not in train_df.columns or target not in test_df.columns:
        raise KeyError(f"Cible '{target}' absente.")
    feature_cols = get_player_feature_cols_safe(train_df)
    feature_cols = [c for c in feature_cols if c in test_df.columns]
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(train_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    X_test = imp.transform(test_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_train = train_df[target].astype(float).values
    y_test = test_df[target].astype(float).values

    model = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        'MAE': mean_absolute_error(y_test, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),
        'R2': r2_score(y_test, pred),
        'n_features': len(feature_cols),
        'model': model,
        'imputer': imp,
        'feature_cols': feature_cols,
        'pred': pred,
        'y_test': y_test,
    }


def merge_embeddings(player_df, emb_df):
    emb_cols = [c for c in emb_df.columns if c.startswith('emb_')]
    return player_df.merge(emb_df[[PLAYER_COL] + emb_cols], on=PLAYER_COL, how='left')

# %%
# ============================================================
# 8. Configuration V10 corrigée vs V9 best
# ============================================================

# Le type d'embedding V9 retenu par cadence d'après vos expériences V9.
# On l'utilise pour tous les N = 3/5/10/30 afin de comparer proprement la robustesse en faible-N.
V9_KIND_BY_CADENCE = {
    'bullet': 'mean',
    'blitz': 'mean',
    'rapid': 'attention',
}

V9_APPROACH_BY_CADENCE = {
    'bullet': 'V9 best: V8 + embedding mean + LightGBM',
    'blitz': 'V9 best: V8 + embedding mean + LightGBM',
    'rapid': 'V9 best: V8 + embedding attention + LightGBM',
}

V10_AGGREGATORS = ['mean', 'median', 'trimmed_mean', 'weighted_log_nmoves']
UNCERTAINTY_ALPHAS = [0.20, 0.30, 0.40]

print('V9_KIND_BY_CADENCE =', V9_KIND_BY_CADENCE)
print('N_VALUES =', N_VALUES)
print('Seeds =', V10_VS_V9_SEEDS)
print('V10_AGGREGATORS =', V10_AGGREGATORS)
print('Important : si RETRAIN_V9_NEURAL_PER_SEED=True, le modèle neural V9 est réentraîné pour chaque seed.')

# %%
# ============================================================
# 9. V9 best baseline : DeepSets selected embedding + LightGBM
# ============================================================

if not TORCH_AVAILABLE:
    raise ImportError("PyTorch est nécessaire pour reconstruire la baseline V9 embeddings.")

class PlayerSetDataset(Dataset):
    """Dataset joueur = ensemble de parties, avec padding jusqu'à max_n."""
    def __init__(self, df_c, players, feature_cols, max_n=30, n_choices=None, fixed_n=None, seed=42, scaler=None):
        self.df = df_c.copy()
        self.players = list(players)
        self.feature_cols = feature_cols
        self.max_n = max_n
        self.n_choices = n_choices or [max_n]
        self.fixed_n = fixed_n
        self.seed = seed
        self.scaler = scaler
        self.player_groups = {p: g.reset_index(drop=True) for p, g in self.df.groupby(PLAYER_COL)}

    def __len__(self):
        return len(self.players)

    def __getitem__(self, idx):
        p = self.players[idx]
        g = self.player_groups[p]
        if self.fixed_n is not None:
            rng = np.random.default_rng(self.seed + idx)
            n = int(self.fixed_n)
        else:
            rng = np.random.default_rng(self.seed + idx + random.randint(0, 10_000_000))
            n = int(rng.choice(self.n_choices))
        n = min(n, len(g), self.max_n)
        chosen = rng.choice(np.arange(len(g)), size=n, replace=False)
        x = g.loc[chosen, self.feature_cols].to_numpy(dtype=np.float32)
        if self.scaler is not None:            x = self.scaler.transform(x).astype(np.float32)

        x_pad = np.zeros((self.max_n, len(self.feature_cols)), dtype=np.float32)
        mask = np.zeros((self.max_n,), dtype=np.float32)
        x_pad[:n] = x
        mask[:n] = 1.0
        y = float(g[TARGET_COL].median())
        return torch.tensor(x_pad), torch.tensor(mask), torch.tensor(y, dtype=torch.float32), p


class DeepSetsMean(nn.Module):
    def __init__(self, input_dim, emb_dim=48, hidden_dim=96, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim), nn.ReLU()
        )
        self.regressor = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
    def forward(self, x, mask):
        h = self.encoder(x)
        mask_e = mask.unsqueeze(-1)
        emb = (h * mask_e).sum(dim=1) / mask_e.sum(dim=1).clamp(min=1.0)
        pred = self.regressor(emb).squeeze(-1)
        return pred, emb, None


class DeepSetsAttention(nn.Module):
    def __init__(self, input_dim, emb_dim=48, hidden_dim=96, dropout=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_dim), nn.ReLU()
        )
        self.attention = nn.Sequential(nn.Linear(emb_dim, 32), nn.Tanh(), nn.Linear(32, 1))
        self.regressor = nn.Sequential(
            nn.Linear(emb_dim, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1)
        )
    def forward(self, x, mask):
        h = self.encoder(x)
        scores = self.attention(h).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        emb = (h * weights.unsqueeze(-1)).sum(dim=1)
        pred = self.regressor(emb).squeeze(-1)
        return pred, emb, weights


def fit_feature_scaler(df_c, train_players, feature_cols):
    train_rows = df_c[df_c[PLAYER_COL].isin(train_players)]
    imp = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_raw = train_rows[feature_cols].replace([np.inf, -np.inf], np.nan).to_numpy()
    X_imp = imp.fit_transform(X_raw)
    scaler.fit(X_imp)
    return imp, scaler


class ImputeScaleWrapper:
    def __init__(self, imputer, scaler):
        self.imputer = imputer
        self.scaler = scaler
    def transform(self, X):
        X_arr = np.asarray(X)
        X_arr = np.where(np.isfinite(X_arr), X_arr, np.nan)
        return self.scaler.transform(self.imputer.transform(X_arr))


def train_neural_model(model, train_loader, val_loader, epochs=80, patience=12, lr=1e-3, weight_decay=1e-4):
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()
    best_mae = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, mask, y, _ in train_loader:
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            pred, _, _ = model(x, mask)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        preds, ys = [], []
        with torch.no_grad():
            for x, mask, y, _ in val_loader:
                x, mask = x.to(DEVICE), mask.to(DEVICE)
                pred, _, _ = model(x, mask)
                preds.extend(pred.cpu().numpy())
                ys.extend(y.numpy())
        mae = mean_absolute_error(ys, preds)
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f'  epoch {epoch:03d} | train loss={np.mean(losses):.3f} | val MAE={mae:.1f}')
        if wait >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_mae


def predict_and_embed(model, loader):
    model.eval()
    preds, ys, players, embs, weights_all = [], [], [], [], []
    with torch.no_grad():
        for x, mask, y, p in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            pred, emb, weights = model(x, mask)
            preds.extend(pred.cpu().numpy())
            ys.extend(y.numpy())
            players.extend(list(p))
            embs.append(emb.cpu().numpy())
            if weights is not None:
                weights_all.append(weights.cpu().numpy())
    embs = np.vstack(embs)
    out = pd.DataFrame(embs, columns=[f'emb_{i:02d}' for i in range(embs.shape[1])])
    out[PLAYER_COL] = players
    out['neural_pred'] = preds
    out['elo'] = ys
    weights_arr = np.vstack(weights_all) if weights_all else None
    return out, weights_arr


def train_selected_v9_neural_model_for_seed(cadence, seed):
    """Entraîne le modèle V9 choisi pour une cadence et une seed."""
    seed_everything(seed)
    kind = V9_KIND_BY_CADENCE[cadence]
    obj = cadence_data[cadence]
    df_c = obj['df']
    train_players = obj['train_players']
    val_players = obj['val_players']

    # Le modèle V9 est entraîné pour être multi-N, donc il voit N=3/5/10/30 pendant l'entraînement.
    TRAIN_N_CHOICES = N_VALUES
    fixed_val_n = 10 if 10 in N_VALUES else max(N_VALUES)

    print('\n' + '=' * 80)
    print(f"Entraînement V9 neural | {cadence.upper()} | seed={seed} | kind={kind} | val fixed N={fixed_val_n}")
    print('=' * 80)

    imputer_g, scaler_g = fit_feature_scaler(df_c, train_players, GAME_FEATURE_COLS)
    transform_wrapper = ImputeScaleWrapper(imputer_g, scaler_g)

    train_ds = PlayerSetDataset(df_c, train_players, GAME_FEATURE_COLS, max_n=MAX_N,
                                n_choices=TRAIN_N_CHOICES, fixed_n=None, seed=seed,
                                scaler=transform_wrapper)
    val_ds = PlayerSetDataset(df_c, val_players, GAME_FEATURE_COLS, max_n=MAX_N,
                              fixed_n=fixed_val_n, seed=seed + 1,
                              scaler=transform_wrapper)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    input_dim = len(GAME_FEATURE_COLS)
    if kind == 'mean':
        model = DeepSetsMean(input_dim, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    elif kind == 'attention':
        model = DeepSetsAttention(input_dim, emb_dim=EMB_DIM, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    else:
        raise ValueError(f"neural_kind inconnu: {kind}")

    model, val_mae = train_neural_model(model, train_loader, val_loader, EPOCHS, PATIENCE, LR, WEIGHT_DECAY)
    return {'model': model.to(DEVICE), 'scaler': transform_wrapper, 'kind': kind, 'val_mae': val_mae, 'seed': seed}


v9_selected_neural_models = {}

def get_v9_neural_model(cadence, seed):
    """Retourne le modèle neural V9. Si RETRAIN_V9_NEURAL_PER_SEED=True, un modèle par seed."""
    if RETRAIN_V9_NEURAL_PER_SEED:
        key = (cadence, seed)
    else:
        key = (cadence, 'shared')
    if key not in v9_selected_neural_models:
        train_seed = seed if RETRAIN_V9_NEURAL_PER_SEED else RANDOM_STATE
        v9_selected_neural_models[key] = train_selected_v9_neural_model_for_seed(cadence, train_seed)
    return v9_selected_neural_models[key]

print('✓ Fonctions V9 neural prêtes.')

# %%
# ============================================================
# 10. Modèle game-level V10 : une ligne = une partie, entraîné exact-N
# ============================================================

# Correction méthodologique : pour tester N=3/5/10/30, le modèle game-level V10
# est entraîné sur exactement N parties par joueur train, pas sur les 30 parties.
# Les lignes game-level sont pondérées par 1 / nb_parties_du_joueur dans le train.

def get_game_feature_cols_safe(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    banned_exact = {
        TARGET_COL, 'target_elo', 'elo', 'game_id',
        'white_elo', 'black_elo', 'opponent_elo', 'elo_diff', 'rating_diff'
    }
    cols = []
    for c in numeric_cols:
        cl = c.lower()
        if c in banned_exact:
            continue
        if any(p in cl for p in ['elo', 'rating', 'opponent', 'advers', 'bucket']):
            continue
        if any(p in cl for p in ['month', 'timestamp', 'date']):
            continue
        cols.append(c)
    return cols


def fit_game_level_lgbm(df_c, train_players, N, seed=42, random_state=42):
    """Entraîne LightGBM au niveau partie sur exactement N parties par joueur train."""
    train_games = sample_exact_n_games(df_c, train_players, N, seed=seed).copy()
    if train_games.empty:
        raise ValueError(f'Aucune partie train échantillonnée pour N={N}.')

    feature_cols = get_game_feature_cols_safe(train_games)
    X_train = train_games[feature_cols].replace([np.inf, -np.inf], np.nan)
    y_train = train_games[TARGET_COL].astype(float)

    # Pondération : chaque joueur contribue autant au total.
    counts = train_games.groupby(PLAYER_COL)[PLAYER_COL].transform('count').astype(float)
    sample_weight = 1.0 / counts.clip(lower=1.0)

    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train)

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train_imp, y_train, sample_weight=sample_weight.values)

    return {
        'model': model,
        'imputer': imputer,
        'feature_cols': feature_cols,
        'n_train_games': len(train_games),
        'N_train': N,
        'weighted': True,
    }


def add_game_predictions(df_c, game_model_obj):
    """Ajoute pred_game_elo à toutes les parties de df_c."""
    model = game_model_obj['model']
    imputer = game_model_obj['imputer']
    feature_cols = game_model_obj['feature_cols']
    out = df_c.copy()
    X = out[feature_cols].replace([np.inf, -np.inf], np.nan)
    X_imp = imputer.transform(X)
    out['pred_game_elo'] = model.predict(X_imp)
    return out

print('✓ Fonctions V10 game-level corrigées prêtes.')

# %%

# ============================================================
# 10. Agrégation des prédictions game-level au niveau joueur
# ============================================================

def _trimmed_mean(values):
    values = np.asarray(values, dtype=float)
    if len(values) >= 5:
        values = np.sort(values)[1:-1]
    return float(np.mean(values))


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if np.all(~np.isfinite(weights)) or np.nansum(weights) <= 1e-9:
        return float(np.nanmean(values))
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return float(np.sum(values * weights) / (np.sum(weights) + 1e-9))


def sample_exact_n_games_with_predictions(pred_games, players, n_games, seed=42):
    """Échantillonne exactement N parties par joueur depuis une table avec pred_game_elo."""
    rng = np.random.default_rng(seed)
    rows = []
    for p in players:
        g = pred_games[pred_games[PLAYER_COL] == p]
        if len(g) < n_games:
            continue
        idx = rng.choice(g.index.to_numpy(), size=n_games, replace=False)
        rows.append(pred_games.loc[idx])
    if not rows:
        return pd.DataFrame(columns=pred_games.columns)
    return pd.concat(rows, ignore_index=True)


def aggregate_game_predictions(sampled_games, aggregator='mean'):
    """
    Agrège les prédictions game-level pour obtenir une ligne par joueur.
    Retourne aussi des mesures de dispersion utilisées comme incertitude.
    """
    rows = []
    for p, g in sampled_games.groupby(PLAYER_COL):
        preds = g['pred_game_elo'].astype(float).values
        true_elo = float(g[TARGET_COL].median())

        if aggregator == 'mean':
            final_pred = float(np.mean(preds))
        elif aggregator == 'median':
            final_pred = float(np.median(preds))
        elif aggregator == 'trimmed_mean':
            final_pred = _trimmed_mean(preds)
        elif aggregator == 'weighted_log_nmoves':
            if 'n_moves' in g.columns:
                weights = np.log1p(g['n_moves'].astype(float).clip(lower=0).values)
            else:
                weights = np.ones(len(preds))
            final_pred = _weighted_mean(preds, weights)
        else:
            raise ValueError(f'Aggregator inconnu: {aggregator}')

        q25, q75 = np.quantile(preds, [0.25, 0.75]) if len(preds) > 1 else (preds[0], preds[0])
        rows.append({
            PLAYER_COL: p,
            'elo': true_elo,
            'pred_player_elo': final_pred,
            'game_pred_mean': float(np.mean(preds)),
            'game_pred_median': float(np.median(preds)),
            'game_pred_std': float(np.std(preds, ddof=0)),
            'game_pred_iqr': float(q75 - q25),
            'game_pred_range': float(np.max(preds) - np.min(preds)),
            'game_pred_min': float(np.min(preds)),
            'game_pred_max': float(np.max(preds)),
            'n_games': int(len(preds)),
            'mean_n_moves': float(g['n_moves'].mean()) if 'n_moves' in g.columns else np.nan,
            'aggregator': aggregator,
        })
    return pd.DataFrame(rows)


def evaluate_player_predictions(df_pred):
    y = df_pred['elo'].astype(float).values
    pred = df_pred['pred_player_elo'].astype(float).values
    return {
        'MAE': mean_absolute_error(y, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y, pred))),
        'R2': r2_score(y, pred),
        'n_players': len(df_pred),
        'dispersion_abs_error_corr_spearman': df_pred[['game_pred_std', 'abs_error']].corr(method='spearman').iloc[0, 1] if 'abs_error' in df_pred.columns and df_pred['game_pred_std'].nunique() > 1 else np.nan,
        'dispersion_abs_error_corr_pearson': df_pred[['game_pred_std', 'abs_error']].corr(method='pearson').iloc[0, 1] if 'abs_error' in df_pred.columns and df_pred['game_pred_std'].nunique() > 1 else np.nan,
    }

# %%
# ============================================================
# 12. Fonctions d'évaluation V9 best et V10 game-level corrigées
# ============================================================

def build_v9_best_dataset(cadence, players, N, seed, use_embedding=True):
    """Construit le dataset V9 : V8 player features exact-N + embedding V9 exact-N."""
    obj = cadence_data[cadence]
    df_c = obj['df']
    player_df = build_player_dataset_exact_n(df_c, players, N, seed=seed)

    if not use_embedding:
        return player_df

    model_obj = get_v9_neural_model(cadence, seed)
    model = model_obj['model']
    transform_wrapper = model_obj['scaler']

    ds = PlayerSetDataset(df_c, players, GAME_FEATURE_COLS, max_n=MAX_N,
                          fixed_n=N, seed=seed, scaler=transform_wrapper)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    emb_df, _ = predict_and_embed(model, loader)
    return merge_embeddings(player_df, emb_df)


def fit_predict_lgbm(train_df, pred_df, target='elo', random_state=42):
    """Fit LightGBM on train_df and returns predictions for pred_df."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError('LightGBM non disponible')
    feature_cols = get_player_feature_cols_safe(train_df)
    feature_cols = [c for c in feature_cols if c in pred_df.columns]
    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(train_df[feature_cols].replace([np.inf, -np.inf], np.nan))
    y_train = train_df[target].astype(float).values
    X_pred = imp.transform(pred_df[feature_cols].replace([np.inf, -np.inf], np.nan))

    model = lgb.LGBMRegressor(
        n_estimators=450,
        learning_rate=0.04,
        max_depth=5,
        num_leaves=31,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_pred)
    return pred, {'model': model, 'imputer': imp, 'feature_cols': feature_cols}


def regression_metrics(y_true, pred):
    return {
        'MAE': mean_absolute_error(y_true, pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, pred))),
        'R2': r2_score(y_true, pred),
    }


def eval_v9_best_one(cadence, N, seed):
    """Évalue V9 sur train -> test, avec val gardé pour incertitude/calibration."""
    obj = cadence_data[cadence]
    tr, va, te = obj['train_players'], obj['val_players'], obj['test_players']

    train_df = build_v9_best_dataset(cadence, tr, N, seed, use_embedding=True)
    val_df = build_v9_best_dataset(cadence, va, N, seed, use_embedding=True)
    test_df = build_v9_best_dataset(cadence, te, N, seed, use_embedding=True)

    pred_test, fitted = fit_predict_lgbm(train_df, test_df, target='elo', random_state=seed)
    pred_val = fitted['model'].predict(fitted['imputer'].transform(val_df[fitted['feature_cols']].replace([np.inf, -np.inf], np.nan)))

    y_test = test_df['elo'].astype(float).values
    y_val = val_df['elo'].astype(float).values
    m = regression_metrics(y_test, pred_test)
    kind = V9_KIND_BY_CADENCE[cadence]

    row = {
        'cadence': cadence, 'N': N, 'seed': seed,
        'approach': V9_APPROACH_BY_CADENCE[cadence],
        'family': 'V9 embedding + LightGBM',
        'aggregator': f'feature+embedding_{kind}',
        'MAE': m['MAE'], 'RMSE': m['RMSE'], 'R2': m['R2'],
        'n_train': len(train_df), 'n_val': len(val_df), 'n_test': len(test_df),
        'n_features': len(fitted['feature_cols']),
        'v9_neural_seed': seed if RETRAIN_V9_NEURAL_PER_SEED else RANDOM_STATE,
    }
    pred_table_test = pd.DataFrame({PLAYER_COL: test_df[PLAYER_COL].values if PLAYER_COL in test_df.columns else np.arange(len(test_df)),
                                    'elo': y_test, 'pred': pred_test, 'method': 'V9'})
    pred_table_val = pd.DataFrame({PLAYER_COL: val_df[PLAYER_COL].values if PLAYER_COL in val_df.columns else np.arange(len(val_df)),
                                   'elo': y_val, 'pred': pred_val, 'method': 'V9'})
    return row, pred_table_val, pred_table_test, fitted


def eval_v10_one(cadence, N, seed, game_model_obj=None):
    """Évalue V10 exact-N pour tous les agrégateurs."""
    obj = cadence_data[cadence]
    df_c = obj['df']
    train_players = obj['train_players']
    val_players = obj['val_players']
    test_players = obj['test_players']

    if game_model_obj is None:
        game_model_obj = fit_game_level_lgbm(df_c, train_players, N=N, seed=seed, random_state=seed)

    pred_games = add_game_predictions(df_c, game_model_obj)

    sampled_val = sample_exact_n_games_with_predictions(pred_games, val_players, N, seed=seed)
    sampled_test = sample_exact_n_games_with_predictions(pred_games, test_players, N, seed=seed)

    rows = []
    pred_tables = {}

    for agg in V10_AGGREGATORS:
        val_agg = aggregate_game_predictions(sampled_val, aggregator=agg)
        test_agg = aggregate_game_predictions(sampled_test, aggregator=agg)

        test_agg['abs_error'] = np.abs(test_agg['elo'].astype(float) - test_agg['pred_player_elo'].astype(float))
        val_agg['abs_error'] = np.abs(val_agg['elo'].astype(float) - val_agg['pred_player_elo'].astype(float))

        y_test = test_agg['elo'].astype(float).values
        pred_test = test_agg['pred_player_elo'].astype(float).values
        m = regression_metrics(y_test, pred_test)

        if len(test_agg) > 3 and 'game_pred_std' in test_agg.columns and test_agg['game_pred_std'].nunique() > 1:
            corr_s = test_agg[['game_pred_std', 'abs_error']].corr(method='spearman').iloc[0, 1]
            corr_p = test_agg[['game_pred_std', 'abs_error']].corr(method='pearson').iloc[0, 1]
        else:
            corr_s = np.nan
            corr_p = np.nan

        rows.append({
            'cadence': cadence,
            'N': N,
            'seed': seed,
            'approach': f'V10 game-level + {agg}',
            'family': 'V10 game-level hierarchical',
            'aggregator': agg,
            'MAE': m['MAE'],
            'RMSE': m['RMSE'],
            'R2': m['R2'],
            'dispersion_abs_error_corr_spearman': corr_s,
            'dispersion_abs_error_corr_pearson': corr_p,
            'n_test': len(test_agg),
            'n_train_games': game_model_obj['n_train_games'],
            'n_features': len(game_model_obj['feature_cols']),
            'v10_N_train': game_model_obj.get('N_train', N),
            'game_weighted': game_model_obj.get('weighted', False),
        })

        pred_tables[agg] = {'val': val_agg, 'test': test_agg}

    return rows, pred_tables, game_model_obj

# %%
# ============================================================
# 13. Expérience principale corrigée : V9 best vs V10 late aggregation
# ============================================================

v9_best_results = []
v10_results = []
v9_prediction_tables = {}
v10_prediction_tables = {}
v10_game_models = {}
v9_fitted_models = {}

for cadence in V9_KIND_BY_CADENCE.keys():
    if cadence not in cadence_data:
        print(f"⚠️ Cadence absente : {cadence}")
        continue

    print('\n' + '=' * 110)
    print(f"CADENCE : {cadence.upper()} | N testés = {N_VALUES}")
    print('=' * 110)

    for seed in V10_VS_V9_SEEDS:
        print(f"\n######## seed={seed} ########")

        # Entraîne/récupère le neural V9 de cette cadence/seed une seule fois,
        # puis l'utilise pour tous les N.
        _ = get_v9_neural_model(cadence, seed)

        for N in N_VALUES:
            print(f"\n--- cadence={cadence} | N={N} | seed={seed} ---")

            # V9 best baseline à N exact.
            row_v9, pred_val_v9, pred_test_v9, fitted_v9 = eval_v9_best_one(cadence, N, seed)
            v9_best_results.append(row_v9)
            v9_prediction_tables[(cadence, N, seed)] = {'val': pred_val_v9, 'test': pred_test_v9}
            v9_fitted_models[(cadence, N, seed)] = fitted_v9
            print(f"V9 best | MAE={row_v9['MAE']:.1f} | R2={row_v9['R2']:.3f} | features={row_v9['n_features']}")

            # V10 game-level hiérarchique entraîné exact-N.
            rows_v10, pred_tables_v10, game_model_obj = eval_v10_one(cadence, N, seed, game_model_obj=None)
            v10_results.extend(rows_v10)
            v10_prediction_tables[(cadence, N, seed)] = pred_tables_v10
            v10_game_models[(cadence, N, seed)] = game_model_obj

            tmp = pd.DataFrame(rows_v10).sort_values('MAE')
            display(tmp[['cadence','N','seed','approach','MAE','R2','dispersion_abs_error_corr_spearman','n_test','n_train_games']])

v9_best_df = pd.DataFrame(v9_best_results)
v10_results_df = pd.DataFrame(v10_results)

print('\n✓ Expérience V9 best vs V10 corrigée terminée.')
display(v9_best_df.head())
display(v10_results_df.head())

# %%
# ============================================================
# 14. Résumé : comparer V10 contre V9 best pour chaque cadence et N
# ============================================================

v9_best_summary = (
    v9_best_df
    .groupby(['cadence','N','approach','family','aggregator'])
    .agg(
        MAE_mean=('MAE','mean'), MAE_std=('MAE','std'),
        RMSE_mean=('RMSE','mean'), R2_mean=('R2','mean'),
        n_runs=('seed','count'), n_test=('n_test','mean'), n_features=('n_features','mean')
    )
    .reset_index()
)
v9_best_summary['dispersion_abs_error_corr_spearman'] = np.nan

v10_summary = (
    v10_results_df
    .groupby(['cadence','N','approach','family','aggregator'])
    .agg(
        MAE_mean=('MAE','mean'), MAE_std=('MAE','std'),
        RMSE_mean=('RMSE','mean'), R2_mean=('R2','mean'),
        n_runs=('seed','count'), n_test=('n_test','mean'), n_features=('n_features','mean'),
        n_train_games=('n_train_games','mean'),
        dispersion_abs_error_corr_spearman=('dispersion_abs_error_corr_spearman','mean'),
        dispersion_abs_error_corr_pearson=('dispersion_abs_error_corr_pearson','mean')
    )
    .reset_index()
)

comparison_v9_v10 = pd.concat([v9_best_summary, v10_summary], ignore_index=True, sort=False)
comparison_v9_v10['MAE_display'] = comparison_v9_v10['MAE_mean'].round(1).astype(str) + ' ± ' + comparison_v9_v10['MAE_std'].round(1).astype(str)
comparison_v9_v10 = comparison_v9_v10.sort_values(['cadence','N','MAE_mean'])

display(comparison_v9_v10[[
    'cadence','N','family','approach','MAE_display','R2_mean',
    'dispersion_abs_error_corr_spearman','n_runs','n_test','n_features'
]])

print('Meilleure méthode par cadence et N :')
best_by_cadence_N = comparison_v9_v10.sort_values(['cadence','N','MAE_mean']).groupby(['cadence','N']).first().reset_index()
display(best_by_cadence_N[['cadence','N','family','approach','MAE_display','R2_mean','dispersion_abs_error_corr_spearman']])

# Gains/pertes de chaque V10 par rapport à V9 best de la même cadence et du même N.
v9_ref = v9_best_summary[['cadence','N','MAE_mean']].rename(columns={'MAE_mean':'MAE_V9_best'})
gain_vs_v9 = v10_summary.merge(v9_ref, on=['cadence','N'], how='left')
gain_vs_v9['gain_vs_V9_best'] = gain_vs_v9['MAE_V9_best'] - gain_vs_v9['MAE_mean']
gain_vs_v9['MAE_display'] = gain_vs_v9['MAE_mean'].round(1).astype(str) + ' ± ' + gain_vs_v9['MAE_std'].round(1).astype(str)

display(
    gain_vs_v9
    .sort_values(['cadence', 'N', 'MAE_mean'])
    [[
        'cadence','N','approach','MAE_display','MAE_V9_best','gain_vs_V9_best',
        'dispersion_abs_error_corr_spearman','n_runs'
    ]]
)

print('Lecture : gain_vs_V9_best > 0 signifie que V10 bat V9 best au même N. Sinon V9 reste meilleure en MAE ponctuelle.')

# %%
# ============================================================
# 14b. Analyse par buckets Elo : biais et régression vers la moyenne
# ============================================================

# Objectif : vérifier si les modèles surestiment les faibles Elo et sous-estiment les forts Elo.
# Biais = moyenne(prediction - Elo réel).
# Biais positif  => surestimation.
# Biais négatif  => sous-estimation.

BUCKET_Q = 5
bucket_rows = []


def _add_elo_buckets(df, q=5):
    out = df.copy()
    out['elo_bucket'] = pd.qcut(
        out['elo'].rank(method='first'),
        q=min(q, len(out)),
        duplicates='drop'
    )
    return out

# V9 predictions par seed/N/cadence
for (cadence, N, seed), tables in v9_prediction_tables.items():
    dfp = tables['test'].copy()
    dfp = dfp.rename(columns={'pred': 'pred_elo'})
    dfp['approach'] = V9_APPROACH_BY_CADENCE[cadence]
    dfp['family'] = 'V9 embedding + LightGBM'
    dfp['aggregator'] = 'feature+embedding'
    dfp['cadence'] = cadence
    dfp['N'] = N
    dfp['seed'] = seed
    dfp['abs_error'] = np.abs(dfp['pred_elo'] - dfp['elo'])
    dfp['bias'] = dfp['pred_elo'] - dfp['elo']
    dfp = _add_elo_buckets(dfp, BUCKET_Q)
    bucket_rows.append(dfp)

# V10 best aggregator for each cadence/N/seed? On garde tous les agrégateurs pour diagnostic.
for (cadence, N, seed), agg_tables in v10_prediction_tables.items():
    for agg, tables in agg_tables.items():
        dfp = tables['test'].copy()
        dfp = dfp.rename(columns={'pred_player_elo': 'pred_elo'})
        dfp['approach'] = f'V10 game-level + {agg}'
        dfp['family'] = 'V10 game-level hierarchical'
        dfp['aggregator'] = agg
        dfp['cadence'] = cadence
        dfp['N'] = N
        dfp['seed'] = seed
        if 'abs_error' not in dfp.columns:
            dfp['abs_error'] = np.abs(dfp['pred_elo'] - dfp['elo'])
        dfp['bias'] = dfp['pred_elo'] - dfp['elo']
        dfp = _add_elo_buckets(dfp, BUCKET_Q)
        bucket_rows.append(dfp)

bucket_pred_df = pd.concat(bucket_rows, ignore_index=True) if bucket_rows else pd.DataFrame()

bucket_summary = (
    bucket_pred_df
    .groupby(['cadence','N','family','approach','aggregator','elo_bucket'], observed=False)
    .agg(
        MAE=('abs_error','mean'),
        bias_mean=('bias','mean'),
        pred_mean=('pred_elo','mean'),
        elo_mean=('elo','mean'),
        n=('elo','count')
    )
    .reset_index()
    .sort_values(['cadence','N','family','approach','elo_bucket'])
)

display(bucket_summary)

print('Lecture : si bias_mean > 0 dans les faibles buckets et < 0 dans les forts buckets, le modèle compresse vers le centre.')

# %%
# ============================================================
# 15. Incertitude V10 : globale vs adaptative par dispersion — CORRIGÉE
# ============================================================

# Idée :
# V10 donne plusieurs prédictions locales par partie.
# Leur dispersion peut servir à adapter la largeur de l'intervalle joueur.
#
# On compare :
# - intervalle global     : pred ± q_global
# - intervalle adaptatif : pred ± q_norm * sigma_joueur
#
# Correction :
# - la colonne de dispersion est game_pred_std, pas pred_std.

EPS_SIGMA = 30.0  # évite les intervalles trop petits quand game_pred_std est proche de 0
v10_uncertainty_rows = []


def conformal_global(val_df, test_df, pred_col="pred_player_elo", alpha=0.20):
    cal_err = np.abs(
        val_df["elo"].astype(float).values
        - val_df[pred_col].astype(float).values
    )

    q = float(np.quantile(cal_err, 1 - alpha))

    lower = test_df[pred_col].astype(float).values - q
    upper = test_df[pred_col].astype(float).values + q

    covered = (
        (test_df["elo"].astype(float).values >= lower)
        & (test_df["elo"].astype(float).values <= upper)
    ).mean()

    width = float(np.mean(upper - lower))

    return q, covered, width


def conformal_adaptive_by_dispersion(
    val_df,
    test_df,
    pred_col="pred_player_elo",
    sigma_col="game_pred_std",
    alpha=0.20
):
    """
    Intervalle adaptatif basé sur la dispersion des prédictions game-level.

    Si game_pred_std est élevée :
        intervalle plus large.
    Si game_pred_std est faible :
        intervalle plus court.

    Remarque :
    EPS_SIGMA empêche d'avoir des intervalles quasi nuls.
    """

    if sigma_col not in val_df.columns:
        raise KeyError(
            f"Colonne {sigma_col} absente dans val_df. "
            f"Colonnes disponibles : {val_df.columns.tolist()}"
        )

    if sigma_col not in test_df.columns:
        raise KeyError(
            f"Colonne {sigma_col} absente dans test_df. "
            f"Colonnes disponibles : {test_df.columns.tolist()}"
        )

    sigma_val = val_df[sigma_col].fillna(0).astype(float).values + EPS_SIGMA
    sigma_test = test_df[sigma_col].fillna(0).astype(float).values + EPS_SIGMA

    norm_err = np.abs(
        val_df["elo"].astype(float).values
        - val_df[pred_col].astype(float).values
    ) / sigma_val

    q = float(np.quantile(norm_err, 1 - alpha))

    lower = test_df[pred_col].astype(float).values - q * sigma_test
    upper = test_df[pred_col].astype(float).values + q * sigma_test

    covered = (
        (test_df["elo"].astype(float).values >= lower)
        & (test_df["elo"].astype(float).values <= upper)
    ).mean()

    width = float(np.mean(upper - lower))

    return q, covered, width


for (cadence, N, seed), agg_tables in v10_prediction_tables.items():
    for agg, tables in agg_tables.items():
        val_df = tables["val"].copy()
        test_df = tables["test"].copy()

        mae = mean_absolute_error(
            test_df["elo"].astype(float),
            test_df["pred_player_elo"].astype(float)
        )

        for alpha in UNCERTAINTY_ALPHAS:
            qg, covg, widthg = conformal_global(
                val_df,
                test_df,
                alpha=alpha
            )

            qa, cova, widtha = conformal_adaptive_by_dispersion(
                val_df,
                test_df,
                sigma_col="game_pred_std",
                alpha=alpha
            )

            v10_uncertainty_rows.append({
                "cadence": cadence,
                "N": N,
                "seed": seed,
                "aggregator": agg,
                "alpha": alpha,
                "target_coverage": 1 - alpha,
                "MAE": mae,

                "global_coverage": covg,
                "global_width": widthg,
                "global_q": qg,

                "adaptive_coverage": cova,
                "adaptive_width": widtha,
                "adaptive_q": qa,

                "width_gain_adaptive_vs_global": widthg - widtha,
            })


v10_uncertainty_df = pd.DataFrame(v10_uncertainty_rows)

v10_uncertainty_summary = (
    v10_uncertainty_df
    .groupby(["cadence", "N", "aggregator", "alpha", "target_coverage"])
    .agg(
        MAE_mean=("MAE", "mean"),
        global_coverage=("global_coverage", "mean"),
        global_width=("global_width", "mean"),
        adaptive_coverage=("adaptive_coverage", "mean"),
        adaptive_width=("adaptive_width", "mean"),
        width_gain_adaptive_vs_global=("width_gain_adaptive_vs_global", "mean"),
        n_runs=("seed", "count")
    )
    .reset_index()
    .sort_values(["cadence", "N", "alpha", "adaptive_width"])
)

display(v10_uncertainty_summary[[
    "cadence",
    "N",
    "aggregator",
    "target_coverage",
    "MAE_mean",
    "global_coverage",
    "global_width",
    "adaptive_coverage",
    "adaptive_width",
    "width_gain_adaptive_vs_global",
    "n_runs"
]])

print("""
Lecture :
- adaptive_width < global_width signifie que l'intervalle adaptatif est plus court.
- Mais il faut aussi regarder adaptive_coverage.
- Le bon cas est :
    adaptive_width plus petit
    ET adaptive_coverage proche de target_coverage.
""")

# %%
# ============================================================
# 16. Analyse : la dispersion prédit-elle les erreurs ? + exemples joueurs
# ============================================================

DISP_COL = "game_pred_std"
IQR_COL = "game_pred_iqr"
RANGE_COL = "game_pred_range"

dispersion_rows = []

for (cadence, N, seed), agg_tables in v10_prediction_tables.items():
    for agg, tables in agg_tables.items():
        test_df = tables["test"].copy()
        if len(test_df) < 10:
            continue
        if DISP_COL not in test_df.columns:
            continue
        if "abs_error" not in test_df.columns:
            test_df["abs_error"] = np.abs(test_df["elo"].astype(float) - test_df["pred_player_elo"].astype(float))

        test_df["dispersion_bin"] = pd.qcut(
            test_df[DISP_COL].rank(method="first"),
            q=3,
            labels=["faible dispersion", "moyenne", "forte dispersion"]
        )

        grp = (
            test_df
            .groupby("dispersion_bin", observed=False)
            .agg(MAE=("abs_error", "mean"), pred_std_mean=(DISP_COL, "mean"), n=("abs_error", "count"))
            .reset_index()
        )
        grp["cadence"] = cadence
        grp["N"] = N
        grp["seed"] = seed
        grp["aggregator"] = agg
        dispersion_rows.append(grp)


dispersion_detail_df = pd.concat(dispersion_rows, ignore_index=True) if dispersion_rows else pd.DataFrame()

if not dispersion_detail_df.empty:
    dispersion_summary = (
        dispersion_detail_df
        .groupby(["cadence", "N", "aggregator", "dispersion_bin"], observed=False)
        .agg(MAE=("MAE", "mean"), pred_std_mean=("pred_std_mean", "mean"), n=("n", "sum"))
        .reset_index()
    )
    display(dispersion_summary.sort_values(["cadence", "N", "aggregator", "dispersion_bin"]))
else:
    dispersion_summary = pd.DataFrame()
    print("Pas assez de données pour l’analyse par tertiles de dispersion.")

# Cas concrets : pour chaque cadence et N, afficher le meilleur agrégateur V10 moyen.
for cadence in V9_KIND_BY_CADENCE.keys():
    for N in N_VALUES:
        sub = v10_summary[(v10_summary['cadence'] == cadence) & (v10_summary['N'] == N)].sort_values('MAE_mean')
        if sub.empty:
            continue
        best_agg = sub.iloc[0]['aggregator']
        key = (cadence, N, V10_VS_V9_SEEDS[0])
        if key not in v10_prediction_tables or best_agg not in v10_prediction_tables[key]:
            continue
        df_case = v10_prediction_tables[key][best_agg]['test'].copy()
        if 'abs_error' not in df_case.columns:
            df_case['abs_error'] = np.abs(df_case['elo'].astype(float) - df_case['pred_player_elo'].astype(float))

        print('\n' + '='*90)
        print(f'CAS {cadence.upper()} | N={N} | meilleur agrégateur V10={best_agg}')
        print('='*90)

        display_cols = [PLAYER_COL, 'elo', 'pred_player_elo', 'abs_error', DISP_COL, IQR_COL, RANGE_COL]
        display_cols = [c for c in display_cols if c in df_case.columns]

        print('Joueurs avec faible dispersion :')
        display(df_case.sort_values(DISP_COL).head(5)[display_cols])
        print('Joueurs avec forte dispersion :')
        display(df_case.sort_values(DISP_COL, ascending=False).head(5)[display_cols])

# %%
# ============================================================
# 17. Visualisations synthétiques
# ============================================================

# 1) MAE par cadence/N : V9 vs meilleur V10
best_v10_by_cadence_N = (
    v10_summary.sort_values(['cadence','N','MAE_mean'])
    .groupby(['cadence','N']).first().reset_index()
)
v9_plot = v9_best_summary.copy()
v9_plot['plot_label'] = 'V9 best'
best_v10_by_cadence_N['plot_label'] = best_v10_by_cadence_N['aggregator'].apply(lambda x: f'V10 best ({x})')
plot_df = pd.concat([
    v9_plot[['cadence','N','MAE_mean','MAE_std','plot_label']],
    best_v10_by_cadence_N[['cadence','N','MAE_mean','MAE_std','plot_label']]
], ignore_index=True)

for cadence in plot_df['cadence'].unique():
    sub = plot_df[plot_df['cadence'] == cadence].copy()
    plt.figure(figsize=(9, 5))
    for label in sub['plot_label'].unique():
        s = sub[sub['plot_label'] == label].sort_values('N')
        plt.errorbar(s['N'], s['MAE_mean'], yerr=s['MAE_std'], marker='o', capsize=4, label=label)
    plt.title(f'V9 best vs V10 best — {cadence.upper()}')
    plt.xlabel('Nombre de parties N')
    plt.ylabel('MAE Elo')
    plt.grid(True, alpha=0.3)
    plt.legend()    plt.show()

# 2) Biais par bucket Elo pour V9 et meilleur V10 uniquement.
if 'bucket_summary' in globals() and not bucket_summary.empty:
    for cadence in bucket_summary['cadence'].unique():
        for N in N_VALUES:
            sub_best_v10 = best_v10_by_cadence_N[(best_v10_by_cadence_N['cadence']==cadence) & (best_v10_by_cadence_N['N']==N)]
            if sub_best_v10.empty:
                continue
            best_agg = sub_best_v10.iloc[0]['aggregator']
            sub = bucket_summary[(bucket_summary['cadence']==cadence) & (bucket_summary['N']==N)]
            sub = sub[(sub['family']=='V9 embedding + LightGBM') | ((sub['family']=='V10 game-level hierarchical') & (sub['aggregator']==best_agg))]
            if sub.empty:
                continue
            plt.figure(figsize=(10, 4))
            for approach in sub['approach'].unique():
                s = sub[sub['approach']==approach].copy()
                x = np.arange(len(s))
                plt.plot(x, s['bias_mean'], marker='o', label=approach)
            plt.axhline(0, linestyle='--')
            plt.title(f'Biais moyen par bucket Elo — {cadence.upper()} N={N}')
            plt.xlabel('Bucket Elo réel croissant')
            plt.ylabel('Biais moyen pred - true')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()

# %%
# ============================================================
# 18. Sauvegarde des résultats V10 corrigée vs V9 best
# ============================================================

SAVE_DIR = CACHE_DIR if 'CACHE_DIR' in globals() else Path('./_v10_cache')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

v9_best_df.to_csv(SAVE_DIR / 'v10_corrected_v9_best_by_seed.csv', index=False)
v10_results_df.to_csv(SAVE_DIR / 'v10_corrected_game_level_by_seed.csv', index=False)
comparison_v9_v10.to_csv(SAVE_DIR / 'v10_corrected_comparison_v9_v10.csv', index=False)
gain_vs_v9.to_csv(SAVE_DIR / 'v10_corrected_gain_vs_v9.csv', index=False)

if 'v10_uncertainty_summary' in globals():
    v10_uncertainty_summary.to_csv(SAVE_DIR / 'v10_corrected_uncertainty_summary.csv', index=False)
if 'dispersion_summary' in globals() and not dispersion_summary.empty:
    dispersion_summary.to_csv(SAVE_DIR / 'v10_corrected_dispersion_summary.csv', index=False)
if 'bucket_summary' in globals() and not bucket_summary.empty:
    bucket_summary.to_csv(SAVE_DIR / 'v10_corrected_bucket_bias_summary.csv', index=False)

print('Résultats sauvegardés dans :', SAVE_DIR)
print('Note méthodologique : si RETRAIN_V9_NEURAL_PER_SEED=True, la variance inclut une partie de la variabilité neural V9.')
print('Si vous mettez RETRAIN_V9_NEURAL_PER_SEED=False, les écarts-types reflètent surtout l’échantillonnage exact-N et LightGBM.')

# %%
# ============================================================
# 19. Analyse de compression des prédictions V10
# ============================================================

# Objectif :
# Montrer si V10 ramène les prédictions vers une zone centrale.
#
# On analyse par bucket Elo réel :
# - Elo réel moyen
# - Elo prédit moyen
# - biais moyen = pred - true
# - MAE
#
# Si :
# - biais positif chez les faibles Elo  => surestimation
# - biais négatif chez les forts Elo    => sous-estimation
# alors le modèle compresse les extrêmes vers la moyenne.

def add_elo_buckets(df, y_col="elo", n_bins=5):
    df = df.copy()
    try:
        df["elo_bucket"] = pd.qcut(
            df[y_col],
            q=n_bins,
            labels=False,
            duplicates="drop"
        )
    except ValueError:
        df["elo_bucket"] = pd.cut(
            df[y_col],
            bins=n_bins,
            labels=False,
            duplicates="drop"
        )
    return df


compression_rows = []

for (cadence, N, seed), agg_tables in v10_prediction_tables.items():
    for agg, tables in agg_tables.items():
        test_df = tables["test"].copy()

        if "pred_player_elo" not in test_df.columns:
            print(f"⚠️ pred_player_elo absent pour {cadence}, N={N}, seed={seed}, agg={agg}")
            continue

        test_df = add_elo_buckets(test_df, y_col="elo", n_bins=5)

        test_df["bias"] = (
            test_df["pred_player_elo"].astype(float)
            - test_df["elo"].astype(float)
        )

        test_df["abs_error"] = np.abs(test_df["bias"])

        bucket_stats = (
            test_df
            .groupby("elo_bucket", observed=False)
            .agg(
                elo_true_mean=("elo", "mean"),
                elo_true_min=("elo", "min"),
                elo_true_max=("elo", "max"),
                pred_mean=("pred_player_elo", "mean"),
                bias_mean=("bias", "mean"),
                MAE=("abs_error", "mean"),
                n=("elo", "count")
            )
            .reset_index()
        )

        bucket_stats["cadence"] = cadence
        bucket_stats["N"] = N
        bucket_stats["seed"] = seed
        bucket_stats["aggregator"] = agg

        compression_rows.append(bucket_stats)


compression_detail_df = (
    pd.concat(compression_rows, ignore_index=True)
    if compression_rows
    else pd.DataFrame()
)

compression_summary = (
    compression_detail_df
    .groupby(["cadence", "N", "aggregator", "elo_bucket"], observed=False)
    .agg(
        elo_true_mean=("elo_true_mean", "mean"),
        elo_true_min=("elo_true_min", "mean"),
        elo_true_max=("elo_true_max", "mean"),
        pred_mean=("pred_mean", "mean"),
        bias_mean=("bias_mean", "mean"),
        MAE=("MAE", "mean"),
        n=("n", "sum")
    )
    .reset_index()
    .sort_values(["cadence", "N", "aggregator", "elo_bucket"])
)

display(compression_summary)

print("""
Lecture :
- bias_mean > 0 : le modèle surestime ce bucket.
- bias_mean < 0 : le modèle sous-estime ce bucket.
- Si les faibles Elo ont un biais positif et les forts Elo un biais négatif,
  alors V10 compresse les prédictions vers la moyenne.
""")

# %%
# ============================================================
# 20. Visualisation : vraie moyenne vs prédite moyenne par bucket
# ============================================================

# On affiche uniquement le meilleur agrégateur V10 par cadence/N,
# pour éviter trop de figures.

if "v10_summary" not in globals():
    raise RuntimeError("v10_summary est absent. Exécute d'abord la cellule de résumé V10.")

best_v10_by_cadence_N = (
    v10_summary
    .sort_values(["cadence", "N", "MAE_mean"])
    .groupby(["cadence", "N"])
    .first()
    .reset_index()
)

for _, row in best_v10_by_cadence_N.iterrows():
    cadence = row["cadence"]
    N = row["N"]
    best_agg = row["aggregator"]

    sub = compression_summary[
        (compression_summary["cadence"] == cadence)
        & (compression_summary["N"] == N)
        & (compression_summary["aggregator"] == best_agg)
    ].copy()

    if sub.empty:
        continue

    plt.figure(figsize=(8, 5))

    plt.plot(
        sub["elo_bucket"],
        sub["elo_true_mean"],
        marker="o",
        label="Elo réel moyen"
    )

    plt.plot(
        sub["elo_bucket"],
        sub["pred_mean"],
        marker="o",
        label="Elo prédit moyen V10"
    )

    plt.title(f"Compression V10 — {cadence.upper()} | N={N} | agg={best_agg}")
    plt.xlabel("Bucket Elo réel, du plus faible au plus fort")
    plt.ylabel("Elo moyen")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

    display(sub[[
        "cadence",
        "N",
        "aggregator",
        "elo_bucket",
        "elo_true_mean",
        "pred_mean",
        "bias_mean",
        "MAE",
        "n"
    ]])

# %%
# ============================================================
# 21. Calibration post-hoc de V10 sur validation
# ============================================================

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def regression_metrics_local(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "bias": float(np.mean(y_pred - y_true)),
    }


def get_v10_calibration_features(df):
    """
    Crée les features utilisées pour calibrer V10.
    Robuste aux colonnes absentes.
    """
    df = df.copy()

    candidate_cols = [
        "pred_player_elo",
        "game_pred_std",
        "game_pred_iqr",
        "game_pred_range",
        "game_pred_min",
        "game_pred_max",
        "game_pred_mean",
        "n_games",
        "N",
    ]

    # Ajouter des colonnes dérivées utiles
    if "pred_player_elo" in df.columns:
        df["pred_centered"] = df["pred_player_elo"] - df["pred_player_elo"].mean()
        candidate_cols.append("pred_centered")

    if "game_pred_std" in df.columns and "pred_player_elo" in df.columns:
        df["std_x_pred"] = df["game_pred_std"] * df["pred_player_elo"]
        candidate_cols.append("std_x_pred")

    feature_cols = [c for c in candidate_cols if c in df.columns]

    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, feature_cols


def calibrate_v10_one(val_df, test_df, method="ridge"):
    """
    Calibre les prédictions V10 sur validation puis applique sur test.

    method='ridge' : calibration linéaire stable.
    method='rf'    : calibration non linéaire, plus risquée.
    """
    val_df = val_df.copy()
    test_df = test_df.copy()

    X_val, feature_cols = get_v10_calibration_features(val_df)
    X_test, _ = get_v10_calibration_features(test_df)

    # Aligner les colonnes au cas où
    X_test = X_test.reindex(columns=feature_cols, fill_value=0)

    y_val = val_df["elo"].astype(float).values

    if method == "ridge":
        calibrator = Ridge(alpha=10.0)
    elif method == "rf":
        calibrator = RandomForestRegressor(
            n_estimators=200,
            max_depth=3,
            random_state=0,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    calibrator.fit(X_val, y_val)

    pred_calibrated = calibrator.predict(X_test)

    return pred_calibrated, calibrator, feature_cols


v10_calibration_rows = []
v10_calibrated_predictions = {}
v10_calibrators = {}

CALIBRATION_METHODS = ["ridge"]  # garde simple. Tu peux ajouter "rf" après.

for (cadence, N, seed), agg_tables in v10_prediction_tables.items():
    for agg, tables in agg_tables.items():
        val_df = tables["val"].copy()
        test_df = tables["test"].copy()

        # Sécurité : ajouter N comme feature explicite
        val_df["N"] = N
        test_df["N"] = N

        y_test = test_df["elo"].astype(float).values
        pred_raw = test_df["pred_player_elo"].astype(float).values

        raw_metrics = regression_metrics_local(y_test, pred_raw)

        for method in CALIBRATION_METHODS:
            pred_calib, calibrator, calib_feature_cols = calibrate_v10_one(
                val_df,
                test_df,
                method=method
            )

            calib_metrics = regression_metrics_local(y_test, pred_calib)

            row = {
                "cadence": cadence,
                "N": N,
                "seed": seed,
                "aggregator": agg,
                "calibration": method,

                "MAE_raw": raw_metrics["MAE"],
                "RMSE_raw": raw_metrics["RMSE"],
                "R2_raw": raw_metrics["R2"],
                "bias_raw": raw_metrics["bias"],

                "MAE_calibrated": calib_metrics["MAE"],
                "RMSE_calibrated": calib_metrics["RMSE"],
                "R2_calibrated": calib_metrics["R2"],
                "bias_calibrated": calib_metrics["bias"],

                "gain_MAE_calibration": raw_metrics["MAE"] - calib_metrics["MAE"],
                "n_test": len(test_df),
                "n_calib_features": len(calib_feature_cols),
            }

            v10_calibration_rows.append(row)

            key = (cadence, N, seed, agg, method)
            v10_calibrators[key] = {
                "calibrator": calibrator,
                "feature_cols": calib_feature_cols,
            }

            pred_table = test_df.copy()
            pred_table["pred_v10_raw"] = pred_raw
            pred_table["pred_v10_calibrated"] = pred_calib
            pred_table["abs_error_raw"] = np.abs(pred_table["elo"] - pred_table["pred_v10_raw"])
            pred_table["abs_error_calibrated"] = np.abs(pred_table["elo"] - pred_table["pred_v10_calibrated"])

            v10_calibrated_predictions[key] = pred_table


v10_calibration_df = pd.DataFrame(v10_calibration_rows)

v10_calibration_summary = (
    v10_calibration_df
    .groupby(["cadence", "N", "aggregator", "calibration"])
    .agg(
        MAE_raw=("MAE_raw", "mean"),
        MAE_raw_std=("MAE_raw", "std"),
        MAE_calibrated=("MAE_calibrated", "mean"),
        MAE_calibrated_std=("MAE_calibrated", "std"),
        gain_MAE_calibration=("gain_MAE_calibration", "mean"),
        R2_raw=("R2_raw", "mean"),
        R2_calibrated=("R2_calibrated", "mean"),
        bias_raw=("bias_raw", "mean"),
        bias_calibrated=("bias_calibrated", "mean"),
        n_runs=("seed", "count")
    )
    .reset_index()
    .sort_values(["cadence", "N", "MAE_calibrated"])
)

display(v10_calibration_summary)

print("""
Lecture :
- gain_MAE_calibration > 0 : la calibration améliore V10.
- bias_calibrated proche de 0 : la calibration corrige le biais global.
- Si la MAE baisse surtout dans les buckets extrêmes, la calibration réduit la compression.
""")

# %%
# ============================================================
# 22. Comparaison finale : V9 best vs V10 brut vs V10 calibré
# ============================================================

# Référence V9 par cadence/N
if "v9_best_df" not in globals():
    raise RuntimeError("v9_best_df absent. Exécute d'abord l'expérience V9/V10.")

v9_ref_summary = (
    v9_best_df
    .groupby(["cadence", "N"])
    .agg(
        MAE_V9=("MAE", "mean"),
        MAE_V9_std=("MAE", "std"),
        R2_V9=("R2", "mean"),
        n_runs_V9=("seed", "count")
    )
    .reset_index()
)

# Meilleur V10 calibré par cadence/N
best_v10_calib = (
    v10_calibration_summary
    .sort_values(["cadence", "N", "MAE_calibrated"])
    .groupby(["cadence", "N"])
    .first()
    .reset_index()
)

final_calib_comparison = best_v10_calib.merge(
    v9_ref_summary,
    on=["cadence", "N"],
    how="left"
)

final_calib_comparison["gain_V10_calibrated_vs_V9"] = (
    final_calib_comparison["MAE_V9"]
    - final_calib_comparison["MAE_calibrated"]
)

final_calib_comparison["gain_V10_raw_vs_V9"] = (
    final_calib_comparison["MAE_V9"]
    - final_calib_comparison["MAE_raw"]
)

final_calib_comparison["MAE_V9_display"] = (
    final_calib_comparison["MAE_V9"].round(1).astype(str)
    + " ± "
    + final_calib_comparison["MAE_V9_std"].round(1).astype(str)
)

final_calib_comparison["MAE_raw_display"] = (
    final_calib_comparison["MAE_raw"].round(1).astype(str)
    + " ± "
    + final_calib_comparison["MAE_raw_std"].round(1).astype(str)
)

final_calib_comparison["MAE_calibrated_display"] = (
    final_calib_comparison["MAE_calibrated"].round(1).astype(str)
    + " ± "
    + final_calib_comparison["MAE_calibrated_std"].round(1).astype(str)
)

display(final_calib_comparison[[
    "cadence",
    "N",
    "aggregator",
    "calibration",
    "MAE_V9_display",
    "MAE_raw_display",
    "MAE_calibrated_display",
    "gain_MAE_calibration",
    "gain_V10_raw_vs_V9",
    "gain_V10_calibrated_vs_V9",
    "bias_raw",
    "bias_calibrated",
    "R2_V9",
    "R2_raw",
    "R2_calibrated",
    "n_runs"
]])

print("""
Lecture :
- gain_MAE_calibration > 0 : calibration améliore V10.
- gain_V10_calibrated_vs_V9 > 0 : V10 calibré bat V9.
- Si V10 calibré reste inférieur à V9, on garde V9 comme modèle final,
  mais on peut discuter la calibration comme amélioration partielle de V10.
""")

# %%
# ============================================================
# 23. Analyse par bucket après calibration : V10 brut vs V10 calibré
# ============================================================

# Objectif :
# Vérifier si la calibration réduit la compression des extrêmes.
#
# On compare par bucket Elo réel :
# - biais brut       = pred_v10_raw - elo
# - biais calibré   = pred_v10_calibrated - elo
# - MAE brute
# - MAE calibrée
#
# Si la calibration marche bien :
# - le biais positif des faibles Elo doit diminuer ;
# - le biais négatif des forts Elo doit se rapprocher de 0 ;
# - la MAE des buckets extrêmes doit baisser.

def add_elo_buckets_for_calibration(df, y_col="elo", n_bins=5):
    df = df.copy()
    try:
        df["elo_bucket"] = pd.qcut(
            df[y_col],
            q=n_bins,
            labels=False,
            duplicates="drop"
        )
    except ValueError:
        df["elo_bucket"] = pd.cut(
            df[y_col],
            bins=n_bins,
            labels=False,
            duplicates="drop"
        )
    return df


calibration_bucket_rows = []

for key, df_pred in v10_calibrated_predictions.items():
    cadence, N, seed, agg, method = key

    df = df_pred.copy()

    required_cols = ["elo", "pred_v10_raw", "pred_v10_calibrated"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"⚠️ Colonnes manquantes pour {key} : {missing_cols}")
        continue

    df = add_elo_buckets_for_calibration(df, y_col="elo", n_bins=5)

    df["bias_raw"] = (
        df["pred_v10_raw"].astype(float)
        - df["elo"].astype(float)
    )

    df["bias_calibrated"] = (
        df["pred_v10_calibrated"].astype(float)
        - df["elo"].astype(float)
    )

    df["abs_error_raw"] = np.abs(df["bias_raw"])
    df["abs_error_calibrated"] = np.abs(df["bias_calibrated"])

    bucket_stats = (
        df
        .groupby("elo_bucket", observed=False)
        .agg(
            elo_true_mean=("elo", "mean"),
            elo_true_min=("elo", "min"),
            elo_true_max=("elo", "max"),

            pred_raw_mean=("pred_v10_raw", "mean"),
            pred_calibrated_mean=("pred_v10_calibrated", "mean"),

            bias_raw_mean=("bias_raw", "mean"),
            bias_calibrated_mean=("bias_calibrated", "mean"),

            MAE_raw=("abs_error_raw", "mean"),
            MAE_calibrated=("abs_error_calibrated", "mean"),

            n=("elo", "count")
        )
        .reset_index()
    )

    bucket_stats["cadence"] = cadence
    bucket_stats["N"] = N
    bucket_stats["seed"] = seed
    bucket_stats["aggregator"] = agg
    bucket_stats["calibration"] = method

    calibration_bucket_rows.append(bucket_stats)


calibration_bucket_detail_df = (
    pd.concat(calibration_bucket_rows, ignore_index=True)
    if calibration_bucket_rows
    else pd.DataFrame()
)

calibration_bucket_summary = (
    calibration_bucket_detail_df
    .groupby(
        ["cadence", "N", "aggregator", "calibration", "elo_bucket"],
        observed=False
    )
    .agg(
        elo_true_mean=("elo_true_mean", "mean"),
        elo_true_min=("elo_true_min", "mean"),
        elo_true_max=("elo_true_max", "mean"),

        pred_raw_mean=("pred_raw_mean", "mean"),
        pred_calibrated_mean=("pred_calibrated_mean", "mean"),

        bias_raw_mean=("bias_raw_mean", "mean"),
        bias_calibrated_mean=("bias_calibrated_mean", "mean"),

        MAE_raw=("MAE_raw", "mean"),
        MAE_calibrated=("MAE_calibrated", "mean"),

        n=("n", "sum")
    )
    .reset_index()
)

calibration_bucket_summary["bias_reduction_abs"] = (
    calibration_bucket_summary["bias_raw_mean"].abs()
    - calibration_bucket_summary["bias_calibrated_mean"].abs()
)

calibration_bucket_summary["MAE_gain_calibration"] = (
    calibration_bucket_summary["MAE_raw"]
    - calibration_bucket_summary["MAE_calibrated"]
)

calibration_bucket_summary = calibration_bucket_summary.sort_values(
    ["cadence", "N", "aggregator", "elo_bucket"]
)

display(calibration_bucket_summary)

print("""
Lecture :
- bias_raw_mean : biais de V10 brut dans le bucket.
- bias_calibrated_mean : biais après calibration.
- bias_reduction_abs > 0 : la calibration réduit le biais en valeur absolue.
- MAE_gain_calibration > 0 : la calibration améliore la MAE du bucket.

Ce qu'on veut voir :
- faibles Elo : biais positif brut réduit après calibration ;
- forts Elo : biais négatif brut moins négatif après calibration ;
- buckets extrêmes : MAE_calibrated < MAE_raw.
""")

# %% [markdown]
#
# ## Comment lire cette V10 corrigée
#
# Cette version répond directement à la remarque méthodologique : **V10 est comparée à la meilleure baseline V9**, pas seulement à V8.
#
# Les questions à regarder sont :
#
# 1. **Performance ponctuelle** : est-ce que V10 game-level bat ou approche V9 best ?
# 2. **Interprétabilité** : est-ce que les prédictions par partie montrent des profils stables/instables ?
# 3. **Incertitude** : est-ce que l’intervalle adaptatif basé sur `pred_std` réduit la largeur par rapport au conformal global ?
#
# Même si V10 ne bat pas V9 en MAE, elle peut être utile si la dispersion des prédictions par partie est corrélée aux erreurs ou permet des intervalles plus personnalisés.