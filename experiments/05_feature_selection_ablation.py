# PMind experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Notebook V8 corrigé — PMind 730 par cadence — Exactement 10 parties par joueur
#
# Cette version reprend le notebook V8 et ajoute :
# - chargement direct d'une arborescence déjà séparée par cadence (`bullet`, `blitz`, `rapid`) ;
# - correction de `rate_castle` : vraie fréquence de roque, plus `n_castles` pour le comptage ;
# - correction de `tag_bloc` pour une classification plus cohérente des features ;
# - conservation de la taxonomie d'erreurs basée sur `cp_loss_clean`.
#
# Structure attendue du dossier racine :
# ```text
# Data PMind 730/
#   Bullet/ moves.csv games.csv players.csv summary.csv
#   Blitz/  moves.csv games.csv players.csv summary.csv
#   Rapid/  moves.csv games.csv players.csv summary.csv
# ```
#
#
#
# **Modification de cette version :** toutes les expériences sont construites avec **exactement 10 parties par joueur**.  
# Les joueurs ayant moins de 10 parties dans une cadence sont exclus, puis 10 parties sont échantillonnées de façon déterministe pour chaque joueur éligible.

# %%
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown]
# ---
# ## STEP 0 — Imports & Configuration

# %%
# NOTEBOOK: !pip install catboost

# %%
# — Google Colab uniquement —
# Si vous exécutez ce notebook dans Colab, montez d'abord votre Drive.
# from google.colab import drive
# drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import zlib
from pathlib import Path
from sklearn.model_selection import (
    train_test_split, cross_val_score, RepeatedKFold, KFold
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print("✓ XGBoost disponible")
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠ XGBoost non disponible")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print("✓ LightGBM disponible")
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠ LightGBM non disponible")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
    print("✓ CatBoost disponible")
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠ CatBoost non disponible")

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted')
plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', 80)
pd.set_option('display.float_format', '{:.3f}'.format)

# ── Chemin du nouveau dataset PMind 730 déjà séparé par cadence ───────────────
# À adapter selon le nom exact dans votre Drive.
DATA_ROOT = Path('/content/drive/MyDrive/Data_Pmind_700-30')

# Si vos dossiers portent des noms légèrement différents, adaptez ici.
# Le code tente aussi une détection automatique.
CADENCE_FOLDER_HINTS = {
    'bullet': ['bullet', 'bulle', 'bulet', 'bul', 'belet'],
    'blitz':  ['blitz', 'bles', 'bletz', 'bilet'],
    'rapid':  ['rapid', 'rapide'],
}

RANDOM_STATE = 42

# ── Seuils partagés ────────────────────────────────────────────────────────────
EVAL_CAP           = 900
LOSS_CAP           = 600
THRESHOLD_WIN      = 150
THRESHOLD_LOSE     = -150
LOW_CLOCK_RATIO    = 0.20
EXACT_N_GAMES_PER_PLAYER = 10
MIN_GAMES_ELIGIBLE = EXACT_N_GAMES_PER_PLAYER

# ── Cadences à traiter ─────────────────────────────────────────────────────────
CADENCES = ['bullet', 'blitz', 'rapid']

# ── Palette blocs ──────────────────────────────────────────────────────────────
PALETTE_BLOC = {
    'Qualite':   '#3498db',
    'Temps':     '#e74c3c',
    'Contexte':  '#2ecc71',
    'Stabilite': '#f39c12',
    'Style':     '#9b59b6',
    'Autre':     '#95a5a6',
}



# ── Mode expérimental : exactement N parties par joueur ───────────────────────
# Important : MIN_GAMES_ELIGIBLE sert seulement à filtrer les joueurs.
# Ensuite, on échantillonne exactement EXACT_N_GAMES_PER_PLAYER parties par joueur.
USE_EXACT_N_GAMES = True

def _stable_group_seed(name, base_seed=RANDOM_STATE):
    """Seed déterministe par joueur, stable entre exécutions Python."""
    key = str(name).encode("utf-8")
    return (base_seed + zlib.adler32(key)) % (2**32 - 1)


def sample_exact_n_games_per_player(df: pd.DataFrame,
                                    n_games: int = EXACT_N_GAMES_PER_PLAYER,
                                    player_col: str = "target_name",
                                    random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Garde uniquement les joueurs ayant au moins n_games parties, puis échantillonne
    exactement n_games lignes/parties par joueur.

    À utiliser sur une table game-level, par exemple game_rich.
    """
    if player_col not in df.columns:
        raise KeyError(f"Colonne joueur absente : {player_col}")
    if "game_id" not in df.columns:
        raise KeyError("La table doit contenir une colonne game_id.")

    counts = df.groupby(player_col)["game_id"].count()
    eligible = counts[counts >= n_games].index
    dff = df[df[player_col].isin(eligible)].copy()

    sampled_parts = []
    for player, group in dff.groupby(player_col, sort=False):
        seed = _stable_group_seed(player, random_state)
        sampled_parts.append(group.sample(n=n_games, random_state=seed, replace=False))

    if len(sampled_parts) == 0:
        return dff.iloc[0:0].copy()

    sampled = pd.concat(sampled_parts, axis=0).sort_values([player_col, "game_id"]).reset_index(drop=True)
    return sampled

print(f"\n✓ Configuration OK")
print(f"  DATA_ROOT={DATA_ROOT}")
print(f"  Mode exact N games/player : {USE_EXACT_N_GAMES} | N={EXACT_N_GAMES_PER_PLAYER}")
print(f"  XGBoost={XGBOOST_AVAILABLE}, LightGBM={LIGHTGBM_AVAILABLE}, CatBoost={CATBOOST_AVAILABLE}")

# %% [markdown]
# ---
# ## STEP 1 — Chargement & Nettoyage

# %%
def _normalize_name(s: str) -> str:
    """Normalisation légère pour détecter les dossiers de cadence."""
    return str(s).strip().lower().replace(" ", "").replace("_", "").replace("-", "")


def find_cadence_folder(root: Path, cadence: str) -> Path:
    """
    Trouve automatiquement le dossier correspondant à une cadence.
    Exemple attendu : Bullet/, Blitz/, Rapid/.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"DATA_ROOT introuvable : {root}\n"
            "Vérifiez le chemin dans la cellule de configuration."
        )

    subdirs = [p for p in root.iterdir() if p.is_dir()]
    hints = CADENCE_FOLDER_HINTS.get(cadence, [cadence])

    # 1) match exact ou quasi exact
    for p in subdirs:
        n = _normalize_name(p.name)
        if n == cadence or cadence in n:
            return p

    # 2) aliases
    for p in subdirs:
        n = _normalize_name(p.name)
        if any(h in n for h in hints):
            return p

    raise FileNotFoundError(
        f"Impossible de trouver le dossier pour cadence={cadence} dans {root}.\n"
        f"Dossiers trouvés : {[p.name for p in subdirs]}"
    )


def _read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant : {path}")
    return pd.read_csv(path)


def load_one_cadence(root: Path, cadence: str):
    """
    Charge moves/games/summary/players pour une cadence déjà séparée.
    On préfixe game_id et target_name/username avec la cadence pour éviter
    les collisions entre dossiers et les doublons de joueurs entre cadences.
    """
    folder = find_cadence_folder(root, cadence)

    moves   = _read_csv_required(folder / 'moves.csv')
    games   = _read_csv_required(folder / 'games.csv')
    players = _read_csv_required(folder / 'players.csv')
    summary_path = folder / 'summary.csv'
    summary = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()

    # Harmonisation cadence
    if 'speed' not in games.columns:
        games['speed'] = cadence
    else:
        games['speed'] = cadence

    # Préfixer les game_id pour éviter collisions entre cadences
    if 'game_id' in games.columns:
        games['game_id'] = cadence + "__" + games['game_id'].astype(str)
    if 'game_id' in moves.columns:
        moves['game_id'] = cadence + "__" + moves['game_id'].astype(str)
    if not summary.empty and 'game_id' in summary.columns:
        summary['game_id'] = cadence + "__" + summary['game_id'].astype(str)

    # Préfixer les joueurs pour avoir un profil joueur-cadence distinct
    # Exemple : bullet__john et blitz__john sont deux profils séparés.
    if 'target_name' in games.columns:
        games['raw_target_name'] = games['target_name'].astype(str)
        games['target_name'] = cadence + "__" + games['target_name'].astype(str)

    if 'username' in players.columns:
        players['raw_username'] = players['username'].astype(str)
        players['username'] = cadence + "__" + players['username'].astype(str)

    # Traces
    moves['source_cadence'] = cadence
    games['source_cadence'] = cadence
    players['source_cadence'] = cadence
    if not summary.empty:
        summary['source_cadence'] = cadence

    print(f"✓ {cadence:6s} | folder={folder.name:15s} | "
          f"moves={moves.shape} games={games.shape} players={players.shape}")

    return moves, games, summary, players


def load_split_cadence_data(root: Path, cadences=CADENCES):
    """
    Charge et concatène les dossiers bullet/blitz/rapid.
    Les profils sont séparés par cadence via le préfixe target_name/username.
    """
    moves_list, games_list, summary_list, players_list = [], [], [], []

    print(f"\nChargement dataset séparé par cadence depuis : {root}")
    for cadence in cadences:
        m, g, s, p = load_one_cadence(root, cadence)
        moves_list.append(m)
        games_list.append(g)
        if not s.empty:
            summary_list.append(s)
        players_list.append(p)

    moves   = pd.concat(moves_list, ignore_index=True)
    games   = pd.concat(games_list, ignore_index=True)
    summary = pd.concat(summary_list, ignore_index=True) if summary_list else pd.DataFrame()
    players = pd.concat(players_list, ignore_index=True)

    print("\n=== Dataset concaténé ===")
    print(f"moves   : {moves.shape[0]:>10,} × {moves.shape[1]}")
    print(f"games   : {games.shape[0]:>10,} × {games.shape[1]}")
    print(f"players : {players.shape[0]:>10,} × {players.shape[1]}")
    if not summary.empty:
        print(f"summary : {summary.shape[0]:>10,} × {summary.shape[1]}")

    return moves, games, summary, players


moves, games, summary, players = load_split_cadence_data(DATA_ROOT, CADENCES)

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

moves_clean = clean_moves(moves)

# %% [markdown]
# ---
# ## STEP 2 — Agrégation Game-Level Enrichie (V8)
#
# **Nouveautés V8 :**
# - Quantiles (p25, p50, p75, p90) et IQR pour les pertes
# - **Lissage Laplace** pour les taux rares (robuste aux petits N)
# - Counts (dénominateurs) — permettent au modèle de pondérer la fiabilité
# - Features d'**interaction Temps × Qualité** (`loss_per_second`, `blunder_pressure_ratio`, etc.)

# %%
def laplace_smooth(k, n, alpha=1.0):
    """Lissage Laplace : (k + alpha) / (n + 2*alpha). Robuste aux petits n."""
    return (k + alpha) / (n + 2 * alpha)


def build_game_features_v8(moves_clean: pd.DataFrame) -> pd.DataFrame:
    m = moves_clean[moves_clean['is_target_move']].copy()
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
    agg['rate_castle']    = ('is_castle',    'mean')  # vraie fréquence de roque
    agg['n_castles']       = ('is_castle',    'sum')   # nombre brut de roques
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


game_features = build_game_features_v8(moves_clean)

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

game_rich = enrich_game_features(game_features, games)

# %% [markdown]
# ---
# ## STEP 3 — Agrégation Player-Level Robuste (V8)
#
# **Nouveautés V8 :**
# - `mean + median + IQR` pour les features importantes (robustesse aux outliers)
# - `std` conservé seulement pour les features stables
# - Stabilité inter-parties enrichie (IQR + quantiles de `loss_mean`)

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
        'rate_capture', 'rate_check', 'rate_castle', 'n_castles', 'rate_promotion', 'n_moves',
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

    agg_dict['n_games']  = ('game_id', 'count')
    agg_dict['win_rate'] = ('target_score', 'mean')

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
    target = players[['username', 'elo', 'elo_bucket']].drop_duplicates(subset=['username']).rename(columns={'username': 'target_name'})
    df = pf.merge(target, on='target_name', how='inner')
    print(f"✓ Dataset final : {df.shape[0]} joueurs × {df.shape[1]} colonnes")
    return df


# Dataset global construit avec exactement 10 parties par joueur
game_rich_model = sample_exact_n_games_per_player(
    game_rich,
    n_games=EXACT_N_GAMES_PER_PLAYER,
    player_col='target_name',
    random_state=RANDOM_STATE
)

print(f"\n✓ Dataset global exact-{EXACT_N_GAMES_PER_PLAYER} games : "
      f"{game_rich_model['target_name'].nunique()} joueurs × {len(game_rich_model):,} parties")

player_feat = build_player_features_v8(game_rich_model)
dataset     = prepare_target(player_feat, players)

# %% [markdown]
# ---
# ## STEP 4 — Séparation par Cadence

# %%
def build_cadence_datasets(game_rich, players, n_games=EXACT_N_GAMES_PER_PLAYER):
    """
    Construit un dataset player-level V8 par cadence en utilisant exactement
    n_games parties par joueur.

    - Les joueurs avec moins de n_games parties dans la cadence sont exclus.
    - Les joueurs éligibles sont représentés par exactement n_games parties.
    """
    datasets = {}

    print(f"\n{'='*60}")
    print(f"  STEP 4 — Datasets par cadence (exactement {n_games} parties/joueur)")
    print(f"{'='*60}")

    for cadence in CADENCES:
        gr_cad_all = game_rich[game_rich['speed'] == cadence].copy()

        n_games_pp = gr_cad_all.groupby('target_name').size()
        n_players_total = n_games_pp.shape[0]
        n_players_eligible = (n_games_pp >= n_games).sum()

        gr_cad = sample_exact_n_games_per_player(
            gr_cad_all,
            n_games=n_games,
            player_col='target_name',
            random_state=RANDOM_STATE
        )

        pf = build_player_features_v8(gr_cad)
        ds = prepare_target(pf, players)
        datasets[cadence] = ds

        print(
            f"\n  {cadence.upper()} : {len(ds)} joueurs utilisés "
            f"/ {n_players_total} joueurs initiaux "
            f"({n_players_eligible} éligibles >= {n_games} parties)"
        )
        print(f"  → parties utilisées : {len(gr_cad):,} "
              f"({n_games} par joueur, si cible disponible)")

    return datasets


cadence_datasets = build_cadence_datasets(game_rich, players, n_games=EXACT_N_GAMES_PER_PLAYER)

# %% [markdown]
# ---
# ## STEP 5 — Utilitaires de Modélisation

# %%
LEAKAGE_COLS = ['target_name', 'elo', 'elo_bucket', 'dominant_speed']


def get_feature_cols(dataset: pd.DataFrame) -> list:
    numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLS]
    suspicious   = [c for c in feature_cols if 'elo' in c.lower() or 'bucket' in c.lower()]
    return [c for c in feature_cols if c not in suspicious]


def make_models():
    """Retourne les modèles à comparer."""
    models = {
        'Ridge': Pipeline([
            ('i', SimpleImputer(strategy='median')),
            ('s', StandardScaler()),
            ('m', Ridge(alpha=10.0))
        ])
    }

    if XGBOOST_AVAILABLE:
        models['XGBoost'] = Pipeline([
            ('i', SimpleImputer(strategy='median')),
            ('m', xgb.XGBRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
            ))
        ])

    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = Pipeline([
            ('i', SimpleImputer(strategy='median')),
            ('m', lgb.LGBMRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
            ))
        ])

    if CATBOOST_AVAILABLE:
        models['CatBoost'] = Pipeline([
            ('i', SimpleImputer(strategy='median')),
            ('m', CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05,
                loss_function='MAE',
                random_seed=RANDOM_STATE,
                verbose=False
            ))
        ])

    return models


def evaluate_cv(name, model, X, y, n_splits=5, n_repeats=3):
    """Évaluation par Repeated K-Fold."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)
    mae_scores  = -cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_absolute_error', n_jobs=-1)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=rkf, scoring='neg_mean_squared_error', n_jobs=-1))
    r2_scores   = cross_val_score(model, X, y, cv=rkf, scoring='r2', n_jobs=-1)
    return {
        'name': name,
        'MAE_mean': mae_scores.mean(),
        'MAE_std': mae_scores.std(),
        'RMSE_mean': rmse_scores.mean(),
        'R2_mean': r2_scores.mean(),
        'R2_std': r2_scores.std()
    }


def tag_bloc(name):
    """
    Classification corrigée des features par bloc.
    L'ordre est important :
    - les features de temps sont détectées avant les phases ;
    - les features de stabilité sont détectées avant les features de qualité ;
    - les features phase + erreur sont mises en Contexte ;
    - les features loss_* par phase restent en Qualité ;
    - win_rate et time-control restent dans Autre.
    """
    n = str(name).lower()

    # Variables globales / méta
    if any(k in n for k in ['win_rate', 'tc_base', 'tc_increment', 'n_games']):
        return 'Autre'

    # Temps
    if any(k in n for k in [
        'time', 'clock', 'fast_move', 'pressure', 'low_time',
        'loss_per_second', 'blunder_pressure', 'time_on_error', 'fast_good'
    ]):
        return 'Temps'

    # Stabilité / cohérence
    if any(k in n for k in [
        'variance', 'stability', 'drift', 'drop', 'cv',
        'prop_games', 'between_game', '_std', '_iqr'
    ]):
        # Les std/iqr de capture/check/n_moves restent plutôt style
        if any(k in n for k in ['capture', 'check', 'castle', 'promotion', 'n_moves', 'ply']):
            return 'Style'
        # Les std/iqr de temps restent temps
        if any(k in n for k in ['time', 'clock', 'pressure']):
            return 'Temps'
        # Les std/iqr de loss restent qualité
        if any(k in n for k in ['loss', 'blunder', 'mistake', 'error']):
            return 'Qualite'
        return 'Stabilite'

    # Contexte positionnel ou phase d'erreur
    if any(k in n for k in ['when_', 'winning', 'equal', 'losing', 'opening_vs_endgame']):
        return 'Contexte'
    if any(k in n for k in [
        'rate_blunder_opening', 'rate_blunder_middlegame', 'rate_blunder_endgame',
        'rate_mistake_opening', 'rate_mistake_middlegame', 'rate_mistake_endgame'
    ]):
        return 'Contexte'

    # Style
    if any(k in n for k in [
        'capture', 'check', 'castle', 'promotion',
        'eco', 'speed', 'n_moves', 'ply'
    ]):
        return 'Style'

    # Qualité brute
    if any(k in n for k in [
        'loss', 'rate_good', 'rate_error', 'rate_blunder', 'rate_mistake',
        'rate_inaccuracy', 'n_blunders', 'n_mistakes', 'n_errors',
        'rate_loss'
    ]):
        return 'Qualite'

    return 'Autre'


def run_experiment(label, dataset, feature_subset=None, verbose=True):
    """Lance tous les modèles sur un dataset. Retourne dict {model_name: metrics}."""
    feature_cols = get_feature_cols(dataset)
    if feature_subset is not None:
        feature_cols = [c for c in feature_cols if c in feature_subset]

    X = dataset[feature_cols].copy()
    y = dataset['elo'].values

    if verbose:
        print(f"\n  [{label}] {len(feature_cols)} features, {len(X)} joueurs")

    results = {}
    for model_name, model in make_models().items():
        res = evaluate_cv(model_name, model, X, y)
        results[model_name] = res
        if verbose:
            print(f"    {model_name:10s} MAE={res['MAE_mean']:.1f} ±{res['MAE_std']:.1f}  "
                  f"R²={res['R2_mean']:.3f} ±{res['R2_std']:.3f}")
    return results, feature_cols


print("✓ Utilitaires chargés")

# %% [markdown]
# ---
# ## STEP 6 — Expérience A : Baseline V8 (toutes features)

# %%
print(f"\n{'='*60}")
print("  STEP 6 — Expérience A : Baseline V8 (toutes features)")
print(f"{'='*60}")

exp_A = {}
for cadence in CADENCES:
    print(f"\n--- {cadence.upper()} ---")
    exp_A[cadence], _ = run_experiment(f"A-baseline-{cadence}", cadence_datasets[cadence])

# %% [markdown]
# ---
# ## STEP 7 — Expérience B : Pruning des Features Négatives
#
# On calcule la **permutation importance** et on retire les features avec importance ≤ 0.

# %%
print(f"\n{'='*60}")
print("  STEP 7 — Expérience B : Pruning features négatives")
print(f"{'='*60}")


def get_positive_features(dataset, n_splits=3, n_repeats=2):
    feature_cols = get_feature_cols(dataset)
    X = dataset[feature_cols].copy()
    y = dataset['elo'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    if XGBOOST_AVAILABLE:
        model = Pipeline([('i', SimpleImputer(strategy='median')),
                          ('m', xgb.XGBRegressor(n_estimators=300, max_depth=5,
                              learning_rate=0.05, random_state=RANDOM_STATE,
                              n_jobs=-1, verbosity=0))])
    else:
        model = Pipeline([('i', SimpleImputer(strategy='median')),
                          ('s', StandardScaler()), ('m', Ridge(alpha=10.0))])

    model.fit(X_tr, y_tr)
    perm = permutation_importance(model, X_te, y_te, n_repeats=5,
                                   random_state=RANDOM_STATE,
                                   scoring='neg_mean_absolute_error')
    imp = pd.Series(perm.importances_mean, index=feature_cols)
    positive_features = imp[imp > 0].index.tolist()
    print(f"    Features positives : {len(positive_features)} / {len(feature_cols)}")
    return positive_features, imp


exp_B = {}
positive_features_by_cadence = {}

for cadence in CADENCES:
    print(f"\n--- {cadence.upper()} ---")
    pos_feats, imp = get_positive_features(cadence_datasets[cadence])
    positive_features_by_cadence[cadence] = pos_feats
    exp_B[cadence], _ = run_experiment(
        f"B-pruned-{cadence}", cadence_datasets[cadence], feature_subset=pos_feats
    )

# %% [markdown]
# ---
# ## STEP 8 — Expérience C : Pruning par Bloc (sans Contexte + Stabilité)

# %%
print(f"\n{'='*60}")
print("  STEP 8 — Expérience C : Pruning par bloc (sans Contexte + Stabilité)")
print(f"{'='*60}")


def get_features_without_blocs(dataset, blocs_to_remove):
    feature_cols = get_feature_cols(dataset)
    kept = [c for c in feature_cols if tag_bloc(c) not in blocs_to_remove]
    print(f"    Features gardées : {len(kept)} / {len(feature_cols)} "
          f"(supprimé blocs : {blocs_to_remove})")
    return kept


exp_C = {}
for cadence in CADENCES:
    print(f"\n--- {cadence.upper()} ---")
    kept_feats = get_features_without_blocs(
        cadence_datasets[cadence], blocs_to_remove=['Contexte', 'Stabilite']
    )
    exp_C[cadence], _ = run_experiment(        f"C-no-context-{cadence}", cadence_datasets[cadence], feature_subset=kept_feats
    )

# %% [markdown]
# ---
# ## STEP 9 — Expérience D : Features Robustes Uniquement
#
# On garde seulement : **Qualité** (median/IQR/smooth) + **Temps** (median/IQR/smooth)
# + **Interactions** + `win_rate` + `ply_count` + counts.

# %%
print(f"\n{'='*60}")
print("  STEP 9 — Expérience D : Features robustes uniquement")
print(f"{'='*60}")

ROBUST_KEYWORDS = [
    'loss_mean_mean', 'loss_median_mean', 'loss_p75_mean', 'loss_p90_mean',
    'loss_iqr_mean', 'loss_mean_median', 'loss_median_median',
    'loss_opening_mean_mean', 'loss_opening_median_mean', 'loss_opening_iqr_mean',
    'loss_middlegame_mean_mean', 'loss_middlegame_median_mean',
    'rate_blunder_smooth_mean', 'rate_mistake_smooth_mean', 'rate_error_smooth_mean',
    'rate_blunder_opening_smooth_mean', 'rate_mistake_opening_smooth_mean',
    'rate_blunder_middlegame_smooth_mean', 'rate_mistake_middlegame_smooth_mean',
    'rate_loss_gt100_mean', 'rate_loss_gt200_mean',
    'loss_mean_cv', 'loss_mean_iqr',
    'time_mean_s_mean', 'time_median_s_mean', 'time_iqr_s_mean',
    'time_opening_mean_s_mean', 'time_opening_median_s_mean',
    'time_middlegame_mean_s_mean', 'time_middlegame_median_s_mean',
    'good_fast_move_rate_smooth_mean', 'good_fast_move_rate_smooth_median',
    'blunder_low_time_rate_smooth_mean', 'blunder_low_time_rate_smooth_median',
    'time_mean_on_blunder_mean', 'time_mean_on_good_mean',
    'time_pressure_by_phase_endgame_mean',
    'loss_per_second_mean', 'blunder_pressure_ratio_mean',
    'time_on_error_vs_mean_mean', 'fast_good_vs_error_mean',
    'win_rate', 'ply_count_mean',
    'n_games', 'n_moves_opening_mean', 'n_moves_equal_mean',
]


def get_robust_features(dataset):
    feature_cols = get_feature_cols(dataset)
    robust = [c for c in feature_cols if c in ROBUST_KEYWORDS]
    if len(robust) < 10:
        robust = [c for c in feature_cols
                  if any(kw in c for kw in ['smooth', 'median', 'iqr', 'win_rate',
                                             'loss_per_second', 'blunder_pressure',
                                             'fast_good', 'time_on_error'])]
    print(f"    Features robustes : {len(robust)} / {len(feature_cols)}")
    return robust


exp_D = {}
for cadence in CADENCES:
    print(f"\n--- {cadence.upper()} ---")
    robust_feats = get_robust_features(cadence_datasets[cadence])
    exp_D[cadence], _ = run_experiment(
        f"D-robust-{cadence}", cadence_datasets[cadence], feature_subset=robust_feats
    )

# %% [markdown]
# ---
# ## STEP 11 — Ablation par Bloc

# %%
print(f"\n{'='*60}")
print("  STEP 11 — Ablation par bloc")
print(f"{'='*60}")

ABLATION_CONFIGS = {
    'Temps seul':          ['Temps'],
    'Qualite seule':       ['Qualite'],
    'Temps+Qualite':       ['Temps', 'Qualite'],
    'Temps+Qualite+Style': ['Temps', 'Qualite', 'Style'],
    'Sans Contexte+Stab':  ['Temps', 'Qualite', 'Style', 'Autre'],
    'Tous blocs':          ['Temps', 'Qualite', 'Contexte', 'Stabilite', 'Style', 'Autre'],
}

exp_ablation = {}

for cadence in CADENCES:
    print(f"\n--- {cadence.upper()} ---")
    exp_ablation[cadence] = {}
    dataset_c    = cadence_datasets[cadence]
    feature_cols = get_feature_cols(dataset_c)

    for config_name, blocs_kept in ABLATION_CONFIGS.items():
        feats_kept = [c for c in feature_cols if tag_bloc(c) in blocs_kept]
        if len(feats_kept) < 3:
            continue
        X = dataset_c[feats_kept].copy()
        y = dataset_c['elo'].values
        # Prendre uniquement le meilleur modèle disponible pour l'ablation
        model_name = 'LightGBM' if LIGHTGBM_AVAILABLE else ('XGBoost' if XGBOOST_AVAILABLE else 'Ridge')
        model = list(make_models().values())[-1]   # dernier = meilleur
        res = evaluate_cv(model_name, model, X, y)
        exp_ablation[cadence][config_name] = res
        print(f"  {config_name:25s} | {len(feats_kept):3d} features | "
              f"MAE={res['MAE_mean']:.1f} ±{res['MAE_std']:.1f}  R²={res['R2_mean']:.3f}")

# %% [markdown]
# ---
# ## STEP 12 — Tableau Comparatif Final Toutes Expériences

# %%
print(f"\n{'='*70}")
print("  STEP 12 — Tableau comparatif final")
print(f"{'='*70}")

all_exps = {
    'A-Baseline':     exp_A,
    'B-Pruned':       exp_B,
    'C-NoContext':    exp_C,
    'D-Robust':       exp_D,
    'E-Hybrid':       exp_E,
}

rows = []
for exp_name, exp_dict in all_exps.items():
    for cadence in CADENCES:
        if cadence not in exp_dict:
            continue
        for model_name, res in exp_dict[cadence].items():
            rows.append({
                'Expérience': exp_name,
                'Cadence':    cadence,
                'Modèle':     model_name,
                'MAE':        round(res['MAE_mean'], 1),
                'MAE std':    round(res['MAE_std'],  1),
                'R²':         round(res['R2_mean'],  3),
            })

df_final = pd.DataFrame(rows)

# Affichage par cadence
for cadence in CADENCES:
    sub = df_final[df_final['Cadence'] == cadence].sort_values('MAE')
    print(f"\n=== {cadence.upper()} ===")
    print(sub[['Expérience', 'Modèle', 'MAE', 'MAE std', 'R²']].to_string(index=False))

# Visualisation : MAE par expérience et cadence (meilleur modèle)
best_per_exp_cad = df_final.groupby(['Expérience', 'Cadence'])['MAE'].min().reset_index()
pivot = best_per_exp_cad.pivot(index='Expérience', columns='Cadence', values='MAE')

fig, ax = plt.subplots(figsize=(13, 6))
x      = np.arange(len(pivot))
width  = 0.25
colors = {'bullet': '#e74c3c', 'blitz': '#3498db', 'rapid': '#27ae60'}

for i, cadence in enumerate(CADENCES):
    if cadence in pivot.columns:
        ax.bar(x + i * width, pivot[cadence].values, width,
               label=cadence.capitalize(), color=colors[cadence],
               edgecolor='white', alpha=0.85)

ax.set_xticks(x + width)
ax.set_xticklabels(pivot.index, rotation=20, ha='right', fontsize=9)
ax.set_ylabel('MAE (Elo points) — meilleur modèle')
ax.set_title('Comparaison MAE par expérience et cadence\n(meilleur modèle par exp.)',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## STEP 13 — Feature Importance du Meilleur Modèle par Cadence

# %%
print(f"\n{'='*60}")
print("  STEP 13 — Feature importance du meilleur modèle par cadence")
print(f"{'='*60}")

feature_importance_results = {}

for cadence in CADENCES:
    dataset_c    = cadence_datasets[cadence]
    feature_cols = get_feature_cols(dataset_c)
    X = dataset_c[feature_cols].copy()
    y = dataset_c['elo'].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    imputer  = SimpleImputer(strategy='median')
    X_tr_imp = imputer.fit_transform(X_tr)
    X_te_imp = imputer.transform(X_te)

    if LIGHTGBM_AVAILABLE:
        model_name = 'LightGBM'
        base_model = lgb.LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                                        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    elif XGBOOST_AVAILABLE:
        model_name = 'XGBoost'
        base_model = xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.05,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    else:
        model_name = 'Ridge'
        scaler     = StandardScaler()
        X_tr_imp   = scaler.fit_transform(X_tr_imp)
        X_te_imp   = scaler.transform(X_te_imp)
        base_model = Ridge(alpha=10.0)

    base_model.fit(X_tr_imp, y_tr)

    perm = permutation_importance(base_model, X_te_imp, y_te,
                                   n_repeats=10, random_state=RANDOM_STATE,
                                   scoring='neg_mean_absolute_error')
    imp_df = pd.DataFrame({'feature': feature_cols,
                            'importance': perm.importances_mean,
                            'std':        perm.importances_std})
    imp_df['bloc'] = imp_df['feature'].apply(tag_bloc)
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)
    feature_importance_results[cadence] = imp_df

    # Graphique 3 panneaux
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    top15    = imp_df.head(15).sort_values('importance')
    bottom15 = imp_df.tail(15).sort_values('importance')
    bloc_imp = imp_df.groupby('bloc')['importance'].sum().sort_values(ascending=False)

    axes[0].barh(top15['feature'], top15['importance'],
                  color=top15['bloc'].map(PALETTE_BLOC), edgecolor='white', alpha=0.85,
                  xerr=top15['std'], capsize=3)
    axes[0].axvline(0, color='black', lw=1, linestyle='--', alpha=0.5)
    axes[0].set_title(f'Top 15 features\n{cadence.upper()} — {model_name}', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Permutation Importance (MAE)')
    legend_els = [mpatches.Patch(facecolor=v, label=k)
                  for k, v in PALETTE_BLOC.items() if k in top15['bloc'].values]
    axes[0].legend(handles=legend_els, fontsize=8, loc='lower right')

    axes[1].barh(bottom15['feature'], bottom15['importance'],
                  color=bottom15['bloc'].map(PALETTE_BLOC), edgecolor='white', alpha=0.85,
                  xerr=bottom15['std'], capsize=3)
    axes[1].axvline(0, color='red', lw=1.5, linestyle='--', alpha=0.7, label='Seuil 0')
    axes[1].set_title(f'Bottom 15 (candidats pruning)\n{cadence.upper()}', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Permutation Importance (MAE)')
    axes[1].legend(fontsize=8)

    bloc_colors = [PALETTE_BLOC.get(b, '#95a5a6') for b in bloc_imp.index]
    bars = axes[2].bar(bloc_imp.index, bloc_imp.values, color=bloc_colors,
                        edgecolor='white', alpha=0.85)
    axes[2].axhline(0, color='black', lw=1, linestyle='--', alpha=0.5)
    axes[2].set_title(f'Importance totale par bloc\n{cadence.upper()}', fontsize=11, fontweight='bold')
    for bar, (bloc, val) in zip(bars, bloc_imp.items()):
        axes[2].text(bar.get_x() + bar.get_width()/2,
                     val + abs(bloc_imp.max()) * 0.02,
                     f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    plt.suptitle(f'Feature Importance — {cadence.upper()} — V8', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

    # Résumé texte
    print(f"\n  {cadence.upper()} — Importance par bloc :")
    for bloc, val in bloc_imp.items():
        bar = '#' * max(0, int(val * 15))
        print(f"    {bloc:12s} : {val:+.3f}  {bar}")

# %% [markdown]
# ---
# ## STEP 14 — Synthèse & Recommandations

# %%
print(f"\n{'='*70}")
print("  STEP 14 — SYNTHÈSE & RECOMMANDATIONS")
print(f"{'='*70}")

# Meilleure expérience par cadence
print("\n--- Meilleure expérience par cadence ---")
for cadence in CADENCES:
    best_exp, best_mae = None, float('inf')
    for exp_name, exp_dict in all_exps.items():
        if cadence not in exp_dict:
            continue
        for model_name, res in exp_dict[cadence].items():
            if res['MAE_mean'] < best_mae:
                best_mae = res['MAE_mean']
                best_exp = f"{exp_name} / {model_name}"
    print(f"  {cadence:8s} -> {best_exp}  MAE={best_mae:.1f}")

# Blocs les plus importants
print("\n--- Blocs les plus importants par cadence ---")
for cadence in CADENCES:
    if cadence not in feature_importance_results:
        continue
    imp_df   = feature_importance_results[cadence]
    bloc_imp = imp_df.groupby('bloc')['importance'].sum().sort_values(ascending=False)
    top_pos  = bloc_imp[bloc_imp > 0].head(3)
    print(f"  {cadence:8s} -> {list(top_pos.index)}")

# Candidats au pruning
print("\n--- Candidats au pruning (importance <= 0) ---")
for cadence in CADENCES:
    if cadence not in feature_importance_results:
        continue
    imp_df     = feature_importance_results[cadence]
    candidates = imp_df[imp_df['importance'] <= 0]['feature'].tolist()
    print(f"  {cadence:8s} -> {len(candidates)} features à supprimer")
    for feat in candidates[:5]:
        print(f"    - {feat}")
    if len(candidates) > 5:
        print(f"    ... et {len(candidates)-5} autres")

print("\n--- Recommandations V9 ---")
print("  1. Garder uniquement les features avec importance > 0 (Exp. B)")
print("  2. Tester si le modèle hybride (Exp. E) apporte un gain réel sur plus de données")
print("  3. Comparer Exp. D (robuste) vs Exp. B (pruned) — laquelle est la plus stable ?")
print("  4. Sur données complètes, tester Optuna sur le meilleur modèle V8")
print("  5. Envisager des features temporelles (évolution Elo dans le temps)")
print("\n✓ Notebook V8 terminé.")