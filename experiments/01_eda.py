# PMind experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # PMind Chess — Notebook EDA complet (v4 corrigé)
#
# Ce notebook construit une analyse exploratoire adaptée au rapport. Version corrigée : chargement plus robuste des données, vérifications explicites, et ajout sécurisé de la cible `elo`.

# %%
# Montage Google Drive (uniquement si le notebook est exécuté sur Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    print('Google Drive non monté automatiquement : environnement hors Colab ou module indisponible.')

# %% [markdown]
# ## 0. Installation / imports / configuration

# %%
# Si vous êtes sur Google Colab, décommentez ces lignes.
# from google.colab import drive
# drive.mount('/content/drive')

from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except Exception:
    STATSMODELS_AVAILABLE = False

pd.set_option('display.max_columns', 120)
pd.set_option('display.width', 200)

# ===== À MODIFIER SI BESOIN =====
BASE_DIR = Path('/content/drive/MyDrive/Data Pmind 700-30')
# Pour test local si le dossier existe ailleurs :
if not BASE_DIR.exists():
    BASE_DIR = Path('/mnt/data/Data Pmind 700-30')

OUT_DIR = Path('/content/drive/MyDrive/pmind_eda_outputs')
if not OUT_DIR.parent.exists():
    OUT_DIR = Path('/mnt/data/pmind_eda_outputs')
OUT_DIR.mkdir(parents=True, exist_ok=True)

CADENCES = ['bullet', 'blitz', 'rapid']
RANDOM_STATE = 42

print('BASE_DIR =', BASE_DIR)
print('OUT_DIR  =', OUT_DIR)
print('SciPy:', SCIPY_AVAILABLE, '| statsmodels:', STATSMODELS_AVAILABLE)

# %% [markdown]
# ## 1. Fonctions utilitaires

# %%
def savefig(name, dpi=160):
    path = OUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print('Saved:', path)
    plt.show()


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().replace(' ', '_') for c in df.columns]
    return df


def read_csv_safely(path):
    try:
        return normalize_columns(pd.read_csv(path, low_memory=False))
    except Exception as e:
        print(f'[WARN] Impossible de lire {path}: {e}')
        return pd.DataFrame()


def find_col(df, candidates):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    low_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low_map:
            return low_map[c.lower()]
    return None


def numeric_cols(df, exclude=()):
    exclude = set(exclude)
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


def iqr(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return x.quantile(0.75) - x.quantile(0.25)


def robust_boxplot(df, x, y, title, ylabel, filename, max_points=None):
    tmp = df[[x, y]].dropna().copy()
    if max_points is not None and len(tmp) > max_points:
        tmp = tmp.sample(max_points, random_state=RANDOM_STATE)
    groups = [g[y].values for _, g in tmp.groupby(x)]
    labels = [str(k) for k, _ in tmp.groupby(x)]
    plt.figure(figsize=(8, 4.5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(ylabel)
    savefig(filename)


def add_elo_bucket(df, elo_col='elo'):
    df = df.copy()
    if elo_col not in df.columns:
        return df
    bins = [1000, 1400, 1600, 1800, 2000, 2200]
    labels = ['[1000,1400[', '[1400,1600[', '[1600,1800[', '[1800,2000[', '[2000,2200]']
    if 'elo_bucket' not in df.columns:
        df['elo_bucket'] = pd.cut(df[elo_col], bins=bins, labels=labels, include_lowest=True, right=False)
    return df


def safe_corr(df, cols, target='elo', method='spearman'):
    cols = [c for c in cols if c in df.columns and c != target]
    rows = []
    for c in cols:
        tmp = df[[c, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tmp) < 5 or tmp[c].nunique() <= 1:
            continue
        try:
            val = tmp[c].corr(tmp[target], method=method)
            rows.append({'feature': c, f'{method}_corr_with_elo': val, 'abs_corr': abs(val)})
        except Exception:
            pass
    return pd.DataFrame(rows).sort_values('abs_corr', ascending=False)


def first_existing_col(df, candidates):
    """Retourne la première colonne existante parmi des candidats, insensible à la casse."""
    if df is None or len(df.columns) == 0:
        return None
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if str(c).lower() in lower:
            return lower[str(c).lower()]
    return None


def ensure_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# %% [markdown]
# ## 2. Chargement des fichiers `players.csv`, `games.csv`, `moves.csv`
#
# Le notebook charge les trois cadences séparément puis ajoute une colonne `speed` pour pouvoir comparer les corpus.

# %%
from pathlib import Path

# Monter Google Drive si tu es sur Colab
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception as e:
    print("Pas sur Colab ou Drive déjà monté :", e)

# Mets ici le vrai chemin vers ton dossier
BASE_DIR = Path('/content/drive/MyDrive/Data_Pmind_700-30')

print("BASE_DIR =", BASE_DIR)
print("Existe ?", BASE_DIR.exists())

if BASE_DIR.exists():
    print("Sous-dossiers trouvés :")
    for p in BASE_DIR.iterdir():
        if p.is_dir():
            print("-", p.name)

# %%
def find_cadence_folder(base_dir, speed):
    """Trouve un sous-dossier cadence même si la casse diffère."""
    if not base_dir.exists():
        return None
    direct = base_dir / speed
    if direct.exists():
        return direct
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.lower() == speed.lower():
            return p
    return None


def show_path_help(base_dir):
    print('[INFO] BASE_DIR actuel :', base_dir)
    print('[INFO] Existe ?', base_dir.exists())
    parent = base_dir.parent
    if parent.exists():
        print('[INFO] Quelques dossiers dans', parent, ':')
        for p in list(parent.iterdir())[:30]:
            if p.is_dir():
                print('  -', p.name)
    else:
        print('[INFO] Le dossier parent n’existe pas non plus :', parent)


def load_cadence_data(base_dir=BASE_DIR, cadences=CADENCES):
    players_list, games_list, moves_list = [], [], []

    if not base_dir.exists():
        show_path_help(base_dir)
        raise FileNotFoundError(
            f"BASE_DIR introuvable : {base_dir}. Modifie BASE_DIR pour pointer vers le dossier qui contient bullet/, blitz/, rapid/."
        )

    found_any = False
    for speed in cadences:
        folder = find_cadence_folder(base_dir, speed)
        if folder is None:
            print(f'[WARN] Dossier cadence introuvable pour {speed} dans {base_dir}')
            continue

        p_path, g_path, m_path = folder / 'players.csv', folder / 'games.csv', folder / 'moves.csv'
        if not p_path.exists() or not g_path.exists() or not m_path.exists():
            print(f'[WARN] Fichiers manquants dans {folder}')
            print('  players.csv existe ?', p_path.exists())
            print('  games.csv existe   ?', g_path.exists())
            print('  moves.csv existe   ?', m_path.exists())
            continue

        p = read_csv_safely(p_path)
        g = read_csv_safely(g_path)
        m = read_csv_safely(m_path)
        for df in (p, g, m):
            if len(df):
                df['speed'] = speed.lower()
        players_list.append(p)
        games_list.append(g)
        moves_list.append(m)
        found_any = True
        print(speed, '| players:', p.shape, '| games:', g.shape, '| moves:', m.shape)

    if not found_any:
        show_path_help(base_dir)
        raise FileNotFoundError(
            "Aucun dossier cadence valide trouvé. Vérifie que BASE_DIR contient bien bullet/players.csv, blitz/players.csv, rapid/players.csv, etc."
        )

    players = pd.concat(players_list, ignore_index=True) if players_list else pd.DataFrame()
    games = pd.concat(games_list, ignore_index=True) if games_list else pd.DataFrame()
    moves = pd.concat(moves_list, ignore_index=True) if moves_list else pd.DataFrame()

    if players.empty or games.empty or moves.empty:
        raise ValueError(
            f"Chargement incomplet : players={players.shape}, games={games.shape}, moves={moves.shape}."
        )
    return players, games, moves

players_raw, games_raw, moves_raw = load_cadence_data()
print('\nTOTAL players:', players_raw.shape, 'games:', games_raw.shape, 'moves:', moves_raw.shape)
print('\nColonnes players:', players_raw.columns.tolist()[:25])
print('Colonnes games:', games_raw.columns.tolist()[:25])
print('Colonnes moves:', moves_raw.columns.tolist()[:25])
display(players_raw.head())
display(games_raw.head())
display(moves_raw.head())

# %% [markdown]
# ## 3. Harmonisation minimale des colonnes importantes

# %%
players = players_raw.copy()
games = games_raw.copy()
moves = moves_raw.copy()

# Colonnes principales
player_name_col = find_col(players, ['username', 'target_name', 'name', 'player', 'user'])
elo_col = find_col(players, ['elo', 'rating', 'mean_elo', 'target_elo'])

if player_name_col and player_name_col != 'target_name':
    players = players.rename(columns={player_name_col: 'target_name'})
if elo_col and elo_col != 'elo':
    players = players.rename(columns={elo_col: 'elo'})

players = add_elo_bucket(players, 'elo')

# games : harmoniser game_id, target_name, target_color si possible
for old, new in [('id', 'game_id'), ('gameId', 'game_id')]:
    if old in games.columns and 'game_id' not in games.columns:
        games = games.rename(columns={old: new})

# moves : harmoniser game_id
for old, new in [('id', 'game_id'), ('gameId', 'game_id')]:
    if old in moves.columns and 'game_id' not in moves.columns:
        moves = moves.rename(columns={old: new})

print('players columns:', list(players.columns)[:30])
print('games columns:', list(games.columns)[:30])
print('moves columns:', list(moves.columns)[:30])

# %% [markdown]
# ## 4. Construction de features EDA au niveau coup, partie et joueur
#
# Cette partie reconstruit seulement les features nécessaires à l’EDA. Elle ne remplace pas le pipeline complet de modélisation.

# %%
def prepare_moves_for_eda(moves, games=None):
    m = moves.copy()
    if len(m) == 0:
        return m

    # Filtrer les coups du joueur cible si disponible
    is_target_col = find_col(m, ['is_target_move', 'target_move'])
    if is_target_col:
        mask = m[is_target_col].astype(str).str.lower().isin(['true', '1', 'yes', 't']) | (m[is_target_col] == 1) | (m[is_target_col] == True)
        # Si le filtrage supprime tout, on garde les données et on affiche un avertissement.
        if mask.sum() > 0:
            m = m[mask].copy()
        else:
            print('[WARN] is_target_move existe mais aucun coup cible détecté. Aucun filtrage appliqué.')

    # cp_loss_clean
    cp_col = find_col(m, ['cp_loss', 'loss_cp', 'centipawn_loss'])
    if cp_col:
        m['cp_loss_clean'] = pd.to_numeric(m[cp_col], errors='coerce').abs().clip(upper=600)
    elif 'cp_loss_clean' not in m.columns:
        m['cp_loss_clean'] = np.nan

    # Qualité des coups : si move_quality existe, on l'utilise ; sinon seuils cp_loss_clean.
    q_col = find_col(m, ['move_quality', 'quality'])
    if q_col:
        q = m[q_col].astype(str).str.lower()
        m['is_good'] = q.str.contains('good').astype(int)
        m['is_inaccuracy'] = q.str.contains('inacc').astype(int)
        m['is_mistake'] = q.str.contains('mistake').astype(int)
        m['is_blunder'] = q.str.contains('blunder').astype(int)
        m['is_error'] = ((m['is_inaccuracy'] + m['is_mistake'] + m['is_blunder']) > 0).astype(int)
    else:
        c = m['cp_loss_clean']
        m['is_good'] = (c < 50).astype(int)
        m['is_inaccuracy'] = ((c >= 50) & (c < 100)).astype(int)
        m['is_mistake'] = ((c >= 100) & (c < 200)).astype(int)
        m['is_blunder'] = (c >= 200).astype(int)
        m['is_error'] = (c >= 50).astype(int)

    # Temps
    tsp_col = find_col(m, ['time_spent_cs', 'time_spent', 'time_used_cs'])
    if tsp_col:
        m[tsp_col] = pd.to_numeric(m[tsp_col], errors='coerce').fillna(0)
        if tsp_col.endswith('_cs'):
            m['time_spent_s'] = m[tsp_col] / 100.0
        else:
            m['time_spent_s'] = m[tsp_col]
    else:
        m['time_spent_s'] = np.nan

    clk_col = find_col(m, ['clock_before_cs', 'clock_before', 'clock_s'])
    if clk_col:
        m[clk_col] = pd.to_numeric(m[clk_col], errors='coerce')
        med = m[clk_col].median()
        m[clk_col] = m[clk_col].fillna(med)
        if clk_col.endswith('_cs'):
            m['clock_before_s'] = m[clk_col] / 100.0
        else:
            m['clock_before_s'] = m[clk_col]
    else:
        m['clock_before_s'] = np.nan

    # Phase
    phase_col = find_col(m, ['phase'])
    if phase_col:
        m['phase_clean'] = m[phase_col].astype(str).str.lower()
    else:
        ply_col = find_col(m, ['ply', 'ply_count', 'move_number'])
        if ply_col:
            ply = pd.to_numeric(m[ply_col], errors='coerce')
            m['phase_clean'] = np.select([ply <= 20, ply <= 60], ['opening', 'middlegame'], default='endgame')
        else:
            m['phase_clean'] = 'unknown'

    # Tactique
    for col in ['is_capture', 'is_check', 'is_castle', 'is_promotion']:
        src = find_col(m, [col])
        if src:
            m[col] = m[src].astype(str).str.lower().isin(['true', '1', 'yes']).astype(int) if m[src].dtype == object else pd.to_numeric(m[src], errors='coerce').fillna(0).astype(int)
        else:
            m[col] = 0

    # Fusionner target_name / target_color depuis games si nécessaire
    if games is not None and 'game_id' in m.columns and 'game_id' in games.columns:
        add_cols = ['game_id']
        for c in ['target_name', 'target_color', 'target_elo', 'opening_eco', 'target_score']:
            if c in games.columns:
                add_cols.append(c)
        if len(add_cols) > 1:
            m = m.merge(games[add_cols].drop_duplicates('game_id'), on='game_id', how='left', suffixes=('', '_g'))

    # Contexte positionnel si évaluation moteur disponible
    eval_col = find_col(m, ['eval_white_before', 'eval_before', 'cp_before'])
    if eval_col:
        ev = pd.to_numeric(m[eval_col], errors='coerce').clip(-900, 900)
        if 'target_color' in m.columns:
            color = m['target_color'].astype(str).str.lower()
            m['eval_player_before'] = np.where(color.str.startswith('b'), -ev, ev)
        else:
            m['eval_player_before'] = ev
        m['pos_context'] = np.select(
            [m['eval_player_before'] > 150, m['eval_player_before'] < -150],
            ['winning', 'losing'],
            default='equal'
        )
    else:
        m['eval_player_before'] = np.nan
        m['pos_context'] = 'unknown'

    return m

moves_eda = prepare_moves_for_eda(moves, games)
print('moves_eda:', moves_eda.shape)
display(moves_eda.head())

# %%
def build_game_features(m):
    if len(m) == 0 or 'game_id' not in m.columns:
        return pd.DataFrame()
    keys = ['speed', 'target_name', 'game_id']
    keys = [k for k in keys if k in m.columns]
    grp = m.groupby(keys, dropna=False)
    g = grp.agg(
        n_moves=('cp_loss_clean', 'size'),
        loss_mean=('cp_loss_clean', 'mean'),
        loss_median=('cp_loss_clean', 'median'),
        loss_std=('cp_loss_clean', 'std'),
        loss_p90=('cp_loss_clean', lambda x: np.nanpercentile(x.dropna(), 90) if x.dropna().size else np.nan),
        loss_max=('cp_loss_clean', 'max'),
        rate_good=('is_good', 'mean'),
        rate_inaccuracy=('is_inaccuracy', 'mean'),
        rate_mistake=('is_mistake', 'mean'),
        rate_blunder=('is_blunder', 'mean'),
        rate_error=('is_error', 'mean'),
        n_blunders=('is_blunder', 'sum'),
        n_mistakes=('is_mistake', 'sum'),
        time_mean_s=('time_spent_s', 'mean'),
        time_median_s=('time_spent_s', 'median'),
        time_std_s=('time_spent_s', 'std'),
        clock_mean_s=('clock_before_s', 'mean'),
        rate_capture=('is_capture', 'mean'),
        rate_check=('is_check', 'mean'),
        rate_castle=('is_castle', 'mean'),
        rate_promotion=('is_promotion', 'mean'),
    ).reset_index()

    # Phases : loss and blunder rates by phase
    for ph in ['opening', 'middlegame', 'endgame']:
        sub = m[m['phase_clean'].astype(str).str.contains(ph, na=False)]
        if len(sub):
            gp = sub.groupby(keys, dropna=False).agg(
                **{f'loss_{ph}_mean': ('cp_loss_clean', 'mean'),
                   f'rate_blunder_{ph}': ('is_blunder', 'mean'),
                   f'time_{ph}_mean_s': ('time_spent_s', 'mean')}
            ).reset_index()
            g = g.merge(gp, on=keys, how='left')

    # Position context losses
    for ctx in ['winning', 'equal', 'losing']:
        sub = m[m['pos_context'] == ctx]
        if len(sub):
            gp = sub.groupby(keys, dropna=False).agg(
                **{f'loss_when_{ctx}': ('cp_loss_clean', 'mean'),
                   f'rate_blunder_when_{ctx}': ('is_blunder', 'mean')}
            ).reset_index()
            g = g.merge(gp, on=keys, how='left')

    # Time-pressure features
    # Low clock threshold: bottom 20% within cadence or <=10s fallback
    if 'clock_before_s' in m.columns:
        tmp = m.copy()
        tmp['low_clock'] = False
        for sp, sub in tmp.groupby('speed') if 'speed' in tmp.columns else [('all', tmp)]:
            q = sub['clock_before_s'].quantile(0.2)
            threshold = min(q, 10) if pd.notna(q) else 10
            tmp.loc[sub.index, 'low_clock'] = sub['clock_before_s'] <= threshold
        gp = tmp.groupby(keys, dropna=False).agg(
            blunder_low_time_rate=('is_blunder', lambda x: np.nan),
            good_fast_move_rate=('is_good', lambda x: np.nan)
        ).reset_index().drop(columns=['blunder_low_time_rate','good_fast_move_rate'])
        # Compute explicitly
        res=[]
        for k, sub in tmp.groupby(keys, dropna=False):
            if not isinstance(k, tuple): k=(k,)
            low = sub[sub['low_clock']]
            fast_thr = sub['time_spent_s'].quantile(0.25)
            fast = sub[sub['time_spent_s'] <= fast_thr]
            row = dict(zip(keys,k))
            row['blunder_low_time_rate'] = low['is_blunder'].mean() if len(low) else np.nan
            row['good_fast_move_rate'] = fast['is_good'].mean() if len(fast) else np.nan
            res.append(row)
        gp = pd.DataFrame(res)
        g = g.merge(gp, on=keys, how='left')

    # Drift within game: last third - first third loss
    ply_col = find_col(m, ['ply'])
    if ply_col:
        res=[]
        for k, sub in m.groupby(keys, dropna=False):
            if not isinstance(k, tuple): k=(k,)
            ss = sub.sort_values(ply_col)
            n = len(ss)
            row = dict(zip(keys,k))
            if n >= 6:
                a = ss.iloc[:max(1,n//3)]['cp_loss_clean'].mean()
                b = ss.iloc[-max(1,n//3):]['cp_loss_clean'].mean()
                row['loss_drift_end_vs_start'] = b - a
            else:
                row['loss_drift_end_vs_start'] = np.nan
            res.append(row)
        drift = pd.DataFrame(res)
        g = g.merge(drift, on=keys, how='left')

    return g

game_features = build_game_features(moves_eda)
print('game_features:', game_features.shape)
display(game_features.head())

# %%
def build_player_features(game_features, players, games=None):
    """
    Construit des features player-level à partir de game_features et ajoute la cible Elo.
    La fonction est robuste aux variations de noms de colonnes : elo, target_elo, mean_elo, rating, etc.
    """
    if len(game_features) == 0:
        return pd.DataFrame()

    gf = game_features.copy()
    p = players.copy() if players is not None else pd.DataFrame()
    gms = games.copy() if games is not None else pd.DataFrame()

    # Clés d'agrégation player-level
    keys = ['speed', 'target_name']
    keys = [k for k in keys if k in gf.columns]
    if 'target_name' not in keys:
        print('[WARN] target_name absent de game_features. Agrégation seulement par cadence si speed existe.')

    # Agrégation des variables numériques au niveau joueur
    feature_cols = [c for c in gf.select_dtypes(include=[np.number]).columns if c not in ['game_id']]
    agg = {}
    for c in feature_cols:
        agg[f'{c}_mean'] = (c, 'mean')
        agg[f'{c}_median'] = (c, 'median')
        agg[f'{c}_std'] = (c, 'std')
        agg[f'{c}_iqr'] = (c, iqr)

    pf = gf.groupby(keys, dropna=False).agg(**agg).reset_index() if keys else pd.DataFrame()
    if keys:
        pf['n_games'] = gf.groupby(keys, dropna=False).size().values

    # Ajout win_rate depuis games si disponible
    if games is not None and len(gms) and 'target_score' in gms.columns and 'game_id' in gf.columns and 'game_id' in gms.columns:
        temp = gf[keys + ['game_id']].merge(gms[['game_id','target_score']].drop_duplicates('game_id'), on='game_id', how='left')
        wr = temp.groupby(keys, dropna=False)['target_score'].mean().reset_index(name='win_rate')
        pf = pf.merge(wr, on=keys, how='left')

    # ECO diversity depuis games si disponible
    if games is not None and len(gms) and 'opening_eco' in gms.columns and 'game_id' in gf.columns and 'game_id' in gms.columns:
        temp = gf[keys + ['game_id']].merge(gms[['game_id','opening_eco']].drop_duplicates('game_id'), on='game_id', how='left')
        temp['eco_family'] = temp['opening_eco'].astype(str).str[0]
        eco = temp.groupby(keys, dropna=False)['eco_family'].nunique().reset_index(name='eco_diversity')
        pf = pf.merge(eco, on=keys, how='left')

    # ----- Ajout de la cible Elo -----
    # 1) Source principale : players.csv
    if len(p):
        # Harmoniser nom joueur et Elo dans players si nécessaire
        name_col = first_existing_col(p, ['target_name', 'username', 'name', 'player', 'user'])
        elo_source_col = first_existing_col(p, ['elo', 'target_elo', 'mean_elo', 'rating', 'player_elo'])
        if name_col and name_col != 'target_name':
            p = p.rename(columns={name_col: 'target_name'})
        if elo_source_col and elo_source_col != 'elo':
            p = p.rename(columns={elo_source_col: 'elo'})
        if 'elo' in p.columns:
            p = ensure_numeric(p, 'elo')
            p = add_elo_bucket(p, 'elo')
            target_cols = ['target_name', 'elo']
            if 'speed' in p.columns and 'speed' in pf.columns:
                target_cols = ['speed'] + target_cols
            if 'elo_bucket' in p.columns:
                target_cols.append('elo_bucket')
            target_cols = [c for c in target_cols if c in p.columns]
            base = p[target_cols].drop_duplicates()
            merge_keys = ['target_name']
            if 'speed' in base.columns and 'speed' in pf.columns:
                merge_keys = ['speed', 'target_name']
            if 'target_name' in pf.columns:
                pf = pf.merge(base, on=merge_keys, how='left')

    # 2) Fallback : games.csv si Elo encore absent ou trop manquant
    need_fallback = ('elo' not in pf.columns) or (pf['elo'].isna().all() if 'elo' in pf.columns else True)
    if need_fallback and games is not None and len(gms):
        elo_game_col = first_existing_col(gms, ['target_elo', 'elo', 'white_elo', 'black_elo'])
        name_game_col = first_existing_col(gms, ['target_name', 'username', 'name', 'player'])
        if elo_game_col and name_game_col:
            tmp = gms.copy()
            if name_game_col != 'target_name':
                tmp = tmp.rename(columns={name_game_col: 'target_name'})
            tmp['elo_from_games'] = pd.to_numeric(tmp[elo_game_col], errors='coerce')
            target_cols = ['target_name', 'elo_from_games']
            if 'speed' in tmp.columns and 'speed' in pf.columns:
                target_cols = ['speed'] + target_cols
            base = tmp[target_cols].dropna(subset=['elo_from_games']).groupby([c for c in target_cols if c != 'elo_from_games'], dropna=False)['elo_from_games'].median().reset_index()
            merge_keys = ['target_name']
            if 'speed' in base.columns and 'speed' in pf.columns:
                merge_keys = ['speed', 'target_name']
            if 'target_name' in pf.columns:
                pf = pf.merge(base, on=merge_keys, how='left')
                if 'elo' not in pf.columns:
                    pf['elo'] = pf['elo_from_games']
                else:
                    pf['elo'] = pf['elo'].fillna(pf['elo_from_games'])
                pf = pf.drop(columns=['elo_from_games'])

    if 'elo' in pf.columns:
        pf = ensure_numeric(pf, 'elo')
        pf = add_elo_bucket(pf, 'elo')
    else:
        print('[WARN] Aucune colonne Elo trouvée après fusion. Les analyses liées à la cible ne pourront pas être exécutées.')

    return pf

player_features = build_player_features(game_features, players, games)
print('player_features:', player_features.shape)
print('Colonnes contenant elo :', [c for c in player_features.columns if 'elo' in c.lower()])
if 'elo' in player_features.columns:
    print('Valeurs Elo manquantes :', player_features['elo'].isna().sum())
display(player_features.head())

# Export dataset EDA
player_features.to_csv(OUT_DIR / 'eda_player_features.csv', index=False)
game_features.to_csv(OUT_DIR / 'eda_game_features.csv', index=False)
print('Saved player/game features to', OUT_DIR)

# %%
# Sanity check après construction des features player-level
print('game_features:', game_features.shape)
print('player_features:', player_features.shape)

if game_features.empty:
    print('\n[DEBUG] moves_eda colonnes:', moves_eda.columns.tolist()[:50])
    print('[DEBUG] moves_eda shape:', moves_eda.shape)
    raise ValueError(
        "game_features est vide. Causes probables : moves.csv non chargé, colonne game_id absente, ou filtrage is_target_move trop strict."
    )

if player_features.empty:
    print('\n[DEBUG] game_features colonnes:', game_features.columns.tolist()[:50])
    raise ValueError(
        "player_features est vide. Vérifier que game_features contient au moins speed/target_name et des variables numériques."
    )

print('\nColonnes d’identification dans player_features :')
print([c for c in player_features.columns if any(k in c.lower() for k in ['name', 'user', 'player', 'speed'])])

print('\nColonnes cible / Elo dans player_features :')
print([c for c in player_features.columns if 'elo' in c.lower()])

if 'elo' not in player_features.columns:
    print('\n[DEBUG] Colonnes players:', players.columns.tolist())
    print('[DEBUG] Colonnes games:', games.columns.tolist()[:80])
    print('[DEBUG] Colonnes player_features:', player_features.columns.tolist()[:80])
    raise ValueError(
        "La colonne 'elo' est absente de player_features. Vérifier que players.csv contient une colonne elo/rating/mean_elo et une colonne username/target_name."
    )

print('\nNombre de profils :', len(player_features))
print('Elo non manquants :', player_features['elo'].notna().sum())
cols_preview = [c for c in ['speed','target_name','elo','elo_bucket','n_games'] if c in player_features.columns]
display(player_features[cols_preview].head() if cols_preview else player_features.head())

# %% [markdown]
# # Partie A — Variable cible : `elo`

# %%
dfp = player_features.copy()

if 'elo' not in dfp.columns:
    raise ValueError("Colonne 'elo' absente de player_features. La fusion de la cible a échoué.")

dfp['elo'] = pd.to_numeric(dfp['elo'], errors='coerce')
dfp = dfp.dropna(subset=['elo']).copy()

if 'elo_bucket' not in dfp.columns:
    dfp = add_elo_bucket(dfp, 'elo')

print('Nombre de profils joueur-cadence:', len(dfp))
print('Elo min/max/std/skew:', dfp['elo'].min(), dfp['elo'].max(), dfp['elo'].std(), dfp['elo'].skew())

q1, q3 = dfp['elo'].quantile([0.25, 0.75])
iqr_elo = q3 - q1
lower, upper = q1 - 1.5 * iqr_elo, q3 + 1.5 * iqr_elo
outliers = dfp[(dfp['elo'] < lower) | (dfp['elo'] > upper)]
print('Bornes IQR outliers:', lower, upper)
print('Nombre outliers Elo:', len(outliers))

target_summary = pd.DataFrame({
    'metric': ['count', 'min', 'q1', 'median', 'mean', 'q3', 'max', 'std', 'skewness', 'iqr_outliers'],
    'value': [len(dfp), dfp['elo'].min(), q1, dfp['elo'].median(), dfp['elo'].mean(), q3, dfp['elo'].max(), dfp['elo'].std(), dfp['elo'].skew(), len(outliers)]
})
target_summary.to_csv(OUT_DIR / 'target_elo_summary.csv', index=False)
display(target_summary)

# %%
plt.figure(figsize=(8,4.5))
plt.hist(dfp['elo'], bins=30, density=True, alpha=0.6, edgecolor='black')
try:
    dfp['elo'].plot(kind='kde')
except Exception:
    pass
plt.title('Distribution de la variable cible Elo')
plt.xlabel('Elo')
plt.ylabel('Densité')
savefig('01_target_elo_distribution.png')

# Distribution par cadence
if 'speed' in dfp.columns:
    plt.figure(figsize=(8,4.5))
    for speed, sub in dfp.groupby('speed'):
        plt.hist(sub['elo'], bins=25, alpha=0.4, density=True, label=str(speed))
    plt.title('Distribution des Elo par cadence')
    plt.xlabel('Elo')
    plt.ylabel('Densité')
    plt.legend()
    savefig('02_elo_distribution_by_cadence.png')

# Buckets par cadence
if 'speed' in dfp.columns and 'elo_bucket' in dfp.columns:
    bucket_counts = dfp.groupby(['speed','elo_bucket'], observed=False).size().reset_index(name='n')
    bucket_counts.to_csv(OUT_DIR / 'elo_bucket_counts_by_cadence.csv', index=False)
    display(bucket_counts)
    pivot = bucket_counts.pivot(index='elo_bucket', columns='speed', values='n').fillna(0)
    ax = pivot.plot(kind='bar', figsize=(9,4.5))
    plt.title('Nombre de profils joueur-cadence par bucket Elo')
    plt.xlabel('Bucket Elo')
    plt.ylabel('Nombre de profils')
    plt.xticks(rotation=30, ha='right')
    savefig('03_elo_bucket_counts_by_cadence.png')

# %% [markdown]
# **Commentaire rapport.**  
# Cette analyse sert à justifier le choix de la régression : `elo` est une variable numérique continue. Les buckets restent utiles pour les analyses stratifiées et la visualisation, mais `elo_bucket` ne doit pas être utilisé comme feature explicative car il est dérivé directement de `elo`.

# %% [markdown]
# # Partie B — Analyse univariée et bivariée des features joueur

# %%
# Missing values sur player_features
missing = dfp.isna().mean().sort_values(ascending=False).reset_index()
missing.columns = ['feature', 'missing_rate']
missing.to_csv(OUT_DIR / 'missing_values_player_features.csv', index=False)
display(missing.head(25))

# Distributions rapides de quelques features clés
candidate_features = [
    'loss_mean_mean', 'loss_median_mean', 'loss_p90_mean', 'loss_max_mean',
    'rate_blunder_mean', 'rate_error_mean', 'rate_mistake_mean',
    'time_mean_s_mean', 'clock_mean_s_mean', 'win_rate', 'eco_diversity',
    'between_game_loss_variance', 'loss_mean_iqr'
]
features_to_plot = [c for c in candidate_features if c in dfp.columns]

for c in features_to_plot[:12]:
    plt.figure(figsize=(7,4))
    plt.hist(dfp[c].dropna(), bins=30, edgecolor='black')
    plt.title(f'Distribution de {c}')
    plt.xlabel(c)
    plt.ylabel('Nombre de profils')
    savefig(f'univariate_{c}.png')

# %%
# Scatter bivarié vs Elo pour les variables importantes disponibles
scatter_candidates = ['loss_mean_mean', 'rate_blunder_mean', 'rate_error_mean', 'win_rate', 'time_mean_s_mean']
for c in [x for x in scatter_candidates if x in dfp.columns]:
    tmp = dfp[[c, 'elo', 'speed']].dropna()
    if len(tmp) > 3000:
        tmp = tmp.sample(3000, random_state=RANDOM_STATE)
    plt.figure(figsize=(7,4.5))
    for sp, sub in tmp.groupby('speed'):
        plt.scatter(sub[c], sub['elo'], s=12, alpha=0.45, label=str(sp))
    plt.title(f'{c} vs Elo')
    plt.xlabel(c)
    plt.ylabel('Elo')
    plt.legend()
    savefig(f'scatter_{c}_vs_elo.png')

# Corrélations Pearson/Spearman des features numériques avec Elo
exclude = ['elo']
num_cols = numeric_cols(dfp, exclude=exclude)
pearson = safe_corr(dfp, num_cols, 'elo', method='pearson')
spearman = safe_corr(dfp, num_cols, 'elo', method='spearman')
pearson.to_csv(OUT_DIR / 'top_pearson_corr_with_elo.csv', index=False)
spearman.to_csv(OUT_DIR / 'top_spearman_corr_with_elo.csv', index=False)
display(spearman.head(20))

plt.figure(figsize=(8,5))
top = spearman.head(20).iloc[::-1]
plt.barh(top['feature'], top['spearman_corr_with_elo'])
plt.title('Top corrélations Spearman avec Elo')
plt.xlabel('Corrélation Spearman')
savefig('04_top_spearman_correlations_with_elo.png')

# %%
# Boxplots stratifiés par bucket Elo pour les features clés
box_features = [c for c in ['loss_mean_mean', 'rate_blunder_mean', 'rate_error_mean', 'win_rate', 'time_mean_s_mean'] if c in dfp.columns]
if 'elo_bucket' in dfp.columns:
    for c in box_features:
        robust_boxplot(dfp, 'elo_bucket', c, f'{c} par bucket Elo', c, f'box_{c}_by_elo_bucket.png')

    # Tests statistiques par bucket : ANOVA et/ou Kruskal-Wallis
    rows = []
    for c in box_features:
        groups = [g[c].dropna().values for _, g in dfp.groupby('elo_bucket', observed=False)]
        groups = [g for g in groups if len(g) > 2]
        row = {'feature': c}
        if SCIPY_AVAILABLE and len(groups) >= 2:
            try:
                row['anova_pvalue'] = stats.f_oneway(*groups).pvalue
            except Exception:
                row['anova_pvalue'] = np.nan
            try:
                row['kruskal_pvalue'] = stats.kruskal(*groups).pvalue
            except Exception:
                row['kruskal_pvalue'] = np.nan
        rows.append(row)
    feature_bucket_tests = pd.DataFrame(rows)
    feature_bucket_tests.to_csv(OUT_DIR / 'feature_bucket_tests.csv', index=False)
    display(feature_bucket_tests)
else:
    print("elo_bucket absent : boxplots/tests par bucket non exécutés.")

# %% [markdown]
# # Partie C — Analyse `moves.csv` et `games.csv`

# %%
# Distribution des pertes en centipions
sample_moves = moves_eda.dropna(subset=['cp_loss_clean']).copy()
if len(sample_moves) > 500_000:
    sample_moves = sample_moves.sample(500_000, random_state=RANDOM_STATE)

plt.figure(figsize=(8,4.5))
plt.hist(sample_moves['cp_loss_clean'], bins=60, edgecolor='black')
plt.title('Distribution de cp_loss_clean')
plt.xlabel('Perte en centipions nettoyée')
plt.ylabel('Nombre de coups')
savefig('05_cp_loss_clean_distribution.png')

# Répartition qualité coups
quality_counts = moves_eda[['is_good','is_inaccuracy','is_mistake','is_blunder']].sum().reset_index()
quality_counts.columns = ['quality', 'count']
quality_counts.to_csv(OUT_DIR / 'move_quality_counts.csv', index=False)
display(quality_counts)
plt.figure(figsize=(7,4))
plt.bar(quality_counts['quality'], quality_counts['count'])
plt.title('Répartition des catégories de coups')
plt.ylabel('Nombre de coups')
plt.xticks(rotation=20)
savefig('06_move_quality_counts.png')

# Qualité par phase
phase_quality = moves_eda.groupby('phase_clean')[['cp_loss_clean','is_error','is_blunder']].mean().reset_index()
phase_quality.to_csv(OUT_DIR / 'move_quality_by_phase.csv', index=False)
display(phase_quality)
phase_quality.plot(x='phase_clean', y=['cp_loss_clean','is_error','is_blunder'], kind='bar', figsize=(8,4.5))
plt.title('Qualité des coups par phase')
plt.xlabel('Phase')
plt.ylabel('Moyenne / taux')
plt.xticks(rotation=0)
savefig('07_move_quality_by_phase.png')

# %%
# Temps vs cp_loss
if 'time_spent_s' in moves_eda.columns:
    tmp = moves_eda[['time_spent_s','cp_loss_clean','speed','clock_before_s','is_blunder','is_good']].dropna(subset=['time_spent_s','cp_loss_clean'])
    if len(tmp) > 150_000:
        tmp = tmp.sample(150_000, random_state=RANDOM_STATE)
    plt.figure(figsize=(7,4.5))
    plt.scatter(tmp['time_spent_s'], tmp['cp_loss_clean'], s=4, alpha=0.15)
    plt.xlim(0, tmp['time_spent_s'].quantile(0.99))
    plt.title('Temps utilisé vs perte en centipions')
    plt.xlabel('Temps utilisé (s)')
    plt.ylabel('cp_loss_clean')
    savefig('08_time_spent_vs_cp_loss.png')

    # Zéro time / premove-like par cadence
    zero_time = moves_eda.assign(zero_time=(moves_eda['time_spent_s'] <= 0)).groupby('speed')['zero_time'].mean().reset_index(name='zero_time_rate')
    zero_time.to_csv(OUT_DIR / 'zero_time_rate_by_cadence.csv', index=False)
    display(zero_time)
    plt.figure(figsize=(6,4))
    plt.bar(zero_time['speed'], zero_time['zero_time_rate'])
    plt.title('Taux de coups avec temps nul par cadence')
    plt.xlabel('Cadence')
    plt.ylabel('Taux time_spent_s = 0')
    savefig('09_zero_time_rate_by_cadence.png')

# Clock before vs qualité
if 'clock_before_s' in moves_eda.columns:
    bins = pd.qcut(moves_eda['clock_before_s'].rank(method='first'), 5, labels=['très bas','bas','moyen','haut','très haut'])
    clock_quality = moves_eda.assign(clock_bin=bins).groupby(['speed','clock_bin'])[['is_blunder','is_error','cp_loss_clean']].mean().reset_index()
    clock_quality.to_csv(OUT_DIR / 'clock_quality_by_cadence.csv', index=False)
    display(clock_quality.head())

# %%
# Contexte parties : résultats, couleurs, elo diff
if len(games):
    # speed proportions
    speed_counts = games.groupby('speed').size().reset_index(name='n_games')
    speed_counts['prop'] = speed_counts['n_games'] / speed_counts['n_games'].sum()
    speed_counts.to_csv(OUT_DIR / 'game_speed_proportions.csv', index=False)
    display(speed_counts)

    # win rate white/black si colonnes disponibles
    if 'target_color' in games.columns and 'target_score' in games.columns:
        color_wr = games.groupby(['speed','target_color'])['target_score'].mean().reset_index(name='score_mean')
        color_wr.to_csv(OUT_DIR / 'target_score_by_color_speed.csv', index=False)
        display(color_wr)

    # elo_diff vs result si possible
    elo_diff_col = find_col(games, ['target_elo_diff', 'elo_diff'])
    if elo_diff_col and 'target_score' in games.columns:
        tmp = games[[elo_diff_col,'target_score','speed']].dropna()
        if len(tmp) > 5000:
            tmp = tmp.sample(5000, random_state=RANDOM_STATE)
        plt.figure(figsize=(7,4.5))
        plt.scatter(tmp[elo_diff_col], tmp['target_score'], s=8, alpha=0.25)
        plt.title('Différence Elo vs résultat du joueur cible')
        plt.xlabel(elo_diff_col)
        plt.ylabel('target_score')
        savefig('10_elo_diff_vs_target_score.png')

    # ouvertures vs niveau : eco family moyen elo
    if 'opening_eco' in games.columns and 'target_elo' in games.columns:
        games_tmp = games.copy()
        games_tmp['eco_family'] = games_tmp['opening_eco'].astype(str).str[0]
        eco_elo = games_tmp.groupby('eco_family')['target_elo'].agg(['count','mean','median']).reset_index().sort_values('count', ascending=False)
        eco_elo.to_csv(OUT_DIR / 'opening_eco_family_elo_summary.csv', index=False)
        display(eco_elo.head(10))

# %% [markdown]
# # Partie D — Séparation par cadence : tests et justification

# %%
# Boxplots cadence : loss, blunder, time
cadence_features = [c for c in ['loss_mean_mean','rate_blunder_mean','rate_error_mean','time_mean_s_mean','n_moves_mean'] if c in dfp.columns]
if 'speed' in dfp.columns:
    for c in cadence_features:
        robust_boxplot(dfp, 'speed', c, f'{c} par cadence', c, f'cadence_box_{c}.png')

    # Tests statistiques entre cadences
    rows=[]
    for c in cadence_features:
        groups = [g[c].dropna().values for _, g in dfp.groupby('speed')]
        groups = [g for g in groups if len(g)>2]
        row={'feature': c}
        if SCIPY_AVAILABLE and len(groups)>=2:
            try: row['anova_pvalue'] = stats.f_oneway(*groups).pvalue
            except Exception: row['anova_pvalue'] = np.nan
            try: row['kruskal_pvalue'] = stats.kruskal(*groups).pvalue
            except Exception: row['kruskal_pvalue'] = np.nan
        rows.append(row)
    cadence_tests = pd.DataFrame(rows)
    cadence_tests.to_csv(OUT_DIR / 'cadence_difference_tests.csv', index=False)
    display(cadence_tests)
else:
    print('Colonne speed absente : comparaison par cadence non exécutée.')

# %% [markdown]
# # Partie E — Features de qualité, contexte de l’erreur, phases, stabilité
# %%
# cp_loss par phase et cadence
if len(moves_eda):
    phase_speed = moves_eda.groupby(['speed','phase_clean'])['cp_loss_clean'].mean().reset_index()
    phase_speed.to_csv(OUT_DIR / 'cp_loss_by_phase_and_cadence.csv', index=False)
    display(phase_speed)
    piv = phase_speed.pivot(index='phase_clean', columns='speed', values='cp_loss_clean')
    ax = piv.plot(kind='bar', figsize=(8,4.5))
    plt.title('Perte moyenne par phase et cadence')
    plt.xlabel('Phase')
    plt.ylabel('cp_loss_clean moyen')
    plt.xticks(rotation=0)
    savefig('11_cp_loss_by_phase_and_cadence.png')

# Contexte de l'erreur vs Elo : corrélations si features dispo
context_features = [c for c in dfp.columns if any(k in c for k in ['loss_when_winning','loss_when_equal','loss_when_losing','rate_blunder_when_equal'])]
context_corr = safe_corr(dfp, context_features, 'elo', method='spearman')
context_corr.to_csv(OUT_DIR / 'context_error_corr_with_elo.csv', index=False)
display(context_corr.head(15))

# Stabilité inter-parties : corrélation avec Elo
stability_candidates = [c for c in dfp.columns if any(k in c for k in ['_std','_iqr','variance','_cv','prop_games_with'])]
stability_corr = safe_corr(dfp, stability_candidates, 'elo', method='spearman')
stability_corr.to_csv(OUT_DIR / 'stability_corr_with_elo.csv', index=False)
display(stability_corr.head(20))

# loss_mean vs dispersion
x_col = 'loss_mean_mean'
y_col = 'loss_mean_iqr' if 'loss_mean_iqr' in dfp.columns else ('loss_mean_std' if 'loss_mean_std' in dfp.columns else None)
if x_col in dfp.columns and y_col:
    tmp = dfp[[x_col,y_col,'elo','speed']].dropna()
    plt.figure(figsize=(7,4.5))
    sc = plt.scatter(tmp[x_col], tmp[y_col], c=tmp['elo'], s=18, alpha=0.65)
    plt.colorbar(sc, label='Elo')
    plt.title('Qualité moyenne vs dispersion entre parties')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    savefig('12_loss_mean_vs_dispersion.png')

# %% [markdown]
# # Partie F — Multicolinéarité, redondance et leakage

# %%
# Préparer matrice numérique en excluant leakage / identifiants
leakage_cols = {'elo', 'elo_bucket', 'target_name', 'username', 'game_id'}
num = dfp.select_dtypes(include=[np.number]).replace([np.inf,-np.inf], np.nan)
num = num[[c for c in num.columns if c not in leakage_cols]]
# Supprimer colonnes trop manquantes / constantes
num = num.loc[:, num.isna().mean() < 0.4]
num = num.loc[:, num.nunique(dropna=True) > 1]
num_imp = num.fillna(num.median(numeric_only=True))

# Top features corrélées avec elo pour heatmap lisible
top_features = spearman.head(25)['feature'].tolist()
top_features = [c for c in top_features if c in num_imp.columns]
heat_cols = top_features[:20]
if len(heat_cols) >= 2:
    corr_mat = num_imp[heat_cols].corr(method='spearman')
    corr_mat.to_csv(OUT_DIR / 'feature_correlation_heatmap_matrix.csv')
    plt.figure(figsize=(10,8))
    im = plt.imshow(corr_mat.values, aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, label='Corrélation Spearman')
    plt.xticks(range(len(heat_cols)), heat_cols, rotation=90)
    plt.yticks(range(len(heat_cols)), heat_cols)
    plt.title('Heatmap des corrélations entre features principales')
    savefig('13_feature_correlation_heatmap.png')

# Paires redondantes corr > 0.90
corr_all = num_imp.corr(method='spearman').abs()
upper = corr_all.where(np.triu(np.ones(corr_all.shape), k=1).astype(bool))
pairs = upper.stack().reset_index()
pairs.columns = ['feature_1','feature_2','abs_spearman_corr']
pairs = pairs[pairs['abs_spearman_corr'] >= 0.90].sort_values('abs_spearman_corr', ascending=False)
pairs.to_csv(OUT_DIR / 'highly_correlated_feature_pairs.csv', index=False)
display(pairs.head(30))

# VIF sur sous-ensemble de top features pour éviter explosion
if STATSMODELS_AVAILABLE and len(top_features) >= 2:
    vif_cols = top_features[:15]
    X = num_imp[vif_cols].copy()
    # Standardisation
    X = (X - X.mean()) / X.std(ddof=0).replace(0, np.nan)
    X = X.dropna(axis=1)
    vif_rows=[]
    for i, c in enumerate(X.columns):
        try:
            vif_rows.append({'feature': c, 'VIF': variance_inflation_factor(X.values, i)})
        except Exception:
            vif_rows.append({'feature': c, 'VIF': np.nan})
    vif_df = pd.DataFrame(vif_rows).sort_values('VIF', ascending=False)
else:
    vif_df = pd.DataFrame({'feature': [], 'VIF': []})

vif_df.to_csv(OUT_DIR / 'vif_top_features.csv', index=False)
display(vif_df)

print('⚠ Leakage : elo_bucket est dérivé de elo. Il doit être exclu des features de prédiction.')

# %% [markdown]
# # Partie G — Few-shot EDA : effet du nombre de parties

# %%
# Analyse de convergence des features selon n_games.
# On compare les features calculées sur n parties à la valeur de référence calculée sur toutes les parties disponibles.

fewshot_game_cols = [c for c in ['loss_mean','loss_median','loss_p90','rate_blunder','rate_error','time_mean_s','good_fast_move_rate','blunder_low_time_rate'] if c in game_features.columns]
keys = ['speed', 'target_name']
keys = [k for k in keys if k in game_features.columns]
N_VALUES = [3, 5, 10, 20, 30]
REPS = 10

if len(fewshot_game_cols) == 0 or len(keys) == 0:
    print('Few-shot EDA non exécutée : colonnes game-level ou clés joueur absentes.')
    fewshot = pd.DataFrame()
else:
    # Référence full par joueur
    full_ref = game_features.groupby(keys, dropna=False)[fewshot_game_cols].mean().reset_index()
    full_ref = full_ref.rename(columns={c: f'{c}_full' for c in fewshot_game_cols})

    rows=[]
    grouped = list(game_features.groupby(keys, dropna=False))
    for n in N_VALUES:
        for rep in range(REPS):
            sampled_parts=[]
            for k, sub in grouped:
                if len(sub) < n:
                    continue
                sampled_parts.append(sub.sample(n=n, replace=False, random_state=RANDOM_STATE + rep * 100 + n))
            if not sampled_parts:
                continue
            sampled = pd.concat(sampled_parts, ignore_index=True)
            agg = sampled.groupby(keys, dropna=False)[fewshot_game_cols].mean().reset_index()
            merged = agg.merge(full_ref, on=keys, how='inner')
            for c in fewshot_game_cols:
                err = (merged[c] - merged[f'{c}_full']).abs()
                rows.append({'n_games': n, 'rep': rep, 'feature': c, 'mae_vs_full': err.mean(), 'median_abs_error_vs_full': err.median(), 'n_players': len(merged)})

    fewshot = pd.DataFrame(rows)
    fewshot.to_csv(OUT_DIR / 'fewshot_feature_stability.csv', index=False)
    display(fewshot.head())

    # Courbe de convergence pour quelques features
    if len(fewshot):
        plt.figure(figsize=(9,5))
        for c in fewshot_game_cols[:6]:
            sub = fewshot[fewshot['feature'] == c].groupby('n_games')['mae_vs_full'].mean().reset_index()
            plt.plot(sub['n_games'], sub['mae_vs_full'], marker='o', label=c)
        plt.title('Stabilité des features selon le nombre de parties')
        plt.xlabel('Nombre de parties échantillonnées')
        plt.ylabel('MAE vs valeur calculée sur toutes les parties')
        plt.legend()
        savefig('14_fewshot_feature_stability.png')

# %%
# Comparaison moyenne vs médiane pour la stabilité de loss_mean selon n_games
if 'loss_mean' in game_features.columns and len(keys) > 0:
    rows=[]
    full_mean = game_features.groupby(keys, dropna=False)['loss_mean'].mean().reset_index(name='full_mean')
    full_median = game_features.groupby(keys, dropna=False)['loss_mean'].median().reset_index(name='full_median')
    ref = full_mean.merge(full_median, on=keys)
    grouped = list(game_features.groupby(keys, dropna=False))
    for n in N_VALUES:
        for rep in range(REPS):
            sampled_parts=[]
            for k, sub in grouped:
                if len(sub) < n:
                    continue
                sampled_parts.append(sub.sample(n=n, replace=False, random_state=1000 + rep * 100 + n))
            if not sampled_parts:
                continue
            sampled = pd.concat(sampled_parts, ignore_index=True)
            agg = sampled.groupby(keys, dropna=False)['loss_mean'].agg(sample_mean='mean', sample_median='median').reset_index()
            merged = agg.merge(ref, on=keys, how='inner')
            rows.append({
                'n_games': n,
                'rep': rep,
                'mean_mae_vs_full_mean': (merged['sample_mean'] - merged['full_mean']).abs().mean(),
                'median_mae_vs_full_median': (merged['sample_median'] - merged['full_median']).abs().mean(),
                'n_players': len(merged)
            })
    medmean = pd.DataFrame(rows)
    medmean.to_csv(OUT_DIR / 'fewshot_mean_vs_median_loss_stability.csv', index=False)
    if len(medmean):
        display(medmean.groupby('n_games')[['mean_mae_vs_full_mean','median_mae_vs_full_median']].mean())
        plt.figure(figsize=(7,4.5))
        plotdf = medmean.groupby('n_games')[['mean_mae_vs_full_mean','median_mae_vs_full_median']].mean().reset_index()
        plt.plot(plotdf['n_games'], plotdf['mean_mae_vs_full_mean'], marker='o', label='Moyenne')
        plt.plot(plotdf['n_games'], plotdf['median_mae_vs_full_median'], marker='o', label='Médiane')
        plt.title('Few-shot : stabilité moyenne vs médiane')
        plt.xlabel('Nombre de parties')
        plt.ylabel('Erreur absolue vs référence full')
        plt.legend()
        savefig('15_fewshot_mean_vs_median.png')
else:
    print('Comparaison moyenne/médiane non exécutée : loss_mean ou clés joueur absentes.')

# %% [markdown]
# # Partie H — Tableau de synthèse pour le rapport

# %%
# Création d'un petit rapport CSV avec les fichiers produits
outputs = sorted([p.name for p in OUT_DIR.glob('*')])
out_df = pd.DataFrame({'output_file': outputs})
out_df.to_csv(OUT_DIR / 'eda_outputs_index.csv', index=False)
display(out_df)
print('Tous les résultats sont dans :', OUT_DIR)

# %% [markdown]
# ## Figures recommandées dans le rapport
#
# À insérer en priorité :
#
# 1. `01_target_elo_distribution.png` — distribution de la variable cible.
# 2. `03_elo_bucket_counts_by_cadence.png` — équilibre des buckets par cadence.
# 3. `cadence_box_time_mean_s_mean.png` — différences de gestion du temps selon la cadence.
# 4. `cadence_box_loss_mean_mean.png` — différences de qualité selon la cadence.
# 5. `05_cp_loss_clean_distribution.png` — distribution des pertes en centipions.
# 6. `box_rate_blunder_mean_by_elo_bucket.png` ou `box_rate_error_mean_by_elo_bucket.png` — erreurs selon Elo.
# 7. `11_cp_loss_by_phase_and_cadence.png` — qualité par phase de jeu.
# 8. `12_loss_mean_vs_dispersion.png` — moyenne vs stabilité.
# 9. `04_top_spearman_correlations_with_elo.png` — variables liées au Elo.
# 10. `14_fewshot_feature_stability.png` — effet du nombre de parties.
#
# À utiliser en annexe si besoin : heatmap, VIF, paires de features redondantes, distributions univariées.
