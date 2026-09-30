# PMind experiment export
#
# Clean text export of the original Jupyter/Colab experiment.
# Cell boundaries are preserved with VS Code/Jupytext-style markers.
# Notebook outputs are intentionally omitted; verified metrics are documented in docs/RESULTS.md.

# %% [markdown]
# # Chess Elo stacking from CSV - 1000x100
#
# This notebook improves the base player-level CSV by stacking three complementary models:
# - `CatBoost`
# - `XGBoost (tuned)`
# - `Ridge`
#
# The key idea is to train a meta-model on out-of-fold predictions from the train split only, then evaluate on the held-out test split.
# This avoids tuning the blend directly on the test set.

# %%
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression, HuberRegressor, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
from catboost import CatBoostRegressor

sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110
pd.set_option('display.max_columns', 200)
pd.set_option('display.float_format', '{:.3f}'.format)
RANDOM_STATE = 42

# %%
IN_COLAB = Path('/content').exists()
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')

candidates = [
    Path('data/processed/player_level_features_v4_1000x100.csv'),
    Path('/content/drive/MyDrive/projetMind/data/processed/player_level_features_v4_1000x100.csv'),
    Path('/content/drive/MyDrive/projetMind/data/processed/player_level_features_v4_1000x100.csv'),
]

def pick_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

PLAYER_CSV = pick_first_existing(candidates)
print('PLAYER_CSV =', PLAYER_CSV)
if PLAYER_CSV is None:
    raise FileNotFoundError('player_level_features_v4_1000x100.csv introuvable')

# %%
player_df = pd.read_csv(PLAYER_CSV)
print('player_df shape:', player_df.shape)
player_df.head(2)

# %%
LEAKAGE_COLS = [
    'target_name',
    'elo',
    'elo_bucket',
    'dominant_speed',
]

numeric_cols = player_df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in LEAKAGE_COLS]
feature_cols = [c for c in feature_cols if ('elo' not in c.lower() and 'bucket' not in c.lower())]

X = player_df[feature_cols].copy()
y = player_df['elo'].values
stratify_labels = player_df['elo_bucket'] if 'elo_bucket' in player_df.columns else None

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=stratify_labels,
)

print(f'Features used : {len(feature_cols)}')
print(f'Train players : {len(X_train)}')
print(f'Test players  : {len(X_test)}')

# %%
def score_row(name, y_true, y_pred):
    return {
        'Model': name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
    }

base_models = {
    'CatBoost': Pipeline([
        ('i', SimpleImputer(strategy='median')),
        ('m', CatBoostRegressor(
            iterations=500,
            depth=5,
            learning_rate=0.05,
            l2_leaf_reg=3.0,
            random_seed=RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
        )),
    ]),
    'XGBoost (tuned)': Pipeline([
        ('i', SimpleImputer(strategy='median')),
        ('m', xgb.XGBRegressor(
            n_estimators=800,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=1,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )),
    ]),
    'Ridge': Pipeline([
        ('i', SimpleImputer(strategy='median')),
        ('s', StandardScaler()),
        ('m', Ridge(alpha=10.0)),
    ]),
}

# %% [markdown]
# ## Base predictions
#
# First we fit the three base models on the same train/test split.

# %%
base_results = []
base_test_pred = {}
for name, model in base_models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    base_test_pred[name] = pred
    row = score_row(name, y_test, pred)
    base_results.append(row)
    print(f"{name:18s} MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  R2={row['R2']:.3f}")

base_results_df = pd.DataFrame(base_results).sort_values('MAE').reset_index(drop=True)
base_results_df

# %% [markdown]
# ## 6c. Analyse des Features Importantes
#
# On verifie si `CatBoost` et `XGBoost (tuned)` confirment les memes blocs de variables importantes avant la phase de stacking.

# %%
# Feature importance analysis for the two tree models used in stacking
cat_model = base_models['CatBoost'].named_steps['m']
xgb_model = base_models['XGBoost (tuned)'].named_steps['m']

imp_cat = pd.Series(cat_model.feature_importances_, index=feature_cols, name='CatBoost')
imp_xgb = pd.Series(xgb_model.feature_importances_, index=feature_cols, name='XGBoost')

# Normalize for easier comparison
imp_cat = 100 * imp_cat / imp_cat.sum()
imp_xgb = 100 * imp_xgb / imp_xgb.sum()

imp_compare = pd.concat([imp_cat, imp_xgb], axis=1).fillna(0)
imp_compare['mean_importance'] = imp_compare[['CatBoost', 'XGBoost']].mean(axis=1)


def tag_bloc(name):
    if any(k in name for k in ['time', 'clock', 'low_time', 'fast_move', 'pressure']):
        return 'Temps'
    if any(k in name for k in ['loss', 'rate_good', 'rate_error', 'rate_blunder',
                               'rate_mistake', 'rate_inaccuracy', 'n_blunders',
                               'n_mistakes', 'rate_loss']):
        return 'Qualite'
    if any(k in name for k in ['when_', 'opening', 'middlegame', 'endgame',
                               'winning', 'equal', 'losing']):
        return 'Contexte'
    if any(k in name for k in ['variance', 'stability', 'drift', 'gap',
                               'drop', 'cv', 'prop_games']):
        return 'Stabilite'
    if any(k in name for k in ['capture', 'check', 'castle', 'promotion',
                               'eco', 'speed', 'n_moves', 'ply']):
        return 'Style'
    return 'Autre'

imp_compare['bloc'] = imp_compare.index.map(tag_bloc)

top20 = imp_compare.sort_values('mean_importance', ascending=False).head(20).sort_values('mean_importance')
bloc_compare = imp_compare.groupby('bloc')[['CatBoost', 'XGBoost']].sum().sort_values('CatBoost', ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(17, 8))

# Top features by average importance
axes[0].barh(top20.index, top20['CatBoost'], color='#e74c3c', alpha=0.8, label='CatBoost')
axes[0].barh(top20.index, top20['XGBoost'], color='#e67e22', alpha=0.55, label='XGBoost')
axes[0].set_title('Top 20 features - CatBoost vs XGBoost')
axes[0].set_xlabel('Normalized importance (%)')
axes[0].legend()

# Bloc-level comparison
x = np.arange(len(bloc_compare.index))
width = 0.38
axes[1].bar(x - width/2, bloc_compare['CatBoost'], width, color='#e74c3c', alpha=0.85, label='CatBoost')
axes[1].bar(x + width/2, bloc_compare['XGBoost'], width, color='#e67e22', alpha=0.85, label='XGBoost')
axes[1].set_xticks(x)
axes[1].set_xticklabels(bloc_compare.index, rotation=20)
axes[1].set_title('Importance totale par bloc')
axes[1].set_ylabel('Normalized importance (%)')
axes[1].legend()

plt.tight_layout()
plt.show()

# Text summaries
top10_cat = set(imp_compare.sort_values('CatBoost', ascending=False).head(10).index)
top10_xgb = set(imp_compare.sort_values('XGBoost', ascending=False).head(10).index)
overlap = sorted(top10_cat & top10_xgb)

print('=== Importance par bloc ===')
print(bloc_compare.round(2).to_string())

print('\n=== Top 15 features moyennees ===')
print(
    imp_compare.sort_values('mean_importance', ascending=False)
    .head(15)[['CatBoost', 'XGBoost', 'mean_importance', 'bloc']]
    .round(3)
    .to_string()
)

print('\n=== Overlap top 10 CatBoost / XGBoost ===')
print(f'{len(overlap)}/10 features en commun')
for feat in overlap:
    print(f'[{tag_bloc(feat):10s}] {feat}')

imp_compare.sort_values('mean_importance', ascending=False).to_csv(
    'results/tables/feature_importances_stacking_base_1000x100.csv'
)
print('\nSaved feature_importances_stacking_base_1000x100.csv')

# %% [markdown]
# ## OOF blend and stacking
#
# We build train meta-features with out-of-fold predictions only.
# Then we test:
# - an OOF-optimized convex blend
# - several light meta-models

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
model_order = ['CatBoost', 'XGBoost (tuned)', 'Ridge']

oof_train = []
test_pred = []
for name in model_order:
    model = base_models[name]
    oof = cross_val_predict(model, X_train, y_train, cv=cv, method='predict', n_jobs=1)
    oof_train.append(oof)
    model.fit(X_train, y_train)
    test_pred.append(model.predict(X_test))
    print(f'{name:18s} OOF train MAE={mean_absolute_error(y_train, oof):.3f}')

X_meta_train = np.vstack(oof_train).T
X_meta_test = np.vstack(test_pred).T
meta_feature_names = ['pred_catboost', 'pred_xgb', 'pred_ridge']

print('Meta train shape:', X_meta_train.shape)
print('Meta test shape :', X_meta_test.shape)

# %%
# OOF-optimized convex blend
best_blend = None
for w_cat in np.arange(0, 1.0001, 0.02):
    for w_xgb in np.arange(0, 1.0001, 0.02):
        w_ridge = 1 - w_cat - w_xgb
        if w_ridge < -1e-9 or w_ridge > 1:
            continue
        pred_train = (
            w_cat * X_meta_train[:, 0] +
            w_xgb * X_meta_train[:, 1] +
            w_ridge * X_meta_train[:, 2]
        )
        mae_train = mean_absolute_error(y_train, pred_train)
        if best_blend is None or mae_train < best_blend['train_mae']:
            best_blend = {
                'w_cat': float(w_cat),
                'w_xgb': float(w_xgb),
                'w_ridge': float(w_ridge),
                'train_mae': float(mae_train),
            }

pred_blend = (
    best_blend['w_cat'] * X_meta_test[:, 0] +
    best_blend['w_xgb'] * X_meta_test[:, 1] +
    best_blend['w_ridge'] * X_meta_test[:, 2]
)

blend_row = score_row('OOF blend (Cat/XGB/Ridge)', y_test, pred_blend)
print('Best OOF blend weights:', best_blend)
print(blend_row)
blend_row

# %%
meta_models = {
    'Stacking - LinearRegression': LinearRegression(),
    'Stacking - Ridge': Ridge(alpha=1.0),
    'Stacking - Huber': Pipeline([
        ('s', StandardScaler()),
        ('m', HuberRegressor()),
    ]),
    'Stacking - ElasticNet': Pipeline([
        ('s', StandardScaler()),
        ('m', ElasticNet(alpha=0.01, l1_ratio=0.2, random_state=RANDOM_STATE, max_iter=10000)),
    ]),
}

stack_rows = []
stack_coefs = {}
stack_test_pred = {}
for name, meta in meta_models.items():
    meta.fit(X_meta_train, y_train)
    pred = meta.predict(X_meta_test)
    stack_test_pred[name] = pred
    row = score_row(name, y_test, pred)
    stack_rows.append(row)
    print(f"{name:28s} MAE={row['MAE']:.3f}  RMSE={row['RMSE']:.3f}  R2={row['R2']:.3f}")

    coef = None
    if hasattr(meta, 'coef_'):
        coef = meta.coef_
    elif hasattr(meta, 'named_steps') and hasattr(meta.named_steps.get('m', None), 'coef_'):
        coef = meta.named_steps['m'].coef_
    if coef is not None:
        stack_coefs[name] = {k: float(v) for k, v in zip(meta_feature_names, coef)}

stack_results_df = pd.DataFrame(stack_rows).sort_values('MAE').reset_index(drop=True)
stack_results_df

# %%
all_results = pd.concat([
    base_results_df,
    pd.DataFrame([blend_row]),
    stack_results_df,
], ignore_index=True).sort_values('MAE').reset_index(drop=True)

all_test_pred = {}
all_test_pred.update(base_test_pred)
all_test_pred['OOF blend (Cat/XGB/Ridge)'] = pred_blend
all_test_pred.update(stack_test_pred)

print('Final ranking')
print(all_results.to_string(index=False))
all_results

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

plot_df = all_results.sort_values('MAE', ascending=True)
axes[0].barh(plot_df['Model'], plot_df['MAE'], color='steelblue', alpha=0.9)
axes[0].set_title('MAE by model')
axes[0].set_xlabel('MAE')

plot_df2 = all_results.sort_values('R2', ascending=False)
axes[1].barh(plot_df2['Model'], plot_df2['R2'], color='darkorange', alpha=0.9)
axes[1].set_title('R2 by model')
axes[1].set_xlabel('R2')

plt.tight_layout()
plt.show()

# %%
best_model_row = all_results.iloc[0]
print('Best result on this split:')
print(best_model_row.to_string())

print('\nMeta-model coefficients (when available):')
for name, coef_map in stack_coefs.items():
    print(name)
    print(coef_map)

# %% [markdown]
# • ## 7. Évaluation Visuelle
#
#   On visualise le meilleur modèle retenu sur ce split avec :
#   - nuage `Elo réel` vs `Elo prédit`
#   - histogramme des résidus
#   - résidus en fonction des prédictions

# %%
best_model_name = all_results.iloc[0]['Model']
best_pred = all_test_pred[best_model_name]
residuals = y_test - best_pred

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Real vs predicted
min_elo = min(y_test.min(), best_pred.min())
max_elo = max(y_test.max(), best_pred.max())
axes[0].scatter(y_test, best_pred, s=18, alpha=0.55, color='teal', edgecolors='none')
axes[0].plot([min_elo, max_elo], [min_elo, max_elo], '--', color='red', linewidth=2)
axes[0].set_title(f'Elo reel vs predit - {best_model_name}')
axes[0].set_xlabel('Elo reel')
axes[0].set_ylabel('Elo predit')

# Residual histogram
axes[1].hist(residuals, bins=25, color='slateblue', alpha=0.85, edgecolor='white')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Distribution des residus')
axes[1].set_xlabel('Residus = reel - predit')
axes[1].set_ylabel('Frequence')

# Residuals vs predicted
axes[2].scatter(best_pred, residuals, s=18, alpha=0.55, color='darkorange', edgecolors='none')
axes[2].axhline(0, color='red', linestyle='--', linewidth=2)
axes[2].set_title('Residus vs Elo predit')
axes[2].set_xlabel('Elo predit')
axes[2].set_ylabel('Residus')

plt.tight_layout()
plt.show()

print('Best visualized model:', best_model_name)
print('MAE  =', round(mean_absolute_error(y_test, best_pred), 3))
print('RMSE =', round(np.sqrt(mean_squared_error(y_test, best_pred)), 3))
print('R2   =', round(r2_score(y_test, best_pred), 3))

# %%
import json
from pathlib import Path

all_results.to_csv('results/tables/results_stacking_from_csv_1000x100.csv', index=False)
print('Saved results_stacking_from_csv_1000x100.csv')

Path('results/configs/stacking_blend_weights_1000x100.json').write_text(
    json.dumps(best_blend, indent=2),
    encoding='utf-8'
)
print('Saved stacking_blend_weights_1000x100.json')

Path('results/configs/stacking_meta_coeffs_1000x100.json').write_text(
    json.dumps(stack_coefs, indent=2),
    encoding='utf-8'
)
print('Saved stacking_meta_coeffs_1000x100.json')
