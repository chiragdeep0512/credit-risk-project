# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 04: MINIMUM VERSION (under 800MB RAM)
#  Author: Naman Deep Singh
# ============================================================
#  pip install lightgbm scikit-learn matplotlib pandas pyarrow

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, gc, warnings
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report,
                             precision_recall_curve)
import lightgbm as lgb

warnings.filterwarnings('ignore')

print('=' * 50)
print('  Phase 04 — Credit Risk Model (Mini)')
print('=' * 50)

# ── PATHS ─────────────────────────────────────────────────────
PROCESSED = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed')
MODELS    = Path(r'D:\Project\Claude Project\credit-risk-project\models')
CHARTS    = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed\model_charts')
MODELS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

# ── LOAD SPLIT INFO ───────────────────────────────────────────
with open(PROCESSED / 'split_info.json') as f:
    info = json.load(f)
split_week = info['split_week']
print(f'Split week: {split_week}')

# ── LOAD ONLY 30 COLS + SAMPLE ────────────────────────────────
print('\nLoading data (30 cols, 100K sample)...')

with open(PROCESSED / 'feature_list.txt') as f:
    lines = f.readlines()
feat_cols = [l.strip() for l in lines
             if l.strip() and not l.startswith('Total')][:30]

load_cols = ['case_id', 'target', 'WEEK_NUM'] + feat_cols

df = pd.read_parquet(
    PROCESSED / 'master_train.parquet',
    columns=load_cols
)
# Fix duplicate column names if any
df = df.loc[:, ~df.columns.duplicated()]
print(f'Loaded shape: {df.shape}')

# ── SAMPLE 100K TRAIN + FULL VAL ─────────────────────────────
train_df = df[df['WEEK_NUM'] <= split_week]
val_df   = df[df['WEEK_NUM'] >  split_week].copy()

np.random.seed(42)
def_idx   = train_df[train_df['target'] == 1].index
nodef_idx = train_df[train_df['target'] == 0].index

s_idx = np.concatenate([
    np.random.choice(def_idx,   min(5000,  len(def_idx)),   replace=False),
    np.random.choice(nodef_idx, min(95000, len(nodef_idx)), replace=False)
])
np.random.shuffle(s_idx)
train_sample = train_df.loc[s_idx].copy()

print(f'Train sample : {len(train_sample):,}')
print(f'Val set      : {len(val_df):,}')
print(f'Default rate : {train_sample["target"].mean()*100:.2f}%')

del df, train_df
gc.collect()

# ── ARRAYS ────────────────────────────────────────────────────
X_train  = train_sample[feat_cols].values.astype(np.float32)
y_train  = train_sample['target'].values
X_val    = val_df[feat_cols].values.astype(np.float32)
y_val    = val_df['target'].values
week_val = val_df['WEEK_NUM'].reset_index(drop=True)
y_val_s  = val_df['target'].reset_index(drop=True)

del train_sample, val_df
gc.collect()

print(f'X_train: {X_train.shape}  X_val: {X_val.shape}')

# ── TRAIN LIGHTGBM ────────────────────────────────────────────
print('\nTraining LightGBM...')

params = {
    'objective':        'binary',
    'metric':           'auc',
    'num_leaves':       31,
    'learning_rate':    0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'is_unbalance':     True,
    'verbose':          -1,
    'n_jobs':           2,
    'random_state':     42,
}

lgb_tr = lgb.Dataset(X_train, label=y_train,
                     feature_name=feat_cols, free_raw_data=True)
lgb_vl = lgb.Dataset(X_val, label=y_val,
                     reference=lgb_tr, free_raw_data=True)

model = lgb.train(
    params, lgb_tr,
    num_boost_round=200,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(20, verbose=False),
        lgb.log_evaluation(50)
    ]
)
print(f'Best iteration: {model.best_iteration}')

# ── EVALUATE ──────────────────────────────────────────────────
print('\nResults:')
y_prob = model.predict(X_val)

prec_a, rec_a, thresh_a = precision_recall_curve(y_val, y_prob)
f1_a   = 2 * prec_a * rec_a / (prec_a + rec_a + 1e-9)
best_t = float(thresh_a[np.argmax(f1_a[:-1])])
y_pred = (y_prob >= best_t).astype(int)

auc = round(roc_auc_score(y_val, y_prob), 4)
f1  = round(f1_score(y_val, y_pred, zero_division=0), 4)
pr  = round(precision_score(y_val, y_pred, zero_division=0), 4)
rc  = round(recall_score(y_val, y_pred, zero_division=0), 4)

print(f'  AUC-ROC   : {auc}')
print(f'  F1        : {f1}')
print(f'  Precision : {pr}')
print(f'  Recall    : {rc}')
print(f'  Threshold : {best_t:.4f}')
print()
print(classification_report(y_val, y_pred,
      target_names=['Non-Default', 'Default']))

# ── SAVE ──────────────────────────────────────────────────────
model.save_model(str(MODELS / 'best_model_lgb.txt'))
with open(MODELS / 'best_threshold.json', 'w') as f:
    json.dump({
        'threshold':    best_t,
        'auc':          auc,
        'f1':           f1,
        'precision':    pr,
        'recall':       rc,
        'feature_cols': feat_cols
    }, f, indent=2)
print('Saved: best_model_lgb.txt')
print('Saved: best_threshold.json')

# ── 3 CHARTS ──────────────────────────────────────────────────
print('\nGenerating 3 charts...')
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

# Chart 1 — ROC
fpr, tpr, _ = roc_curve(y_val, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, '#4A90D9', linewidth=2.5, label=f'LightGBM  AUC={auc}')
ax.plot([0,1],[0,1], 'k--', alpha=0.4, label='Random')
ax.fill_between(fpr, tpr, alpha=0.1, color='#4A90D9')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Credit Risk Model', fontsize=13, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(CHARTS / '01_roc_curve.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()
print('  01_roc_curve.png')

# Chart 2 — Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_val, y_pred),
    display_labels=['Non-Default', 'Default']
).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix  |  threshold={best_t:.3f}', fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS / '02_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()
print('  02_confusion_matrix.png')

# Chart 3 — Feature Importance
imp = pd.DataFrame({
    'Feature':    feat_cols,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
imp.to_csv(MODELS / 'feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 8))
clrs = ['#E05252' if i < 5 else '#4A90D9' for i in range(len(imp))]
ax.barh(range(len(imp)), imp['Importance'],
        color=clrs, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(imp)))
ax.set_yticklabels(imp['Feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Feature Importance (Top 30)\nRed = Top 5', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS / '03_feature_importance.png', dpi=100, bbox_inches='tight')
plt.show(); plt.close()
print('  03_feature_importance.png')

# ── DONE ──────────────────────────────────────────────────────
print('\n' + '=' * 50)
print('  PHASE 04 COMPLETE')
print('=' * 50)
print(f'''
  AUC-ROC   : {auc}
  F1        : {f1}
  Precision : {pr}
  Recall    : {rc}
  Threshold : {best_t:.4f}

  Saved:
  - models/best_model_lgb.txt
  - models/best_threshold.json
  - models/feature_importance.csv
  - 3 charts saved

  Next -> Phase 05: SHAP
''')