# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 04 + 05: Model (Fixed) + SHAP Explainability
#  Author: Naman Deep Singh
#  RAM: works under 1.5GB available
# ============================================================
#  pip install lightgbm scikit-learn matplotlib shap

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # no GUI popup — saves RAM
import json, gc, warnings
from pathlib import Path
from sklearn.metrics import (roc_auc_score, f1_score,
                             precision_score, recall_score,
                             roc_curve, confusion_matrix,
                             ConfusionMatrixDisplay,
                             classification_report,
                             precision_recall_curve)
import lightgbm as lgb
import shap

warnings.filterwarnings('ignore')

print('=' * 55)
print('  Phase 04+05 — LightGBM + SHAP (RAM Safe)')
print('=' * 55)

# ── PATHS ─────────────────────────────────────────────────────
PROCESSED = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed')
MODELS    = Path(r'D:\Project\Claude Project\credit-risk-project\models')
CHARTS    = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed\model_charts')
MODELS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

# ── LOAD SPLIT INFO ───────────────────────────────────────────
with open(PROCESSED / 'split_info.json') as f:
    info = json.load(f)
split_week   = info['split_week']
default_rate = info['default_rate']
print(f'split_week   : {split_week}')
print(f'default_rate : {default_rate*100:.2f}%')

# ============================================================
# STEP 1 — SMART FEATURE SELECTION
# Pick best features from EACH table group (diverse)
# ============================================================

print('\nSelecting diverse features from all table groups...')

# Manually picked best features from each source table
# These are the most business-meaningful and high-importance cols
BEST_FEATURES = [
    # ── applprev_1 (previous applications) ───────────────────
    'ap1_cancelreason_3545846M_nunique',
    'ap1_count',
    'ap1_credamount_590A_mean',
    'ap1_annuity_853A_mean',
    'ap1_credacc_actualbalance_314A_mean',
    'ap1_actualdpd_943P_mean',
    'ap1_credamount_590A_max',
    'ap1_annuity_853A_max',

    # ── static_0 (applicant info) ────────────────────────────
    's0_credamount_770A',
    's0_pmtnum_254L',
    's0_annuity_780A',
    's0_currdebt_22A',
    's0_numinstpaidearly3d_817L',
    's0_numinstpaid_4499770L',
    's0_numinstlate_4469289L',
    's0_pctinstlate_3496139L',

    # ── static_cb_0 (external credit bureau static) ──────────
    'cb0_pmts_overdue_1140A',
    'cb0_pmts_dpd_1073P',
    'cb0_dpdmaxdateyear_596T',

    # ── bureau_b_1 (bureau historical) ───────────────────────
    'burb1_count',
    'burb1_pmts_dpd_1073P_mean',
    'burb1_pmts_dpd_1073P_max',
    'burb1_pmts_overdue_1140A_mean',
    'burb1_pmts_overdue_1140A_sum',

    # ── bureau_a_1 (bureau A aggregated) ─────────────────────
    'bura1_count',
    'bura1_annuity_853A_max',
    'bura1_annuity_853A_mean',

    # ── tax registry ─────────────────────────────────────────
    'taxa_count',
    'taxb_count',

    # ── person_1 ─────────────────────────────────────────────
    'per1_total_persons',

    # ── engineered features ───────────────────────────────────
    'fe_month',
    'fe_year',
    'fe_dayofweek',
    'WEEK_NUM',
]

# ── Load only these columns from parquet ─────────────────────
# First check which ones actually exist in the file
print('Checking which features exist in master_train.parquet...')
all_cols = pd.read_parquet(
    PROCESSED / 'master_train.parquet',
    columns=['case_id']
).columns.tolist()

# Peek at actual columns
sample_df = pd.read_parquet(
    PROCESSED / 'master_train.parquet'
).columns.tolist()

available = [c for c in BEST_FEATURES if c in sample_df]
missing   = [c for c in BEST_FEATURES if c not in sample_df]

print(f'Features found   : {len(available)} / {len(BEST_FEATURES)}')
if missing:
    print(f'Not found (skip) : {missing[:5]}...')

# Use available features + add more from feature_list if needed
if len(available) < 15:
    print('Too few features found — loading top features from feature_list.txt')
    with open(PROCESSED / 'feature_list.txt') as f:
        lines = f.readlines()
    all_feats = [l.strip() for l in lines
                 if l.strip() and not l.startswith('Total')]
    # Pick from different prefixes for diversity
    prefixes  = ['s0_', 'cb0_', 'ap1_', 'burb1_', 'bura1_',
                 'per1_', 'fe_', 'dep1_', 'deb1_', 'taxa_']
    available = []
    for pfx in prefixes:
        cols = [c for c in all_feats if c.startswith(pfx)][:4]
        available.extend(cols)
    available = available[:35]
    print(f'Fallback: loaded {len(available)} diverse features')

FEAT_COLS = available
load_cols = ['case_id', 'target', 'WEEK_NUM'] + FEAT_COLS

print(f'\nFinal feature count : {len(FEAT_COLS)}')
print(f'Loading parquet...')

df = pd.read_parquet(
    PROCESSED / 'master_train.parquet',
    columns=[c for c in load_cols if c in sample_df]
)
df = df.loc[:, ~df.columns.duplicated()]
FEAT_COLS = [c for c in FEAT_COLS if c in df.columns]
print(f'Loaded shape : {df.shape}')
print(f'Using features: {len(FEAT_COLS)}')

# ============================================================
# STEP 2 — STRATIFIED SAMPLE
# ============================================================

train_df = df[df['WEEK_NUM'] <= split_week]
val_df   = df[df['WEEK_NUM'] >  split_week].copy()

# Stratified 150K sample — more data = better AUC
np.random.seed(42)
def_idx   = train_df[train_df['target'] == 1].index
nodef_idx = train_df[train_df['target'] == 0].index

n_def   = min(len(def_idx),   8000)
n_nodef = min(len(nodef_idx), 142000)

s_idx = np.concatenate([
    np.random.choice(def_idx,   n_def,   replace=False),
    np.random.choice(nodef_idx, n_nodef, replace=False)
])
np.random.shuffle(s_idx)
train_sample = train_df.loc[s_idx].copy()

print(f'\nTrain sample : {len(train_sample):,}')
print(f'Val set      : {len(val_df):,}')
print(f'Default rate : {train_sample["target"].mean()*100:.2f}%')

del df, train_df
gc.collect()

X_train  = train_sample[FEAT_COLS].values.astype(np.float32)
y_train  = train_sample['target'].values
X_val    = val_df[FEAT_COLS].values.astype(np.float32)
y_val    = val_df['target'].values
week_val = val_df['WEEK_NUM'].reset_index(drop=True)
y_val_s  = val_df['target'].reset_index(drop=True)

# Save val_df for SHAP (need feature names)
X_val_df = val_df[FEAT_COLS].copy()

del train_sample, val_df
gc.collect()

print(f'X_train: {X_train.shape}  X_val: {X_val.shape}')

# ============================================================
# STEP 3 — TRAIN LIGHTGBM
# ============================================================

print('\n' + '-' * 55)
print('Training LightGBM...')
print('-' * 55)

params = {
    'objective':        'binary',
    'metric':           'auc',
    'num_leaves':       50,
    'learning_rate':    0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'min_child_samples': 20,
    'is_unbalance':     True,
    'verbose':          -1,
    'n_jobs':           2,
    'random_state':     42,
}

lgb_tr = lgb.Dataset(X_train, label=y_train,
                     feature_name=FEAT_COLS, free_raw_data=True)
lgb_vl = lgb.Dataset(X_val, label=y_val,
                     reference=lgb_tr, free_raw_data=True)

model = lgb.train(
    params, lgb_tr,
    num_boost_round=500,
    valid_sets=[lgb_vl],
    callbacks=[
        lgb.early_stopping(30, verbose=False),
        lgb.log_evaluation(100)
    ]
)
print(f'Best iteration: {model.best_iteration}')

# ============================================================
# STEP 4 — EVALUATE
# ============================================================

print('\nEvaluating...')
y_prob = model.predict(X_val)

prec_a, rec_a, thresh_a = precision_recall_curve(y_val, y_prob)
f1_a   = 2 * prec_a * rec_a / (prec_a + rec_a + 1e-9)
best_t = float(thresh_a[np.argmax(f1_a[:-1])])
y_pred = (y_prob >= best_t).astype(int)

auc = round(roc_auc_score(y_val, y_prob), 4)
f1  = round(f1_score(y_val, y_pred, zero_division=0), 4)
pr  = round(precision_score(y_val, y_pred, zero_division=0), 4)
rc  = round(recall_score(y_val, y_pred, zero_division=0), 4)

print(f'\n  AUC-ROC   : {auc}')
print(f'  F1        : {f1}')
print(f'  Precision : {pr}')
print(f'  Recall    : {rc}')
print(f'  Threshold : {best_t:.4f}')
print()
print(classification_report(y_val, y_pred,
      target_names=['Non-Default', 'Default']))

# Stability
df_s = pd.DataFrame({'week': week_val.values,
                     'target': y_val_s.values, 'prob': y_prob})
wg_rows = []
for wk, grp in df_s.groupby('week'):
    if grp['target'].nunique() < 2: continue
    g = 2 * roc_auc_score(grp['target'], grp['prob']) - 1
    wg_rows.append({'week': wk, 'gini': g})
wg   = pd.DataFrame(wg_rows).sort_values('week')
slp  = np.polyfit(wg['week'], wg['gini'], 1)[0]
stab = round(wg['gini'].mean() + 88.0 * min(0, slp), 4)
print(f'  Stability : {stab}')

# ── Save model ────────────────────────────────────────────────
model.save_model(str(MODELS / 'best_model_lgb.txt'))
with open(MODELS / 'best_threshold.json', 'w') as f:
    json.dump({
        'threshold':    best_t,
        'auc':          auc,
        'f1':           f1,
        'precision':    pr,
        'recall':       rc,
        'feature_cols': FEAT_COLS
    }, f, indent=2)
print('\nSaved: best_model_lgb.txt + best_threshold.json')

# ============================================================
# STEP 5 — PHASE 04 CHARTS
# ============================================================

print('\nGenerating Phase 04 charts...')
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

def save_fig(name):
    plt.savefig(CHARTS / name, dpi=100, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {name}')

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr, tpr, '#4A90D9', linewidth=2.5, label=f'AUC = {auc}')
ax.plot([0,1],[0,1], 'k--', alpha=0.4)
ax.fill_between(fpr, tpr, alpha=0.1, color='#4A90D9')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve — Credit Risk Model', fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('01_roc_curve.png')

# Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix(y_val, y_pred),
    display_labels=['Non-Default', 'Default']
).plot(ax=ax, cmap='Blues', colorbar=False)
ax.set_title(f'Confusion Matrix | threshold={best_t:.3f}', fontweight='bold')
plt.tight_layout()
save_fig('02_confusion_matrix.png')

# Feature Importance
imp = pd.DataFrame({
    'Feature':    FEAT_COLS,
    'Importance': model.feature_importance(importance_type='gain')
}).sort_values('Importance', ascending=False)
imp.to_csv(MODELS / 'feature_importance.csv', index=False)

fig, ax = plt.subplots(figsize=(10, 8))
top_imp = imp.head(20)
clrs = ['#E05252' if i < 5 else '#4A90D9' for i in range(len(top_imp))]
ax.barh(range(len(top_imp)), top_imp['Importance'],
        color=clrs, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(top_imp)))
ax.set_yticklabels(top_imp['Feature'], fontsize=9)
ax.invert_yaxis()
ax.set_title('Top 20 Feature Importance | Red=Top 5', fontweight='bold')
plt.tight_layout()
save_fig('03_feature_importance.png')

# Score Distribution
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(y_prob[y_val==0], bins=50, alpha=0.6, color='#4A90D9',
        density=True, label='Non-Default')
ax.hist(y_prob[y_val==1], bins=50, alpha=0.6, color='#E05252',
        density=True, label='Default')
ax.axvline(best_t, color='black', linestyle='--', linewidth=2,
           label=f'Threshold: {best_t:.3f}')
ax.set_xlabel('Predicted Default Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution by True Class', fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('04_score_distribution.png')

# Stability
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(wg['week'], wg['gini'], '#4A90D9',
        linewidth=2.5, marker='o', markersize=4)
ax.fill_between(wg['week'], wg['gini'], alpha=0.1, color='#4A90D9')
ax.axhline(wg['gini'].mean(), color='gray', linestyle='--',
           label=f'Mean Gini: {wg["gini"].mean():.3f}')
ax.set_xlabel('WEEK_NUM')
ax.set_ylabel('Gini')
ax.set_title(f'Model Stability | Score: {stab}', fontweight='bold')
ax.legend()
plt.tight_layout()
save_fig('05_stability.png')

gc.collect()
print('Phase 04 charts done!')

# ============================================================
# PHASE 05 — SHAP EXPLAINABILITY
# ============================================================

print('\n' + '=' * 55)
print('  PHASE 05 — SHAP Explainability')
print('=' * 55)

# SHAP on 500 val samples only — RAM safe
SHAP_SAMPLE = 500
np.random.seed(42)
shap_idx    = np.random.choice(len(X_val_df), SHAP_SAMPLE, replace=False)
X_shap      = X_val_df.iloc[shap_idx].reset_index(drop=True)
y_shap      = y_val[shap_idx]

print(f'SHAP sample: {len(X_shap)} rows')
print('Calculating SHAP values (TreeExplainer)...')

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

# For binary LightGBM — shap_values is a list [neg, pos]
# We want class 1 (default) SHAP values
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

print(f'SHAP values shape: {sv.shape}')
print('SHAP done!')

# ── SHAP Chart 1: Summary Plot (Beeswarm) ─────────────────────
print('\nGenerating SHAP charts...')
fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(
    sv, X_shap,
    feature_names=FEAT_COLS,
    show=False,
    max_display=20,
    plot_size=None
)
plt.title('SHAP Summary Plot — Top 20 Features\n(Red=increases default risk, Blue=decreases)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS / '06_shap_summary.png', dpi=100, bbox_inches='tight')
plt.close()
print('  Saved: 06_shap_summary.png')

# ── SHAP Chart 2: Bar Plot (Mean |SHAP|) ─────────────────────
fig, ax = plt.subplots(figsize=(10, 7))
shap.summary_plot(
    sv, X_shap,
    feature_names=FEAT_COLS,
    plot_type='bar',
    show=False,
    max_display=20,
    plot_size=None
)
plt.title('SHAP Feature Importance (Mean |SHAP value|)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS / '07_shap_bar.png', dpi=100, bbox_inches='tight')
plt.close()
print('  Saved: 07_shap_bar.png')

# ── SHAP Chart 3: Waterfall — 1 Defaulter ────────────────────
# Pick one actual defaulter from shap sample
defaulters = np.where(y_shap == 1)[0]
if len(defaulters) > 0:
    idx = defaulters[0]
    explanation = shap.Explanation(
        values        = sv[idx],
        base_values   = explainer.expected_value
                        if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
        data          = X_shap.iloc[idx].values,
        feature_names = FEAT_COLS
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title('SHAP Waterfall — Single Defaulter Explained',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHARTS / '08_shap_waterfall_default.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print('  Saved: 08_shap_waterfall_default.png')

# ── SHAP Chart 4: Waterfall — 1 Non-Defaulter ────────────────
nondefaulters = np.where(y_shap == 0)[0]
if len(nondefaulters) > 0:
    idx = nondefaulters[0]
    explanation = shap.Explanation(
        values        = sv[idx],
        base_values   = explainer.expected_value
                        if not isinstance(explainer.expected_value, list)
                        else explainer.expected_value[1],
        data          = X_shap.iloc[idx].values,
        feature_names = FEAT_COLS
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(explanation, show=False, max_display=15)
    plt.title('SHAP Waterfall — Single Non-Defaulter Explained',
              fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(CHARTS / '09_shap_waterfall_nondefault.png',
                dpi=100, bbox_inches='tight')
    plt.close()
    print('  Saved: 09_shap_waterfall_nondefault.png')

# ── SHAP Chart 5: Dependence Plot (top feature) ───────────────
top_feat = imp.iloc[0]['Feature']
top_idx  = FEAT_COLS.index(top_feat) if top_feat in FEAT_COLS else 0

fig, ax = plt.subplots(figsize=(9, 6))
shap.dependence_plot(
    top_idx, sv, X_shap,
    feature_names=FEAT_COLS,
    show=False, ax=ax
)
ax.set_title(f'SHAP Dependence Plot — {top_feat}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CHARTS / '10_shap_dependence.png',
            dpi=100, bbox_inches='tight')
plt.close()
print('  Saved: 10_shap_dependence.png')

# ── SHAP Business Reasons ─────────────────────────────────────
print('\nGenerating SHAP business reason dictionary...')

# Mean SHAP per feature
mean_shap = np.abs(sv).mean(axis=0)
shap_df   = pd.DataFrame({
    'Feature':   FEAT_COLS,
    'Mean_SHAP': mean_shap
}).sort_values('Mean_SHAP', ascending=False)

# Human-readable reasons for top features
REASON_MAP = {
    'ap1_count':                      'Number of previous loan applications',
    'ap1_cancelreason_3545846M_nunique': 'Variety of cancellation reasons in past',
    'ap1_credamount_590A_mean':        'Average credit amount in previous applications',
    'ap1_annuity_853A_mean':           'Average annuity in previous applications',
    'ap1_actualdpd_943P_mean':         'Average days past due in previous loans',
    'ap1_credacc_actualbalance_314A_mean': 'Average actual balance in credit accounts',
    'bura1_count':                     'Number of bureau A credit records',
    'burb1_count':                     'Number of bureau B credit records',
    'burb1_pmts_dpd_1073P_mean':       'Average payment days past due (bureau B)',
    'burb1_pmts_dpd_1073P_max':        'Maximum payment days past due (bureau B)',
    'burb1_pmts_overdue_1140A_mean':   'Average overdue payment amount (bureau B)',
    's0_credamount_770A':              'Current loan credit amount requested',
    's0_annuity_780A':                 'Monthly annuity of current loan',
    's0_currdebt_22A':                 'Current outstanding debt amount',
    's0_numinstlate_4469289L':         'Number of late instalments historically',
    's0_pctinstlate_3496139L':         'Percentage of late instalments',
    'cb0_pmts_dpd_1073P':              'Credit bureau DPD (days past due)',
    'cb0_pmts_overdue_1140A':          'Credit bureau overdue amount',
    'per1_total_persons':              'Number of persons in application',
    'fe_month':                        'Month of loan application',
    'fe_year':                         'Year of loan application',
    'taxa_count':                      'Number of tax registry A records',
    'taxb_count':                      'Number of tax registry B records',
}

print('\n  Top 15 features with business explanations:')
print('-' * 60)
for _, row in shap_df.head(15).iterrows():
    feat   = row['Feature']
    reason = REASON_MAP.get(feat, feat.replace('_', ' ').title())
    print(f'  {feat:<40} → {reason}')

# Save SHAP values + reasons
shap_df['Business_Reason'] = shap_df['Feature'].map(
    lambda x: REASON_MAP.get(x, x.replace('_', ' ').title())
)
shap_df.to_csv(MODELS / 'shap_feature_importance.csv', index=False)
print('\n  Saved: shap_feature_importance.csv')

# Save SHAP values array for Streamlit
np.save(str(MODELS / 'shap_values_sample.npy'), sv)
X_shap.to_parquet(str(MODELS / 'shap_X_sample.parquet'), index=False)
print('  Saved: shap_values_sample.npy')
print('  Saved: shap_X_sample.parquet')

gc.collect()

# ============================================================
# FINAL SUMMARY
# ============================================================

print('\n' + '=' * 55)
print('  PHASE 04 + 05 COMPLETE')
print('=' * 55)
print(f'''
  MODEL RESULTS
  AUC-ROC   : {auc}
  F1        : {f1}
  Precision : {pr}
  Recall    : {rc}
  Stability : {stab}
  Threshold : {best_t:.4f}

  PHASE 04 CHARTS (model_charts/):
    01_roc_curve.png
    02_confusion_matrix.png
    03_feature_importance.png
    04_score_distribution.png
    05_stability.png

  PHASE 05 SHAP CHARTS (model_charts/):
    06_shap_summary.png        ← beeswarm (most important chart)
    07_shap_bar.png            ← mean SHAP bar
    08_shap_waterfall_default.png   ← 1 defaulter explained
    09_shap_waterfall_nondefault.png ← 1 non-defaulter explained
    10_shap_dependence.png     ← top feature effect

  SAVED MODELS (models/):
    best_model_lgb.txt
    best_threshold.json
    feature_importance.csv
    shap_feature_importance.csv
    shap_values_sample.npy     ← for Streamlit
    shap_X_sample.parquet      ← for Streamlit

  Next -> Phase 06: Streamlit Deployment
''')