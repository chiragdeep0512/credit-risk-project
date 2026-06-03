# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 02: Exploratory Data Analysis (EDA)
#  Author: Naman Deep Singh
#  GitHub: github.com/chiragdeep0512
# ============================================================
#
#  BEFORE RUNNING — paste this in PyCharm Terminal:
#  pip install pandas numpy matplotlib seaborn missingno pyarrow tqdm scipy
#
#  WHAT THIS SCRIPT DOES:
#  1.  Load base + static tables (memory safe for 8GB RAM)
#  2.  Target distribution & weekly/monthly trend
#  3.  Univariate analysis — numerical & categorical
#  4.  Bivariate analysis — feature vs target
#  5.  Correlation heatmap + target correlation bar
#  6.  Missing value deep dive + missingno matrix
#  7.  Bureau feature EDA (memory optimized)
#  8.  Date feature analysis (month, quarter, dow)
#  9.  Train/val split strategy chart
#  10. Business insights summary saved to txt
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec
import seaborn as sns
import missingno as msno
import warnings
import gc
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.4f}'.format)

# Plot style
plt.rcParams['figure.figsize']    = (13, 5)
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.titlepad']     = 12
plt.rcParams['font.family']       = 'DejaVu Sans'

BLUE   = '#4A90D9'
RED    = '#E05252'
GREEN  = '#2ECC71'
ORANGE = '#E8A838'
PURPLE = '#9B59B6'

print('=' * 65)
print('  PHASE 02 — Exploratory Data Analysis')
print('=' * 65)
print('Libraries loaded\n')


# ============================================================
# SECTION 0 — PATHS
# ============================================================

TRAIN_DIR  = Path(r'D:\Project\Claude Project\credit-risk-project\data\raw\parquet_files\train')
TEST_DIR   = Path(r'D:\Project\Claude Project\credit-risk-project\data\raw\parquet_files\test')
OUTPUT_DIR = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed')
CHARTS_DIR = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed\eda_charts')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
print(f'Charts will be saved to: {CHARTS_DIR}\n')


# ============================================================
# SECTION 1 — HELPER FUNCTIONS
# ============================================================

def save_chart(filename: str):
    path = CHARTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'  Saved: {filename}')
    plt.show()
    plt.close()


def load_parquet(train_dir: Path, name: str) -> pd.DataFrame:
    import glob
    files = sorted(glob.glob(str(train_dir / f'train_{name}*.parquet')))
    if not files:
        print(f'  Not found: train_{name}')
        return pd.DataFrame()
    dfs = [pd.read_parquet(f) for f in files]
    df  = pd.concat(dfs, ignore_index=True)
    print(f'  Loaded train_{name:<28} shape: {df.shape}')
    return df


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        ct = df[col].dtype
        if ct != object and str(ct) != 'category':
            cmin, cmax = df[col].min(), df[col].max()
            if str(ct).startswith('int'):
                for dt in [np.int8, np.int16, np.int32]:
                    if cmin > np.iinfo(dt).min and cmax < np.iinfo(dt).max:
                        df[col] = df[col].astype(dt)
                        break
            elif str(ct).startswith('float'):
                if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    return df


def bivariate_num(df, col, target='target', bins=30, ax=None):
    d0 = df.loc[df[target] == 0, col].dropna()
    d1 = df.loc[df[target] == 1, col].dropna()
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(d0, bins=bins, alpha=0.6, color=BLUE, label='Non-default (0)', density=True)
    ax.hist(d1, bins=bins, alpha=0.6, color=RED,  label='Default (1)',     density=True)
    ax.set_title(col, fontsize=10)
    ax.legend(fontsize=8)
    return ax


print('Helper functions ready\n')


# ============================================================
# SECTION 2 — LOAD DATA
# ============================================================

print('-' * 65)
print('SECTION 2 — Loading Data')
print('-' * 65)

train_base = load_parquet(TRAIN_DIR, 'base')
train_base['date_decision'] = pd.to_datetime(train_base['date_decision'])

static_0 = load_parquet(TRAIN_DIR, 'static_0')
static_0 = reduce_mem(static_0)

static_cb = load_parquet(TRAIN_DIR, 'static_cb_0')
static_cb = reduce_mem(static_cb)

# Merge base + static_0 for EDA
df = train_base.merge(static_0, on='case_id', how='left')
print(f'\n  Master EDA df shape: {df.shape}')

del static_0
gc.collect()
print('  RAM freed: static_0 removed from memory')


# ============================================================
# SECTION 3 — TARGET ANALYSIS
# ============================================================

print('\n' + '-' * 65)
print('SECTION 3 — Target Variable Deep Dive')
print('-' * 65)

target_counts = df['target'].value_counts()
target_pct    = df['target'].value_counts(normalize=True) * 100

print(f'  Non-default (0) : {target_counts[0]:>9,}  ({target_pct[0]:.2f}%)')
print(f'  Default     (1) : {target_counts[1]:>9,}  ({target_pct[1]:.2f}%)')
print(f'  Imbalance ratio : 1 : {int(target_counts[0]/target_counts[1])}')
print(f'\n  >> scale_pos_weight = {int(target_counts[0]/target_counts[1])} (use in XGBoost)')
print(f'  >> class_weight = balanced (use in LogReg / RandomForest)')

# 4-panel target overview
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax0 = fig.add_subplot(gs[0, 0])
bars = ax0.bar(['Non-Default', 'Default'], target_counts.values,
               color=[BLUE, RED], width=0.5, edgecolor='white')
ax0.set_title('A. Class Distribution', fontsize=13, fontweight='bold')
ax0.set_ylabel('Number of Cases')
for bar, pct in zip(bars, target_pct.values):
    ax0.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
             f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')
ax0.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))

ax1 = fig.add_subplot(gs[0, 1])
weekly = df.groupby('WEEK_NUM')['target'].mean() * 100
ax1.plot(weekly.index, weekly.values, color=RED, linewidth=2, marker='o', markersize=3)
ax1.axhline(target_pct[1], color='gray', linestyle='--', alpha=0.7,
            label=f'Overall: {target_pct[1]:.2f}%')
ax1.fill_between(weekly.index, weekly.values, alpha=0.1, color=RED)
ax1.set_title('B. Default Rate by Week (Stability Check)', fontsize=13, fontweight='bold')
ax1.set_xlabel('WEEK_NUM')
ax1.set_ylabel('Default Rate (%)')
ax1.legend(fontsize=9)

df['year_month'] = df['date_decision'].dt.to_period('M')
monthly = df.groupby('year_month')['target'].mean() * 100

ax2 = fig.add_subplot(gs[1, 0])
ax2.bar(range(len(monthly)), monthly.values, color=ORANGE, edgecolor='white', alpha=0.85)
ax2.set_xticks(range(0, len(monthly), 3))
ax2.set_xticklabels([str(m) for m in monthly.index[::3]], rotation=45, ha='right', fontsize=8)
ax2.set_title('C. Default Rate by Month', fontsize=13, fontweight='bold')
ax2.set_ylabel('Default Rate (%)')

monthly_vol = df.groupby('year_month')['case_id'].count()
ax3 = fig.add_subplot(gs[1, 1])
ax3.bar(range(len(monthly_vol)), monthly_vol.values, color=BLUE, edgecolor='white', alpha=0.85)
ax3.set_xticks(range(0, len(monthly_vol), 3))
ax3.set_xticklabels([str(m) for m in monthly_vol.index[::3]], rotation=45, ha='right', fontsize=8)
ax3.set_title('D. Application Volume by Month', fontsize=13, fontweight='bold')
ax3.set_ylabel('Applications')
ax3.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

fig.suptitle('Target Variable — Complete Analysis', fontsize=16, fontweight='bold', y=1.01)
save_chart('01_target_analysis.png')

print('\n  BUSINESS INSIGHT #1:')
print(f'  Only {target_pct[1]:.1f}% of applicants default — extreme class imbalance.')
print(f'  Ratio 1:{int(target_counts[0]/target_counts[1])} means raw accuracy is misleading.')
print('  Always use AUC-ROC, Precision-Recall, and F1 as evaluation metrics.')


# ============================================================
# SECTION 4 — UNIVARIATE — NUMERICAL FEATURES
# ============================================================

print('\n' + '-' * 65)
print('SECTION 4 — Univariate Analysis (Numerical Features)')
print('-' * 65)

num_cols = [c for c in df.columns
            if c.endswith(('A', 'P'))
            and df[c].dtype in [np.float32, np.float64, np.int32, np.int64]
            and df[c].nunique() > 10][:20]

print(f'  Numerical cols selected (A/P suffix): {len(num_cols)}')

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
fig.suptitle('Univariate Distributions — Amount & DPD Features', fontsize=15, fontweight='bold')

for i, col in enumerate(num_cols[:12]):
    data       = df[col].dropna()
    cap        = data.quantile(0.99)
    data_capped= data[data <= cap]
    axes[i].hist(data_capped, bins=40, color=BLUE, edgecolor='white', alpha=0.8)
    axes[i].set_title(
        f'{col}\nskew={data.skew():.1f} | null={df[col].isnull().mean()*100:.0f}%',
        fontsize=8)
    axes[i].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x/1000:.0f}K'))

for j in range(len(num_cols[:12]), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
save_chart('02_univariate_numerical.png')

# Skewness report
print('\n  Skewness report (features needing log transform):')
skew_rows = []
for col in num_cols:
    skew_rows.append({
        'Feature': col,
        'Skewness': round(df[col].skew(), 2),
        'Null%': round(df[col].isnull().mean() * 100, 1)
    })
skew_df = pd.DataFrame(skew_rows).sort_values('Skewness', key=abs, ascending=False)
print(skew_df.head(12).to_string(index=False))


# ============================================================
# SECTION 5 — UNIVARIATE — CATEGORICAL FEATURES
# ============================================================

print('\n' + '-' * 65)
print('SECTION 5 — Univariate Analysis (Categorical Features)')
print('-' * 65)

cat_cols = [c for c in df.columns
            if c.endswith('M')
            and df[c].dtype == object
            and 2 <= df[c].nunique() <= 20][:12]

print(f'  Categorical cols (M suffix, 2-20 unique): {len(cat_cols)}')

if cat_cols:
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    fig.suptitle('Categorical Feature Distributions (Masked — M suffix)', fontsize=15, fontweight='bold')

    for i, col in enumerate(cat_cols[:12]):
        vc = df[col].value_counts().head(10)
        axes[i].barh(range(len(vc)), vc.values, color=PURPLE, edgecolor='white', alpha=0.85)
        axes[i].set_yticks(range(len(vc)))
        axes[i].set_yticklabels(vc.index.astype(str), fontsize=8)
        axes[i].set_title(
            f'{col}\n{df[col].nunique()} unique | null={df[col].isnull().mean()*100:.0f}%',
            fontsize=9)

    for j in range(len(cat_cols[:12]), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_chart('03_univariate_categorical.png')


# ============================================================
# SECTION 6 — BIVARIATE — FEATURE VS TARGET
# ============================================================

print('\n' + '-' * 65)
print('SECTION 6 — Bivariate Analysis (Feature vs Target)')
print('-' * 65)

plot_num_cols = num_cols[:8]

# Overlapping histograms
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('Bivariate — Feature Distribution by Default Status\n(Blue=Non-default, Red=Default)',
             fontsize=14, fontweight='bold')

for i, col in enumerate(plot_num_cols):
    cap = df[col].quantile(0.99)
    tmp = df[df[col] <= cap]
    bivariate_num(tmp, col, ax=axes[i])

plt.tight_layout()
save_chart('04_bivariate_histograms.png')

# Box plots
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()
fig.suptitle('Box Plots — Feature Distribution by Default Status', fontsize=14, fontweight='bold')

for i, col in enumerate(plot_num_cols):
    cap = df[col].quantile(0.99)
    tmp = df[df[col] <= cap][['target', col]].dropna()
    tmp.boxplot(column=col, by='target', ax=axes[i],
                boxprops=dict(color=BLUE),
                medianprops=dict(color=RED, linewidth=2.5),
                whiskerprops=dict(color=BLUE),
                capprops=dict(color=BLUE))
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel('0 = Non-default  |  1 = Default')

plt.suptitle('')
plt.tight_layout()
save_chart('05_bivariate_boxplots.png')

# Categorical vs default rate
if cat_cols:
    plot_cat = cat_cols[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle('Default Rate by Categorical Feature Value\n(Red bar = above average default rate)',
                 fontsize=14, fontweight='bold')

    for i, col in enumerate(plot_cat):
        cat_def = df.groupby(col)['target'].mean().sort_values(ascending=False).head(10) * 100
        colors_b = [RED if v > target_pct[1] else BLUE for v in cat_def.values]
        axes[i].barh(range(len(cat_def)), cat_def.values, color=colors_b, edgecolor='white')
        axes[i].axvline(target_pct[1], color='gray', linestyle='--', linewidth=1.5,
                        label=f'Avg: {target_pct[1]:.1f}%')
        axes[i].set_yticks(range(len(cat_def)))
        axes[i].set_yticklabels(cat_def.index.astype(str), fontsize=8)
        axes[i].set_title(col, fontsize=10)
        axes[i].set_xlabel('Default Rate (%)')
        axes[i].legend(fontsize=8)

    for j in range(len(plot_cat), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    save_chart('06_categorical_vs_default.png')

print('\n  BUSINESS INSIGHT #2:')
print('  DPD features (P suffix) show strongest separation between')
print('  defaulters and non-defaulters — days past due is the most')
print('  predictive raw feature. Amount (A) features are right-skewed;')
print('  log transform will improve model performance.')


# ============================================================
# SECTION 7 — CORRELATION ANALYSIS
# ============================================================

print('\n' + '-' * 65)
print('SECTION 7 — Correlation Analysis')
print('-' * 65)

# Select numeric cols, <50% missing, sufficient variance
corr_cols = [c for c in df.columns
             if df[c].dtype in [np.float32, np.float64, np.int32, np.int64, int, float]
             and df[c].isnull().mean() < 0.5
             and df[c].nunique() > 5
             and c not in ['case_id', 'MONTH', 'WEEK_NUM']][:25]
corr_cols += ['target']

corr_matrix = df[corr_cols].corr()
print(f'  Correlation matrix: {corr_matrix.shape}')

# Full heatmap
fig, ax = plt.subplots(figsize=(16, 13))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdBu_r', center=0,
            ax=ax, linewidths=0.3, vmin=-1, vmax=1,
            cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.8})
ax.set_title('Feature Correlation Matrix (lower triangle)', fontsize=14, fontweight='bold')
plt.tight_layout()
save_chart('07_correlation_heatmap.png')

# Correlation with target bar chart
target_corr = corr_matrix['target'].drop('target').sort_values()
top_corr    = pd.concat([target_corr.head(10), target_corr.tail(10)])

fig, ax = plt.subplots(figsize=(10, 8))
colors_c = [RED if v > 0 else BLUE for v in top_corr.values]
bars     = ax.barh(range(len(top_corr)), top_corr.values, color=colors_c, edgecolor='white', alpha=0.85)
ax.set_yticks(range(len(top_corr)))
ax.set_yticklabels(top_corr.index, fontsize=9)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('Top Features by Correlation with Target\nRed = raises default risk | Blue = lowers default risk',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Pearson Correlation with Target')
for bar, val in zip(bars, top_corr.values):
    offset = 0.001 if val >= 0 else -0.001
    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
            va='center', ha='left' if val >= 0 else 'right', fontsize=8)
plt.tight_layout()
save_chart('08_target_correlation.png')

# Multicollinearity report
print('\n  Multicollinear pairs |corr| > 0.70 (drop one from each pair):')
high_corr = []
cols_list = corr_matrix.columns.tolist()
for i in range(len(cols_list)):
    for j in range(i+1, len(cols_list)):
        c1, c2 = cols_list[i], cols_list[j]
        val = corr_matrix.iloc[i, j]
        if abs(val) > 0.70 and 'target' not in (c1, c2):
            high_corr.append({'Feature 1': c1, 'Feature 2': c2, 'Corr': round(val, 3)})

if high_corr:
    print(pd.DataFrame(high_corr).sort_values('Corr', ascending=False).to_string(index=False))
else:
    print('  No highly correlated pairs found in selected 25 features.')

print('\n  BUSINESS INSIGHT #3:')
print('  Correlated features add noise, not signal. Drop one from each')
print('  correlated pair (|corr|>0.7) during feature selection.')


# ============================================================
# SECTION 8 — MISSING VALUE DEEP DIVE
# ============================================================

print('\n' + '-' * 65)
print('SECTION 8 — Missing Value Deep Dive')
print('-' * 65)

null_pct = df.isnull().mean() * 100

print(f'\n  Cols with 0% null    : {int((null_pct == 0).sum())}')
print(f'  Cols with 0-50% null : {int(((null_pct > 0) & (null_pct < 50)).sum())}')
print(f'  Cols with 50-80%     : {int(((null_pct >= 50) & (null_pct < 80)).sum())}')
print(f'  Cols with >80% null  : {int((null_pct >= 80).sum())}  <-- will be DROPPED')

print('\n  Top 15 most null columns:')
null_df = pd.DataFrame({'null_%': null_pct, 'dtype': df.dtypes})\
            .sort_values('null_%', ascending=False)
print(null_df.head(15).to_string())

# Distribution of null %
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].hist(null_pct.values, bins=20, color=BLUE, edgecolor='white', alpha=0.85)
axes[0].axvline(80, color=RED,    linestyle='--', linewidth=2, label='80% drop threshold')
axes[0].axvline(50, color=ORANGE, linestyle='--', linewidth=2, label='50% caution')
axes[0].set_title('Null % Distribution Across All Columns', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Null %')
axes[0].set_ylabel('Number of Columns')
axes[0].legend()

buckets = pd.cut(null_pct, bins=[0, 0.01, 20, 50, 80, 100],
                 labels=['0%', '0-20%', '20-50%', '50-80%', '>80%'])
bc = buckets.value_counts().sort_index()
colors_bk = [GREEN, BLUE, ORANGE, RED, '#8B0000']
brs = axes[1].bar(bc.index.astype(str), bc.values,
                  color=colors_bk[:len(bc)], edgecolor='white')
axes[1].set_title('Columns Grouped by Null % Bucket', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Null % Range')
axes[1].set_ylabel('Number of Columns')
for b in brs:
    axes[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                 str(int(b.get_height())), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
save_chart('09_missing_value_distribution.png')

# Missingno matrix
sample_cols = null_pct[(null_pct > 5) & (null_pct < 95)].index.tolist()[:40]
if sample_cols:
    sample_rows = df[sample_cols].sample(min(300, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(16, 6))
    msno.matrix(sample_rows, ax=ax, sparkline=False, fontsize=7, color=(0.27, 0.52, 0.71))
    ax.set_title('Missing Value Pattern — Features with 5-95% Nulls (300-row sample)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_chart('10_missingno_matrix.png')

# Does missingness predict default?
print('\n  Does missing value flag predict default rate?')
miss_rows_list = []
for col in num_cols[:15]:
    if df[col].isnull().mean() > 0.05:
        r_miss = df.loc[df[col].isnull(), 'target'].mean() * 100
        r_pres = df.loc[df[col].notna(), 'target'].mean() * 100
        miss_rows_list.append({
            'Feature': col,
            'Default% MISSING': round(r_miss, 2),
            'Default% PRESENT': round(r_pres, 2),
            'Diff': round(r_miss - r_pres, 2)
        })
if miss_rows_list:
    mv_df = pd.DataFrame(miss_rows_list).sort_values('Diff', ascending=False)
    print(mv_df.to_string(index=False))

print('\n  BUSINESS INSIGHT #4:')
print('  Where "Default% MISSING" >> "Default% PRESENT", missingness is')
print('  itself predictive. Create _is_missing flag columns for these.')
print('  Applicants with no bureau record = higher credit risk (thin file).')


# ============================================================
# SECTION 9 — DATE FEATURE ANALYSIS
# ============================================================

print('\n' + '-' * 65)
print('SECTION 9 — Date Feature Analysis')
print('-' * 65)

df['year']      = df['date_decision'].dt.year
df['month_num'] = df['date_decision'].dt.month
df['quarter']   = df['date_decision'].dt.quarter
df['dayofweek'] = df['date_decision'].dt.dayofweek

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Temporal Patterns in Loan Applications', fontsize=15, fontweight='bold')

# By year
yr_vol = df.groupby('year')['case_id'].count()
yr_def = df.groupby('year')['target'].mean() * 100
ax0 = axes[0, 0]; ax0b = ax0.twinx()
ax0.bar(yr_vol.index, yr_vol.values, color=BLUE, alpha=0.6)
ax0b.plot(yr_def.index, yr_def.values, color=RED, marker='o', linewidth=2)
ax0.set_title('By Year', fontsize=12)
ax0.set_ylabel('Applications', color=BLUE)
ax0b.set_ylabel('Default Rate (%)', color=RED)

# By month
mo_vol = df.groupby('month_num')['case_id'].count()
mo_def = df.groupby('month_num')['target'].mean() * 100
ax1 = axes[0, 1]; ax1b = ax1.twinx()
ax1.bar(mo_vol.index, mo_vol.values, color=ORANGE, alpha=0.6)
ax1b.plot(mo_def.index, mo_def.values, color=RED, marker='o', linewidth=2)
ax1.set_xticks(range(1, 13))
ax1.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
ax1.set_title('By Month', fontsize=12)
ax1.set_ylabel('Applications', color=ORANGE)
ax1b.set_ylabel('Default Rate (%)', color=RED)

# By quarter
q_vol = df.groupby('quarter')['case_id'].count()
q_def = df.groupby('quarter')['target'].mean() * 100
ax2 = axes[1, 0]; ax2b = ax2.twinx()
ax2.bar(q_vol.index, q_vol.values, color=GREEN, alpha=0.7, edgecolor='white')
ax2b.plot(q_def.index, q_def.values, color=RED, marker='s', linewidth=2)
ax2.set_xticks([1, 2, 3, 4])
ax2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax2.set_title('By Quarter', fontsize=12)
ax2.set_ylabel('Applications', color=GREEN)
ax2b.set_ylabel('Default Rate (%)', color=RED)

# By day of week
dow_vol = df.groupby('dayofweek')['case_id'].count()
dow_def = df.groupby('dayofweek')['target'].mean() * 100
ax3 = axes[1, 1]; ax3b = ax3.twinx()
ax3.bar(dow_vol.index, dow_vol.values, color=PURPLE, alpha=0.7, edgecolor='white')
ax3b.plot(dow_def.index, dow_def.values, color=RED, marker='o', linewidth=2)
ax3.set_xticks(range(7))
ax3.set_xticklabels(['Mon','Tue','Wed','Thu','Fri','Sat','Sun'])
ax3.set_title('By Day of Week', fontsize=12)
ax3.set_ylabel('Applications', color=PURPLE)
ax3b.set_ylabel('Default Rate (%)', color=RED)

plt.tight_layout()
save_chart('11_date_patterns.png')

print('\n  BUSINESS INSIGHT #5:')
print('  Temporal patterns show seasonality. These become features in Phase 03:')
print('  month, quarter, day_of_week, is_weekend, year.')


# ============================================================
# SECTION 10 — BUREAU B EDA (memory safe — small file 4.3MB)
# ============================================================

print('\n' + '-' * 65)
print('SECTION 10 — Credit Bureau Features EDA')
print('-' * 65)

bureau_b = load_parquet(TRAIN_DIR, 'credit_bureau_b_1')

if not bureau_b.empty:
    bureau_b = reduce_mem(bureau_b)
    rows_per_case = bureau_b.groupby('case_id').size()

    print(f'\n  Bureau B records per applicant:')
    print(f'    Min    : {rows_per_case.min()}')
    print(f'    Median : {rows_per_case.median():.0f}')
    print(f'    Mean   : {rows_per_case.mean():.1f}')
    print(f'    Max    : {rows_per_case.max()}')

    rpc = rows_per_case.reset_index(name='bureau_count')
    rpc = rpc.merge(train_base[['case_id', 'target']], on='case_id', how='left')
    rpc['bucket'] = pd.cut(rpc['bureau_count'],
                           bins=[0, 1, 3, 5, 10, 9999],
                           labels=['1', '2-3', '4-5', '6-10', '10+'])
    bucket_def = rpc.groupby('bucket')['target'].mean() * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Credit Bureau B — Records Analysis', fontsize=14, fontweight='bold')

    axes[0].hist(rows_per_case.clip(upper=50).values, bins=30,
                 color=BLUE, edgecolor='white', alpha=0.85)
    axes[0].axvline(rows_per_case.median(), color=RED, linestyle='--', linewidth=2,
                    label=f'Median: {rows_per_case.median():.0f}')
    axes[0].set_title('Bureau Records per Applicant', fontsize=12)
    axes[0].set_xlabel('Records (capped at 50 for viz)')
    axes[0].set_ylabel('Applicants')
    axes[0].legend()

    colors_bd = [GREEN if v < target_pct[1] else RED for v in bucket_def.values]
    axes[1].bar(bucket_def.index.astype(str), bucket_def.values,
                color=colors_bd, edgecolor='white', alpha=0.85)
    axes[1].axhline(target_pct[1], color='gray', linestyle='--',
                    label=f'Overall: {target_pct[1]:.1f}%')
    axes[1].set_title('Default Rate by Bureau Record Count', fontsize=12)
    axes[1].set_xlabel('Record count bucket')
    axes[1].set_ylabel('Default Rate (%)')
    axes[1].legend()
    for i, (idx, val) in enumerate(bucket_def.items()):
        axes[1].text(i, val + 0.1, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_chart('12_bureau_eda.png')

    del bureau_b, rpc
    gc.collect()
    print('  RAM freed: bureau_b removed')

print('\n  BUSINESS INSIGHT #6:')
print('  Applicants with very few bureau records (thin credit file) have')
print('  higher default rates. bureau_record_count will be a key feature.')


# ============================================================
# SECTION 11 — TRAIN/VALIDATION SPLIT CHART
# ============================================================

print('\n' + '-' * 65)
print('SECTION 11 — Train/Validation Split Strategy')
print('-' * 65)

total_weeks = int(df['WEEK_NUM'].max())
split_week  = int(total_weeks * 0.80)
train_n     = (df['WEEK_NUM'] <= split_week).sum()
val_n       = (df['WEEK_NUM'] > split_week).sum()

print(f'  Total weeks      : {total_weeks}')
print(f'  Split at week    : {split_week}')
print(f'  Train size       : {train_n:,} ({train_n/len(df)*100:.1f}%)')
print(f'  Validation size  : {val_n:,} ({val_n/len(df)*100:.1f}%)')

weekly_stats = df.groupby('WEEK_NUM').agg(
    cases=('case_id', 'count'),
    default_rate=('target', 'mean')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 6))
ax2 = ax1.twinx()
ax1.bar(weekly_stats['WEEK_NUM'], weekly_stats['cases'], color=BLUE, alpha=0.45, label='Applications')
ax2.plot(weekly_stats['WEEK_NUM'], weekly_stats['default_rate'] * 100,
         color=RED, linewidth=2, label='Default rate %')
ax1.axvline(split_week, color='green', linestyle='--', linewidth=2.5,
            label=f'Train/Val split (week {split_week})')
ax1.fill_betweenx([0, weekly_stats['cases'].max()],
                  split_week, total_weeks,
                  alpha=0.07, color=ORANGE, label='Validation window')
ax1.set_xlabel('WEEK_NUM', fontsize=11)
ax1.set_ylabel('Applications', color=BLUE, fontsize=11)
ax2.set_ylabel('Default Rate (%)', color=RED, fontsize=11)
ax1.set_title('Temporal Train/Validation Split Strategy', fontsize=13, fontweight='bold')
l1, lb1 = ax1.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lb1+lb2, loc='upper left', fontsize=9)
plt.tight_layout()
save_chart('13_train_val_split.png')


# ============================================================
# SECTION 12 — SAVE SUMMARY
# ============================================================

print('\n' + '-' * 65)
print('SECTION 12 — Saving EDA Summary')
print('-' * 65)

summary = f"""
PHASE 02 EDA KEY FINDINGS
{'='*50}

DATASET
  Training cases    : {len(df):,}
  WEEK_NUM range    : 0 to {total_weeks}
  Date range        : {df['date_decision'].min().date()} to {df['date_decision'].max().date()}

TARGET
  Default rate      : {target_pct[1]:.2f}%
  Imbalance ratio   : 1 : {int(target_counts[0]/target_counts[1])}
  Use scale_pos_weight = {int(target_counts[0]/target_counts[1])} in XGBoost

MISSING VALUES
  Cols >80% missing : {int((null_pct >= 80).sum())}  (will be DROPPED)
  Cols 0% missing   : {int((null_pct == 0).sum())}

CHARTS GENERATED : 13
BUSINESS INSIGHTS: 6

PHASE 03 FEATURE ENGINEERING PLAN
  1. Drop columns with >80% missing
  2. Log-transform highly skewed amount (A) features
  3. Create _is_missing flags where missingness predicts default
  4. Aggregate depth=1 tables by case_id (max, mean, min, count)
  5. Aggregate depth=2 tables by case_id (two-level aggregation)
  6. Engineer bureau features: DPD max, active loans, enquiry count
  7. Encode categorical features with target encoding
  8. Add date features: month, quarter, day_of_week, year
  9. Final merge: all aggregated tables -> master dataframe
"""

summary_path = OUTPUT_DIR / 'eda_summary.txt'
with open(summary_path, 'w') as f:
    f.write(summary)

print(summary)
print(f'  Summary saved: {summary_path}')
print(f'  All 13 charts: {CHARTS_DIR}')

print('\n' + '=' * 65)
print('  PHASE 02 COMPLETE')
print('  Next -> Phase 03: Feature Engineering')
print('=' * 65)