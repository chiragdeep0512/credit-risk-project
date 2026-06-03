# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 01: Data Understanding & Setup
#  Run this in PyCharm directly — no Jupyter needed
#  Author: Naman Deep Singh
# ============================================================

# ── STEP 0: Install libraries (run once in PyCharm Terminal) ─────────────────
# Open PyCharm Terminal (bottom bar) and paste:
# pip install pandas numpy matplotlib seaborn pyarrow fastparquet missingno tqdm
# -----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import missingno as msno
import os
import glob
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)

plt.rcParams['figure.figsize'] = (12, 5)
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_palette('muted')

print('=' * 60)
print('  PHASE 01 — Data Understanding & Setup')
print('=' * 60)
print('✅ Libraries imported successfully')


# ============================================================
# SECTION 1 — PATH SETUP
# ⚠️  CHANGE DATA_DIR to your dataset folder path
# ============================================================

# Example Windows : Path(r'C:\Users\Naman\Downloads\home-credit-data')
# Example Linux   : Path('/home/naman/datasets/home-credit')
DATA_DIR   = Path(r'D:\Project\Claude Project\credit-risk-project\data\raw\parquet_files\train')
TEST_DIR   = Path(r'D:\Project\Claude Project\credit-risk-project\data\raw\parquet_files\test')
OUTPUT_DIR = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed')


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f'\nDATA_DIR   : {DATA_DIR}')
print(f'OUTPUT_DIR : {OUTPUT_DIR}')

if not DATA_DIR.exists():
    print('\n❌ ERROR: DATA_DIR does not exist — update the path above!')
    exit()
else:
    print('✅ DATA_DIR found!\n')


# ============================================================
# SECTION 2 — UTILITY FUNCTIONS
# ============================================================

def load_table(data_dir: Path, table_name: str, split: str = 'train') -> pd.DataFrame:
    """
    Load a table group (possibly split across multiple files) into one DataFrame.
    Automatically detects parquet or csv.
    """
    pattern_parquet = str(data_dir / f'{split}_{table_name}*.parquet')
    pattern_csv     = str(data_dir / f'{split}_{table_name}*.csv')

    files = sorted(glob.glob(pattern_parquet)) or sorted(glob.glob(pattern_csv))

    if not files:
        print(f'  ⚠️  No files found for: {split}_{table_name} — skipping')
        return pd.DataFrame()

    dfs = []
    for fp in files:
        print(f'    Loading: {Path(fp).name}')
        if fp.endswith('.parquet'):
            dfs.append(pd.read_parquet(fp))
        else:
            dfs.append(pd.read_csv(fp, low_memory=False))

    df = pd.concat(dfs, ignore_index=True)
    print(f'  ✅ {split}_{table_name:<28} shape: {df.shape}  |  files: {len(files)}')
    return df


def reduce_mem_usage(df: pd.DataFrame, name: str = '') -> pd.DataFrame:
    """Downcast numeric columns to save memory."""
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type) != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type).startswith('int'):
                for dtype in [np.int8, np.int16, np.int32, np.int64]:
                    if c_min > np.iinfo(dtype).min and c_max < np.iinfo(dtype).max:
                        df[col] = df[col].astype(dtype)
                        break
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f'  Memory [{name}]: {start_mem:.1f} MB → {end_mem:.1f} MB  '
          f'({100 * (start_mem - end_mem) / start_mem:.1f}% saved)')
    return df


def missing_summary(df: pd.DataFrame, name: str) -> dict:
    null_pct = df.isnull().mean() * 100
    return {
        'Table':             name,
        'Rows':              df.shape[0],
        'Cols':              df.shape[1],
        'Cols 0% null':      int((null_pct == 0).sum()),
        'Cols <50% null':    int(((null_pct > 0) & (null_pct < 50)).sum()),
        'Cols 50-80% null':  int(((null_pct >= 50) & (null_pct < 80)).sum()),
        'Cols >80% null':    int((null_pct >= 80).sum()),
        'Avg null %':        round(null_pct.mean(), 1),
    }


print('✅ Utility functions ready\n')


# ============================================================
# SECTION 3 — FILE INVENTORY
# ============================================================

print('─' * 60)
print('SECTION 3 — File Inventory')
print('─' * 60)

all_files = sorted(DATA_DIR.glob('*.parquet'))
if not all_files:
    all_files = sorted(DATA_DIR.glob('*.csv'))

print(f'Total files found: {len(all_files)}\n')
for i, f in enumerate(all_files, 1):
    size_mb = f.stat().st_size / (1024 * 1024)
    print(f'  {i:2}. {f.name:<55} {size_mb:6.1f} MB')


# ============================================================
# SECTION 4 — LOAD TRAIN BASE (Anchor Table)
# ============================================================

print('\n' + '─' * 60)
print('SECTION 4 — Loading train_base (anchor table)')
print('─' * 60)

train_base = load_table(DATA_DIR, 'base', split='train')
test_base  = load_table(TEST_DIR, 'base', split='test')

print(f'\nTrain base : {train_base.shape}')
print(f'Test  base : {test_base.shape}')
print(f'\nColumns    : {list(train_base.columns)}')
print(f'\nFirst 5 rows:')
print(train_base.head())

# Duplicate check
dup = train_base['case_id'].duplicated().sum()
print(f'\nDuplicate case_ids: {dup}  (expected: 0)')

# Date + week range
#print(f'WEEK_NUM range  (train): {train_base["WEEK_NUM"].min()} → {train_base["WEEK_NUM"].max()}')
#print(f'WEEK_NUM range  (test) : {test_base["WEEK_NUM"].min()} → {test_base["WEEK_NUM"].max()}')
print(f'Date range      (train): {train_base["date_decision"].min()} → {train_base["date_decision"].max()}')


# ============================================================
# SECTION 5 — TARGET / CLASS IMBALANCE ANALYSIS
# ============================================================

print('\n' + '─' * 60)
print('SECTION 5 — Target Variable & Class Imbalance')
print('─' * 60)

target_counts = train_base['target'].value_counts()
target_pct    = train_base['target'].value_counts(normalize=True) * 100

print(f"Non-default (0) : {target_counts[0]:>8,}  ({target_pct[0]:.2f}%)")
print(f"Default     (1) : {target_counts[1]:>8,}  ({target_pct[1]:.2f}%)")
print(f"Imbalance ratio : 1 : {target_counts[0] // target_counts[1]}")
print(f"\n→ Use scale_pos_weight = {target_counts[0] // target_counts[1]} in XGBoost")
print(f"→ Use class_weight='balanced' in Logistic Regression / Random Forest")

# ── Chart 1: Class distribution + weekly default rate ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Home Credit 2023 — Class Imbalance Analysis', fontsize=14)

colors = ['#4A90D9', '#E05252']
bars = axes[0].bar(['Non-Default (0)', 'Default (1)'],
                   target_counts.values, color=colors, width=0.5, edgecolor='white')
axes[0].set_title('Target Class Distribution', fontsize=13, pad=10)
axes[0].set_ylabel('Number of Cases')
for bar, pct in zip(bars, target_pct.values):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{pct:.1f}%', ha='center', fontsize=12, fontweight='bold')
axes[0].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:,.0f}'))

weekly_default = train_base.groupby('WEEK_NUM')['target'].mean() * 100
axes[1].plot(weekly_default.index, weekly_default.values, color='#E05252', linewidth=2, marker='o', markersize=3)
axes[1].axhline(y=target_pct[1], color='gray', linestyle='--', alpha=0.7,
                label=f'Overall avg: {target_pct[1]:.1f}%')
axes[1].fill_between(weekly_default.index, weekly_default.values, alpha=0.1, color='#E05252')
axes[1].set_title('Default Rate by Week Number', fontsize=13, pad=10)
axes[1].set_xlabel('WEEK_NUM')
axes[1].set_ylabel('Default Rate (%)')
axes[1].legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'target_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Chart saved: target_distribution.png')


# ============================================================
# SECTION 6 — LOAD ALL TABLE GROUPS
# ============================================================

print('\n' + '─' * 60)
print('SECTION 6 — Loading all table groups')
print('─' * 60)

print('\n[depth=0 tables]')
train_static_0  = load_table(DATA_DIR, 'static_0',    split='train')
train_static_cb = load_table(DATA_DIR, 'static_cb_0', split='train')

print('\n[depth=1 tables]')
train_applprev_1    = load_table(DATA_DIR, 'applprev_1',         split='train')
train_person_1      = load_table(DATA_DIR, 'person_1',           split='train')
train_credit_bur_a1 = load_table(DATA_DIR, 'credit_bureau_a_1',  split='train')
train_credit_bur_b1 = load_table(DATA_DIR, 'credit_bureau_b_1',  split='train')
train_deposit_1     = load_table(DATA_DIR, 'deposit_1',          split='train')
train_debitcard_1   = load_table(DATA_DIR, 'debitcard_1',        split='train')
train_other_1       = load_table(DATA_DIR, 'other_1',            split='train')
train_tax_a         = load_table(DATA_DIR, 'tax_registry_a_1',   split='train')
train_tax_b         = load_table(DATA_DIR, 'tax_registry_b_1',   split='train')
train_tax_c         = load_table(DATA_DIR, 'tax_registry_c_1',   split='train')

print('\n[depth=2 tables — largest files, may take 5-10 mins]')
train_credit_bur_a2 = load_table(DATA_DIR, 'credit_bureau_a_2',  split='train')
train_credit_bur_b2 = load_table(DATA_DIR, 'credit_bureau_b_2',  split='train')
train_applprev_2    = load_table(DATA_DIR, 'applprev_2',          split='train')
train_person_2      = load_table(DATA_DIR, 'person_2',            split='train')


# ============================================================
# SECTION 7 — MASTER SHAPE INVENTORY
# ============================================================

print('\n' + '─' * 60)
print('SECTION 7 — Master Shape Inventory')
print('─' * 60)

tables = {
    'train_base':              train_base,
    'train_static_0':          train_static_0,
    'train_static_cb_0':       train_static_cb,
    'train_applprev_1':        train_applprev_1,
    'train_person_1':          train_person_1,
    'train_credit_bureau_a_1': train_credit_bur_a1,
    'train_credit_bureau_b_1': train_credit_bur_b1,
    'train_deposit_1':         train_deposit_1,
    'train_debitcard_1':       train_debitcard_1,
    'train_other_1':           train_other_1,
    'train_tax_registry_a':    train_tax_a,
    'train_tax_registry_b':    train_tax_b,
    'train_tax_registry_c':    train_tax_c,
    'train_credit_bureau_a_2': train_credit_bur_a2,
    'train_credit_bureau_b_2': train_credit_bur_b2,
    'train_applprev_2':        train_applprev_2,
    'train_person_2':          train_person_2,
}

shape_rows = []
for name, df in tables.items():
    if df.empty:
        continue
    shape_rows.append({
        'Table':    name,
        'Rows':     f'{df.shape[0]:,}',
        'Columns':  df.shape[1],
        'Mem(MB)':  f'{df.memory_usage(deep=True).sum() / 1024 ** 2:.1f}',
    })

shape_df = pd.DataFrame(shape_rows)
print(shape_df.to_string(index=False))


# ============================================================
# SECTION 8 — DEPTH CONCEPT VERIFICATION
# ============================================================

print('\n' + '─' * 60)
print('SECTION 8 — Depth Concept Verification')
print('─' * 60)

if not train_applprev_1.empty:
    rows_per_case = train_applprev_1.groupby('case_id').size()
    print(f'\napplprev_1 (depth=1) — rows per case_id:')
    print(rows_per_case.describe().to_string())
    print(f'\nMax previous applications for 1 applicant: {rows_per_case.max()}')
    print('→ We MUST aggregate this before joining to base table')

if not train_credit_bur_a2.empty:
    rows_per_case2 = train_credit_bur_a2.groupby(['case_id', 'num_group1']).size()
    print(f'\ncredit_bureau_a_2 (depth=2) — rows per case_id+num_group1:')
    print(rows_per_case2.describe().to_string())


# ============================================================
# SECTION 9 — MISSING VALUE ANALYSIS
# ============================================================

print('\n' + '─' * 60)
print('SECTION 9 — Missing Value Analysis')
print('─' * 60)

miss_rows = [missing_summary(df, name) for name, df in tables.items() if not df.empty]
miss_df = pd.DataFrame(miss_rows)
print(miss_df.to_string(index=False))

# Missing value heatmap — static_0 sample
if not train_static_0.empty:
    sample = train_static_0.sample(min(200, len(train_static_0)), random_state=42)
    cols_with_missing = sample.columns[sample.isnull().any()].tolist()[:60]
    if cols_with_missing:
        fig, ax = plt.subplots(figsize=(14, 5))
        msno.matrix(sample[cols_with_missing], ax=ax, sparkline=False,
                    fontsize=7, color=(0.27, 0.52, 0.71))
        ax.set_title('Missing Value Pattern — static_0 (200-row sample, columns with nulls)',
                     fontsize=12, pad=10)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'missing_values_static0.png', dpi=150, bbox_inches='tight')
        plt.show()
        print('✅ Chart saved: missing_values_static0.png')


# ============================================================
# SECTION 10 — TEMPORAL SPLIT STRATEGY
# ============================================================

print('\n' + '─' * 60)
print('SECTION 10 — Temporal Split Strategy')
print('─' * 60)

total_weeks = int(train_base['WEEK_NUM'].max())
split_week  = int(total_weeks * 0.80)

train_count = (train_base['WEEK_NUM'] <= split_week).sum()
val_count   = (train_base['WEEK_NUM'] >  split_week).sum()

print(f'Total WEEK_NUMs  : {total_weeks}')
print(f'Split at week    : {split_week}')
print(f'Train cases      : {train_count:,}  (weeks 0–{split_week})')
print(f'Validation cases : {val_count:,}  (weeks {split_week+1}–{total_weeks})')

weekly_cases = train_base.groupby('WEEK_NUM').agg(
    total_cases=('case_id', 'count'),
    default_rate=('target', 'mean')
).reset_index()

fig, ax1 = plt.subplots(figsize=(13, 5))
ax2 = ax1.twinx()
ax1.bar(weekly_cases['WEEK_NUM'], weekly_cases['total_cases'],
        color='#4A90D9', alpha=0.5, label='Cases per week')
ax2.plot(weekly_cases['WEEK_NUM'], weekly_cases['default_rate'] * 100,
         color='#E05252', linewidth=2, label='Default rate %')
ax1.axvline(x=split_week, color='green', linestyle='--', linewidth=2,
            label=f'Train/Val split (week {split_week})')
ax1.fill_betweenx([0, weekly_cases['total_cases'].max()],
                  split_week, total_weeks, alpha=0.08, color='orange', label='Validation window')
ax1.set_xlabel('WEEK_NUM')
ax1.set_ylabel('Number of loan applications', color='#4A90D9')
ax2.set_ylabel('Default Rate (%)', color='#E05252')
ax1.set_title('Temporal Distribution — Train/Validation Split Strategy', fontsize=13, pad=10)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'temporal_split.png', dpi=150, bbox_inches='tight')
plt.show()
print('✅ Chart saved: temporal_split.png')


# ============================================================
# SECTION 11 — MEMORY OPTIMIZATION & SAVE
# ============================================================

print('\n' + '─' * 60)
print('SECTION 11 — Memory Optimization & Save')
print('─' * 60)

if not train_static_0.empty:
    train_static_0 = reduce_mem_usage(train_static_0, 'static_0')
if not train_credit_bur_a1.empty:
    train_credit_bur_a1 = reduce_mem_usage(train_credit_bur_a1, 'credit_bureau_a_1')

print('\nSaving processed files...')
train_base.to_parquet(OUTPUT_DIR / 'train_base.parquet', index=False)
test_base.to_parquet(OUTPUT_DIR  / 'test_base.parquet',  index=False)
if not train_static_0.empty:
    train_static_0.to_parquet(OUTPUT_DIR / 'train_static_0.parquet', index=False)
if not train_static_cb.empty:
    train_static_cb.to_parquet(OUTPUT_DIR / 'train_static_cb_0.parquet', index=False)
print('✅ Files saved to:', OUTPUT_DIR)


# ============================================================
# FINAL SUMMARY
# ============================================================

print('\n' + '=' * 60)
print('  PHASE 01 COMPLETE ✅')
print('=' * 60)
print(f'  Total tables loaded        : {len([d for d in tables.values() if not d.empty])}')
print(f'  Training cases             : {len(train_base):,}')
print(f'  Test cases                 : {len(test_base):,}')
print(f'  Default rate               : {train_base["target"].mean() * 100:.2f}%')
print(f'  Imbalance ratio (neg:pos)  : {target_counts[0] // target_counts[1]} : 1')
print(f'  Recommended val split      : WEEK_NUM > {split_week}')
print(f'  Charts saved to            : {OUTPUT_DIR}')
print()
print('  Next → Phase 02: EDA')
print('=' * 60)