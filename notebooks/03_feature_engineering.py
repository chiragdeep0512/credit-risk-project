# ============================================================
#  CREDIT RISK SCORING — Home Credit 2023
#  Phase 03: Feature Engineering (8GB RAM OPTIMIZED)
#  Author: Naman Deep Singh
#  GitHub: github.com/chiragdeep0512
# ============================================================
#
#  CHANGES FROM PREVIOUS VERSION:
#  - Selective column loading (only useful cols, not full table)
#  - Polars NOT required — pure pandas but smarter
#  - Each table: load only needed cols -> aggregate -> merge -> delete
#  - bureau_a_1 & a_2: only DPD + Amount cols loaded (not all 100+ cols)
#  - person_1: only 8 key cols loaded (not all 37)
#  - Expected time: 30-45 minutes on 8GB RAM
#  - All outputs saved to D: drive
#
#  INSTALL (run once in PyCharm Terminal):
#  pip install pandas numpy pyarrow scikit-learn tqdm
# ============================================================

import pandas as pd
import numpy as np
import glob
import gc
import json
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)

print('=' * 65)
print('  PHASE 03 — Feature Engineering (8GB RAM Optimized)')
print('=' * 65)


# ============================================================
# SECTION 0 — PATHS  ← CHANGE HERE IF NEEDED
# ============================================================

TRAIN_DIR  = Path(r'D:\Project\Claude Project\credit-risk-project\data\raw\parquet_files\train')

# ── Output saved to D: drive ──────────────────────────────────
OUTPUT_DIR = Path(r'D:\Project\Claude Project\credit-risk-project\data\processed')

# If you want F: drive instead, change to:
# OUTPUT_DIR = Path(r'F:\credit-risk-processed')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Output will be saved to: {OUTPUT_DIR}\n')


# ============================================================
# SECTION 1 — UTILITY FUNCTIONS
# ============================================================

def mem_mb(df):
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_mem(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast all numeric columns to smallest possible type."""
    for col in df.columns:
        ct = str(df[col].dtype)
        if 'int' in ct:
            cmin, cmax = df[col].min(), df[col].max()
            for dt in [np.int8, np.int16, np.int32]:
                if cmin > np.iinfo(dt).min and cmax < np.iinfo(dt).max:
                    df[col] = df[col].astype(dt)
                    break
        elif 'float' in ct:
            cmin, cmax = df[col].min(), df[col].max()
            if cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df


def get_parquet_columns(filepath: str) -> list:
    """Read only column names without loading data."""
    import pyarrow.parquet as pq
    return pq.read_schema(filepath).names


def smart_load(table_name: str,
               usecols: list = None,
               split: str = 'train') -> pd.DataFrame:
    """
    Load a table group. If usecols given, load ONLY those columns.
    This is the key RAM saving technique.
    """
    pattern = str(TRAIN_DIR / f'{split}_{table_name}*.parquet')
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f'  [SKIP] {split}_{table_name} — no files found')
        return pd.DataFrame()

    dfs = []
    for fp in files:
        if usecols:
            # Load only specified columns
            available = get_parquet_columns(fp)
            load_cols = [c for c in usecols if c in available]
            dfs.append(pd.read_parquet(fp, columns=load_cols))
        else:
            dfs.append(pd.read_parquet(fp))

    df = pd.concat(dfs, ignore_index=True)
    df = reduce_mem(df)
    mb = mem_mb(df)
    print(f'  [OK] {split}_{table_name:<28} shape: {str(df.shape):<20} {mb:.0f} MB')
    return df


def safe_agg(df: pd.DataFrame,
             group_col: str,
             num_cols: list,
             cat_cols: list,
             prefix: str) -> pd.DataFrame:
    """
    Aggregate numeric with mean/max/min/sum.
    Aggregate categorical with nunique only (fast).
    """
    agg_dict = {}
    for c in num_cols:
        if c in df.columns:
            agg_dict[c] = ['mean', 'max', 'min', 'sum']
    for c in cat_cols:
        if c in df.columns:
            agg_dict[c] = ['nunique']

    if not agg_dict:
        return pd.DataFrame()

    agg = df.groupby(group_col).agg(agg_dict)
    agg.columns = [f'{prefix}{c}_{f}' for c, f in agg.columns]
    agg = agg.reset_index()

    # Add record count
    counts = df.groupby(group_col).size().reset_index(name=f'{prefix}count')
    agg = agg.merge(counts, on=group_col, how='left')
    agg = reduce_mem(agg)
    return agg


def merge_to_master(master: pd.DataFrame,
                    agg_df: pd.DataFrame,
                    name: str) -> pd.DataFrame:
    if agg_df.empty:
        return master
    master = master.merge(agg_df, on='case_id', how='left')
    print(f'    master after {name}: {master.shape}  |  {mem_mb(master):.0f} MB')
    return master


print('Utility functions ready\n')


# ============================================================
# STEP 1 — LOAD TRAIN BASE
# ============================================================

print('-' * 65)
print('STEP 1 — Load train_base')
print('-' * 65)

master = smart_load('base', split='train')
master['date_decision'] = pd.to_datetime(master['date_decision'])
print(f'  master initialized: {master.shape}\n')


# ============================================================
# STEP 2 — STATIC_0  (depth=0)
# ============================================================

print('-' * 65)
print('STEP 2 — static_0  (depth=0, internal)')
print('-' * 65)

# ── Find which columns exist and are useful ───────────────────
s0_files = sorted(glob.glob(str(TRAIN_DIR / 'train_static_0*.parquet')))
all_s0_cols = get_parquet_columns(s0_files[0])

# Load everything EXCEPT columns that are almost certainly junk:
# Date columns (D suffix) — too many, high null, we engineer dates from date_decision
# Keep: Amount (A), DPD (P), categorical (M), transform (T/L)
keep_s0 = ['case_id'] + [c for c in all_s0_cols
                          if not c.endswith('D')        # skip raw date cols
                          and c != 'case_id'][:80]      # cap at 80 cols

print(f'  Selective load: {len(keep_s0)} cols (from {len(all_s0_cols)} total)')

static_0 = smart_load('static_0', usecols=keep_s0)

# Drop still-high-null cols
null_pct = static_0.isnull().mean()
drop_cols = null_pct[null_pct > 0.75].index.tolist()
static_0  = static_0.drop(columns=drop_cols)
print(f'  Dropped {len(drop_cols)} cols >75% null. Shape: {static_0.shape}')

# Impute numeric with median
for c in static_0.select_dtypes(include=[np.number]).columns:
    if static_0[c].isnull().any():
        static_0[c] = static_0[c].fillna(static_0[c].median())

# Frequency-encode categoricals inline
for c in static_0.select_dtypes(include='object').columns:
    freq = static_0[c].value_counts(normalize=True)
    static_0[c] = static_0[c].map(freq).astype(np.float32)

static_0 = reduce_mem(static_0)

# Rename with prefix
static_0.columns = ['case_id'] + [f's0_{c}' for c in static_0.columns if c != 'case_id']
master = merge_to_master(master, static_0, 'static_0')
del static_0; gc.collect()


# ============================================================
# STEP 3 — STATIC_CB_0  (depth=0, external bureau)
# ============================================================

print('\n' + '-' * 65)
print('STEP 3 — static_cb_0  (depth=0, external)')
print('-' * 65)

cb0_files  = sorted(glob.glob(str(TRAIN_DIR / 'train_static_cb_0*.parquet')))
all_cb_cols= get_parquet_columns(cb0_files[0])
keep_cb    = ['case_id'] + [c for c in all_cb_cols if c != 'case_id'][:40]

static_cb = smart_load('static_cb_0', usecols=keep_cb)

null_pct_cb = static_cb.isnull().mean()
drop_cb     = null_pct_cb[null_pct_cb > 0.75].index.tolist()
static_cb   = static_cb.drop(columns=drop_cb)

for c in static_cb.select_dtypes(include=[np.number]).columns:
    if static_cb[c].isnull().any():
        static_cb[c] = static_cb[c].fillna(static_cb[c].median())

for c in static_cb.select_dtypes(include='object').columns:
    freq = static_cb[c].value_counts(normalize=True)
    static_cb[c] = static_cb[c].map(freq).astype(np.float32)

static_cb = reduce_mem(static_cb)
static_cb.columns = ['case_id'] + [f'cb0_{c}' for c in static_cb.columns if c != 'case_id']
master = merge_to_master(master, static_cb, 'static_cb_0')
del static_cb; gc.collect()


# ============================================================
# STEP 4 — PERSON_1  (depth=1) — SELECTIVE COLS ONLY
# ============================================================

print('\n' + '-' * 65)
print('STEP 4 — person_1  (depth=1, selective columns)')
print('-' * 65)

# Only load the 8 most useful person columns — NOT all 37
# This drops RAM from 3.4GB -> ~300MB for this table
person_key_cols = [
    'case_id',
    'num_group1',            # 0 = applicant, >0 = co-applicants
    'birth_259D',            # age proxy
    'education_1138M',       # education level
    'empl_employedtotal_800L',  # employment status
    'empl_industry_691L',    # industry
    'familystate_726L',      # family status
    'housetype_905L',        # house ownership
    'incometype_1044T',      # income type
    'mainoccupationinc_384A' # main occupation income (Amount)
]

person_1 = smart_load('person_1', usecols=person_key_cols)

if not person_1.empty:
    # Applicant row only (num_group1 == 0) for categorical features
    applicant = person_1[person_1['num_group1'] == 0].copy()
    applicant = applicant.drop(columns=['num_group1'])

    # Frequency encode categoricals
    for c in applicant.select_dtypes(include='object').columns:
        freq = applicant[c].value_counts(normalize=True)
        applicant[c] = applicant[c].map(freq).astype(np.float32)

    # Impute
    for c in applicant.select_dtypes(include=[np.number]).columns:
        if c != 'case_id' and applicant[c].isnull().any():
            applicant[c] = applicant[c].fillna(applicant[c].median())

    applicant = reduce_mem(applicant)
    applicant.columns = ['case_id'] + [f'per1_{c}' for c in applicant.columns if c != 'case_id']

    # Count of all persons per case (co-applicants etc.)
    person_count = person_1.groupby('case_id').size().reset_index(name='per1_total_persons')

    applicant = applicant.merge(person_count, on='case_id', how='left')
    master = merge_to_master(master, applicant, 'person_1')
    del person_1, applicant, person_count; gc.collect()


# ============================================================
# STEP 5 — APPLPREV_1  (depth=1) — SELECTIVE COLS
# ============================================================

print('\n' + '-' * 65)
print('STEP 5 — applprev_1  (depth=1, previous applications)')
print('-' * 65)

ap1_files   = sorted(glob.glob(str(TRAIN_DIR / 'train_applprev_1*.parquet')))
all_ap1_cols= get_parquet_columns(ap1_files[0])

# Load Amount (A) and DPD (P) cols + key categoricals
ap1_keep = ['case_id', 'num_group1'] + \
           [c for c in all_ap1_cols if c.endswith('A') or c.endswith('P')][:20] + \
           [c for c in all_ap1_cols if c.endswith('M')][:5]

applprev_1 = smart_load('applprev_1', usecols=list(dict.fromkeys(ap1_keep)))

if not applprev_1.empty:
    num_ap = [c for c in applprev_1.columns
              if c not in ['case_id','num_group1']
              and applprev_1[c].dtype in [np.float32, np.float64, np.int8, np.int16, np.int32]]
    cat_ap = [c for c in applprev_1.columns
              if c not in ['case_id','num_group1']
              and applprev_1[c].dtype == object]

    ap1_agg = safe_agg(applprev_1, 'case_id', num_ap, cat_ap, prefix='ap1_')
    master  = merge_to_master(master, ap1_agg, 'applprev_1')
    del applprev_1, ap1_agg; gc.collect()


# ============================================================
# STEP 6 — SMALL DEPTH=1 TABLES (fast, all at once)
# ============================================================

print('\n' + '-' * 65)
print('STEP 6 — Small depth=1 tables')
print('-' * 65)

small_tables = {
    'deposit_1':         'dep1_',
    'debitcard_1':       'deb1_',
    'other_1':           'oth1_',
    'tax_registry_a_1':  'taxa_',
    'tax_registry_b_1':  'taxb_',
    'tax_registry_c_1':  'taxc_',
}

for tname, prefix in small_tables.items():
    print(f'\n  [{tname}]')
    df = smart_load(tname)
    if df.empty:
        continue

    num_c = [c for c in df.columns
             if c not in ['case_id','num_group1','num_group2']
             and df[c].dtype in [np.float32, np.float64,
                                  np.int8, np.int16, np.int32, np.int64]]
    cat_c = [c for c in df.columns
             if c not in ['case_id','num_group1','num_group2']
             and df[c].dtype == object]

    agg = safe_agg(df, 'case_id', num_c, cat_c, prefix=prefix)
    master = merge_to_master(master, agg, tname)
    del df, agg; gc.collect()


# ============================================================
# STEP 7 — CREDIT BUREAU B_1  (depth=1, small 4.3MB)
# ============================================================

print('\n' + '-' * 65)
print('STEP 7 — credit_bureau_b_1  (depth=1, external)')
print('-' * 65)

bur_b1 = smart_load('credit_bureau_b_1')
if not bur_b1.empty:
    num_bb = [c for c in bur_b1.columns
              if c not in ['case_id','num_group1']
              and bur_b1[c].dtype in [np.float32, np.float64,
                                       np.int8, np.int16, np.int32]]
    cat_bb = [c for c in bur_b1.columns
              if c not in ['case_id','num_group1']
              and bur_b1[c].dtype == object]

    bb1_agg = safe_agg(bur_b1, 'case_id', num_bb, cat_bb, prefix='burb1_')
    master  = merge_to_master(master, bb1_agg, 'bureau_b_1')
    del bur_b1, bb1_agg; gc.collect()


# ============================================================
# STEP 8 — CREDIT BUREAU A_1 (heavy, 4 files)
#          KEY TRICK: load ONLY DPD (P) + Amount (A) cols
# ============================================================

print('\n' + '-' * 65)
print('STEP 8 — credit_bureau_a_1  (depth=1, 4 files, selective)')
print('-' * 65)

bur_a1_files = sorted(glob.glob(str(TRAIN_DIR / 'train_credit_bureau_a_1*.parquet')))
print(f'  Files: {len(bur_a1_files)}')

# Check what cols exist
sample_cols_a1 = get_parquet_columns(bur_a1_files[0])
dpd_cols_a1    = [c for c in sample_cols_a1 if c.endswith('P')]
amt_cols_a1    = [c for c in sample_cols_a1 if c.endswith('A')][:10]
keep_a1        = ['case_id'] + dpd_cols_a1 + amt_cols_a1
print(f'  Selective: loading {len(keep_a1)} cols (DPD + Amount only)')

chunk_aggs = []
for fp in bur_a1_files:
    fname = Path(fp).name
    print(f'  Processing: {fname}')

    available = get_parquet_columns(fp)
    load_cols  = [c for c in keep_a1 if c in available]
    chunk = pd.read_parquet(fp, columns=load_cols)
    chunk = reduce_mem(chunk)

    num_a1 = [c for c in chunk.columns
               if c != 'case_id'
               and chunk[c].dtype in [np.float32, np.float64,
                                       np.int8, np.int16, np.int32]]
    if not num_a1:
        del chunk; gc.collect()
        continue

    agg = chunk.groupby('case_id')[num_a1].agg(['max', 'mean', 'sum'])
    agg.columns = [f'bura1_{c}_{f}' for c, f in agg.columns]
    agg = agg.reset_index()
    agg = reduce_mem(agg)
    chunk_aggs.append(agg)
    del chunk, agg; gc.collect()

if chunk_aggs:
    combined = chunk_aggs[0]
    for nxt in chunk_aggs[1:]:
        # Merge overlapping case_ids — take max across files
        combined = pd.concat([combined, nxt], ignore_index=True)
        num_c = [c for c in combined.columns if c != 'case_id']
        combined = combined.groupby('case_id')[num_c].max().reset_index()
        combined = reduce_mem(combined)

    # Add total record count
    count_list = []
    for fp in bur_a1_files:
        tmp = pd.read_parquet(fp, columns=['case_id'])
        count_list.append(tmp)
    all_counts  = pd.concat(count_list, ignore_index=True)
    total_count = all_counts.groupby('case_id').size().reset_index(name='bura1_count')
    combined = combined.merge(total_count, on='case_id', how='left')

    master = merge_to_master(master, combined, 'bureau_a_1')
    del combined, chunk_aggs, count_list, all_counts, total_count; gc.collect()


# ============================================================
# STEP 9 — DEPTH=2 TABLES (small ones first)
# ============================================================

print('\n' + '-' * 65)
print('STEP 9 — Depth=2 tables (small: bureau_b_2, applprev_2, person_2)')
print('-' * 65)

def quick_depth2(table_name: str, prefix: str,
                 usecols: list = None) -> pd.DataFrame:
    """Load depth=2, do two-level aggregation, return ready-to-merge df."""
    df = smart_load(table_name, usecols=usecols)
    if df.empty:
        return pd.DataFrame()

    num_c = [c for c in df.columns
             if c not in ['case_id', 'num_group1', 'num_group2']
             and df[c].dtype in [np.float32, np.float64,
                                  np.int8, np.int16, np.int32, np.int64]]
    if not num_c:
        return pd.DataFrame()

    # Level 1: collapse num_group2
    lvl1 = df.groupby(['case_id', 'num_group1'])[num_c].agg(['mean', 'max'])
    lvl1.columns = [f'{c}_{f}' for c, f in lvl1.columns]
    lvl1 = lvl1.reset_index()
    lvl1 = reduce_mem(lvl1)
    del df; gc.collect()

    # Level 2: collapse num_group1
    lvl1_num = [c for c in lvl1.columns if c not in ['case_id', 'num_group1']]
    lvl2 = lvl1.groupby('case_id')[lvl1_num].agg(['mean', 'max'])
    lvl2.columns = [f'{prefix}{c}_{f}' for c, f in lvl2.columns]
    lvl2 = lvl2.reset_index()
    lvl2 = reduce_mem(lvl2)
    del lvl1; gc.collect()
    return lvl2


# bureau_b_2 (2MB — tiny)
print('\n  [credit_bureau_b_2]')
bb2 = quick_depth2('credit_bureau_b_2', prefix='burb2_')
master = merge_to_master(master, bb2, 'bureau_b_2')
del bb2; gc.collect()

# applprev_2 (27MB)
print('\n  [applprev_2]')
ap2 = quick_depth2('applprev_2', prefix='ap2_')
master = merge_to_master(master, ap2, 'applprev_2')
del ap2; gc.collect()

# person_2 (7MB)
print('\n  [person_2]')
p2 = quick_depth2('person_2', prefix='p2_')
master = merge_to_master(master, p2, 'person_2')
del p2; gc.collect()


# ============================================================
# STEP 10 — CREDIT BUREAU A_2  (heaviest — 11 files)
#           Chunk by chunk, level-1 agg only per chunk
# ============================================================

print('\n' + '-' * 65)
print('STEP 10 — credit_bureau_a_2  (depth=2, 11 files, chunk mode)')
print('-' * 65)

bur_a2_files = sorted(glob.glob(str(TRAIN_DIR / 'train_credit_bureau_a_2*.parquet')))
print(f'  Files: {len(bur_a2_files)}')

# Only load DPD + Amount cols
sample_cols_a2 = get_parquet_columns(bur_a2_files[0])
dpd_a2  = [c for c in sample_cols_a2 if c.endswith('P')]
amt_a2  = [c for c in sample_cols_a2 if c.endswith('A')][:8]
keep_a2 = ['case_id', 'num_group1'] + dpd_a2 + amt_a2
if 'num_group2' in sample_cols_a2:
    keep_a2 = ['case_id', 'num_group1', 'num_group2'] + dpd_a2 + amt_a2

print(f'  Selective: {len(keep_a2)} cols (DPD + Amount only)')

lvl1_chunks = []
for fp in bur_a2_files:
    fname = Path(fp).name
    print(f'  Processing: {fname}')

    available = get_parquet_columns(fp)
    load_cols  = [c for c in keep_a2 if c in available]
    chunk = pd.read_parquet(fp, columns=load_cols)
    chunk = reduce_mem(chunk)

    feat_c = [c for c in chunk.columns
               if c not in ['case_id', 'num_group1', 'num_group2']
               and chunk[c].dtype in [np.float32, np.float64,
                                       np.int8, np.int16, np.int32]]
    if not feat_c:
        del chunk; gc.collect()
        continue

    grp_cols = ['case_id', 'num_group1']
    grp_cols = [c for c in grp_cols if c in chunk.columns]

    lvl1 = chunk.groupby(grp_cols)[feat_c].agg(['mean', 'max'])
    lvl1.columns = [f'{c}_{f}' for c, f in lvl1.columns]
    lvl1 = lvl1.reset_index()
    lvl1 = reduce_mem(lvl1)
    lvl1_chunks.append(lvl1)

    del chunk, lvl1; gc.collect()

if lvl1_chunks:
    print('  Combining chunks for level-2 aggregation...')
    all_lvl1 = pd.concat(lvl1_chunks, ignore_index=True)
    del lvl1_chunks; gc.collect()

    all_lvl1 = reduce_mem(all_lvl1)
    feat_cols = [c for c in all_lvl1.columns
                 if c not in ['case_id', 'num_group1']]

    # Level 2: group by case_id
    lvl2 = all_lvl1.groupby('case_id')[feat_cols].agg(['mean', 'max'])
    lvl2.columns = [f'bura2_{c}_{f}' for c, f in lvl2.columns]
    lvl2 = lvl2.reset_index()
    lvl2 = reduce_mem(lvl2)

    master = merge_to_master(master, lvl2, 'bureau_a_2')
    del all_lvl1, lvl2; gc.collect()


# ============================================================
# STEP 11 — POST MERGE FEATURE ENGINEERING
# ============================================================

print('\n' + '-' * 65)
print('STEP 11 — Post-Merge Feature Engineering')
print('-' * 65)
print(f'  Master shape before engineering: {master.shape}')

# ── Date features ─────────────────────────────────────────────
print('\n  [11a] Date features')
master['fe_year']       = master['date_decision'].dt.year.astype(np.int16)
master['fe_month']      = master['date_decision'].dt.month.astype(np.int8)
master['fe_quarter']    = master['date_decision'].dt.quarter.astype(np.int8)
master['fe_dayofweek']  = master['date_decision'].dt.dayofweek.astype(np.int8)
master['fe_is_weekend'] = (master['date_decision'].dt.dayofweek >= 5).astype(np.int8)
master['fe_dayofmonth'] = master['date_decision'].dt.day.astype(np.int8)
print('    Added: year, month, quarter, dayofweek, is_weekend, dayofmonth')

# ── Log transform skewed amount features ──────────────────────
print('\n  [11b] Log transform skewed amount features')
amt_cols_master = [c for c in master.columns
                   if ('_A_' in c or c.endswith('_A'))
                   and master[c].dtype in [np.float32, np.float64]
                   and master[c].notna().sum() > 1000]
log_count = 0
for col in amt_cols_master[:30]:
    try:
        if master[col].min() >= 0 and abs(master[col].skew()) > 2:
            master[f'{col}_log'] = np.log1p(master[col]).astype(np.float32)
            log_count += 1
    except Exception:
        continue
print(f'    Log-transformed {log_count} features')

# ── Ratio features ────────────────────────────────────────────
print('\n  [11c] Business ratio features')

# Find best income + loan + credit columns
inc_cols  = [c for c in master.columns if 'income' in c.lower()
             and master[c].notna().sum() > 10000]
loan_cols = [c for c in master.columns if 'amount' in c.lower()
             and master[c].notna().sum() > 10000]

if inc_cols and loan_cols:
    inc_c  = inc_cols[0]
    loan_c = loan_cols[0]
    safe_i = master[inc_c].replace(0, np.nan)
    master['fe_loan_to_income'] = (master[loan_c] / safe_i).astype(np.float32)
    print(f'    loan_to_income: {loan_c} / {inc_c}')
else:
    print('    Income or loan col not found — skipping ratio')

# Bureau count ratio
if 'bura1_count' in master.columns and 'ap1_count' in master.columns:
    safe_ap = master['ap1_count'].replace(0, np.nan)
    master['fe_bureau_per_app'] = (master['bura1_count'] / safe_ap).astype(np.float32)
    print('    bureau_per_app added')

# ── Final null fill ───────────────────────────────────────────
print('\n  [11d] Final null imputation')
feat_cols = [c for c in master.columns
             if c not in ['case_id', 'date_decision', 'target']]
for col in feat_cols:
    if master[col].isnull().any():
        if master[col].dtype in [np.float32, np.float64]:
            master[col] = master[col].fillna(master[col].median())
        elif master[col].dtype in [np.int8, np.int16, np.int32, np.int64]:
            master[col] = master[col].fillna(0)

remaining = master[feat_cols].isnull().sum().sum()
print(f'    Remaining nulls: {remaining}')

# ── Drop date/period cols not needed ─────────────────────────
drop_final = [c for c in ['date_decision', 'MONTH', 'year_month']
              if c in master.columns]
master = master.drop(columns=drop_final)
print(f'\n  Dropped non-feature cols: {drop_final}')

# ── Final reduce_mem ──────────────────────────────────────────
master = reduce_mem(master)


# ============================================================
# STEP 12 — SUMMARY & SAVE
# ============================================================

print('\n' + '-' * 65)
print('STEP 12 — Summary & Save')
print('-' * 65)

all_feat_cols = [c for c in master.columns if c not in ['case_id', 'target']]
target        = master['target']
total_weeks   = int(master['WEEK_NUM'].max())
split_week    = int(total_weeks * 0.80)
train_n       = (master['WEEK_NUM'] <= split_week).sum()
val_n         = (master['WEEK_NUM'] >  split_week).sum()
spw           = int((target == 0).sum() / (target == 1).sum())

print(f'\n  Final master shape      : {master.shape}')
print(f'  Total features          : {len(all_feat_cols)}')
print(f'  Total cases             : {len(master):,}')
print(f'  Default rate            : {target.mean()*100:.2f}%')
print(f'  Memory usage            : {mem_mb(master):.0f} MB')
print(f'  Remaining nulls         : {master[all_feat_cols].isnull().sum().sum()}')
print(f'  Train size (week<={split_week}) : {train_n:,}')
print(f'  Val   size (week>{split_week})  : {val_n:,}')
print(f'  scale_pos_weight        : {spw}')

# Save master parquet to D: drive
save_path = OUTPUT_DIR / 'master_train.parquet'
print(f'\n  Saving to: {save_path}')
master.to_parquet(save_path, index=False)
fsize = save_path.stat().st_size / 1024**2
print(f'  Saved! File size: {fsize:.0f} MB')

# Save feature list
feat_path = OUTPUT_DIR / 'feature_list.txt'
with open(feat_path, 'w') as f:
    f.write(f'Total features: {len(all_feat_cols)}\n\n')
    for feat in sorted(all_feat_cols):
        f.write(feat + '\n')
print(f'  Feature list saved: {feat_path}')

# Save split info JSON for Phase 04
split_info = {
    'total_weeks':      total_weeks,
    'split_week':       split_week,
    'train_size':       int(train_n),
    'val_size':         int(val_n),
    'n_features':       len(all_feat_cols),
    'default_rate':     float(round(target.mean(), 4)),
    'scale_pos_weight': spw
}
json_path = OUTPUT_DIR / 'split_info.json'
with open(json_path, 'w') as f:
    json.dump(split_info, f, indent=2)
print(f'  Split info saved : {json_path}')
print(f'\n  split_info.json contents:')
print(json.dumps(split_info, indent=4))

print('\n' + '=' * 65)
print('  PHASE 03 COMPLETE')
print('=' * 65)
print(f'  master_train.parquet -> {save_path}')
print(f'  feature_list.txt     -> {feat_path}')
print(f'  split_info.json      -> {json_path}')
print()
print('  Next -> Phase 04: Modelling & Evaluation')
print('=' * 65)