"""
Неделя 3, шаг 1: метрики восстановления по каждому пожару.

Что делает:
Читает vi_wide_format.csv (готовые VI-агрегации по fire_id × year) и
damage_per_fire.csv (сценарии из недели 2). Для каждого пожара считает:

    - baseline:         медиана VI за pre-fire годы (2002, 2004, 2005)
    - post_min:         минимум VI в 2006–2024 (обычно 2006)
    - min_year:         в каком году был минимум
    - current:          последнее доступное значение (обычно 2024)
    - drop:             baseline − post_min
    - recovery_share:   (current − post_min) / drop (0 = ноль, 1 = полностью восстановлен)
    - recovery_slope:   наклон линейной регрессии VI(year) с min_year по 2024
    - r_squared:        R² регрессии (качество тренда)
    - time_to_90pct:    сколько лет с 2005 до первого года >= 90% baseline (NaN если не достиг)

Считаются для NDVI и NBR — как основных индикаторов вегетации.
Мерж с damage/veg_name даёт готовую таблицу для сценариев и разрезов.

Запуск: python week3_step1_recovery.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA_DIR = Path('data')

STUDY_YEAR = 2005

BASELINE_YEARS = [2002, 2004, 2005]

POST_START = 2006
POST_END = 2024
RECOVERY_THRESHOLD = 0.90

INDICES = ['NDVI', 'NBR']

vi = pd.read_csv(DATA_DIR / 'vi_wide_format.csv')
damage = pd.read_csv(DATA_DIR / 'damage_per_fire.csv')
print(f'Загружено:')
print(f'  vi_wide_format:   {len(vi):,} строк, {vi.fire_id.nunique()} пожаров')
print(f'  damage_per_fire:  {len(damage):,} строк')

def compute_recovery(fire_data: pd.DataFrame, index: str) -> dict:
    col = f'{index}_median'
    if col not in fire_data.columns:
        return {}

    yearly = fire_data.set_index('year')[col].dropna()
    if yearly.empty:
        return {}

    pre = yearly.reindex(BASELINE_YEARS).dropna()
    baseline = pre.mean() if len(pre) > 0 else np.nan

    post = yearly[(yearly.index >= POST_START) & (yearly.index <= POST_END)]
    if post.empty:
        return {f'{index}_baseline': baseline}

    min_val = post.min()
    min_year = int(post.idxmin())
    current = post.iloc[-1]

    drop = baseline - min_val if not np.isnan(baseline) else np.nan

    if drop and drop > 0:
        recovery_share = (current - min_val) / drop
    else:
        recovery_share = np.nan

    recovery_period = post[post.index >= min_year]
    if len(recovery_period) >= 3:
        slope, intercept, r, p, se = stats.linregress(
            recovery_period.index.values.astype(float),
            recovery_period.values,
        )
        r_squared = r ** 2
    else:
        slope, r_squared = np.nan, np.nan

    if not np.isnan(baseline):
        target = baseline * RECOVERY_THRESHOLD
        reached = post[(post.index >= min_year) & (post >= target)]
        time_to_90 = (int(reached.index[0]) - STUDY_YEAR) if not reached.empty else np.nan
    else:
        time_to_90 = np.nan

    return {
        f'{index}_baseline':       baseline,
        f'{index}_post_min':       min_val,
        f'{index}_min_year':       min_year,
        f'{index}_current':        current,
        f'{index}_drop':           drop,
        f'{index}_recovery_share': recovery_share,
        f'{index}_slope':          slope,
        f'{index}_r_squared':      r_squared,
        f'{index}_time_to_90pct':  time_to_90,
    }

print(f'\nРасчёт метрик для {vi.fire_id.nunique()} пожаров...')
records = []
for fire_id, group in vi.groupby('fire_id'):
    row = {'fire_id': fire_id}
    for idx in INDICES:
        row.update(compute_recovery(group, idx))

    if 'veg_name' in group.columns:
        row['veg_name'] = group['veg_name'].iloc[0]
    if 'is_clean' in group.columns:
        row['is_clean'] = bool(group['is_clean'].iloc[0])
    records.append(row)

recovery = pd.DataFrame(records)

merge_cols = [
    'fire_id',
    'fire_area_polygon_ha',
    'clean_area_ha', 'repeat_area_ha',
    'forest_pre_total_ha',
    'burn_rbr_forest_total_ha',
    'damage_share_forest_with_repeats',
    'damage_share_forest_without_repeats',
    'damage_share_forest_only_repeats',
]
merge_cols = [c for c in merge_cols if c in damage.columns]
recovery = recovery.merge(damage[merge_cols], on='fire_id', how='left')

if 'is_clean' not in recovery.columns:
    recovery['is_clean'] = recovery['repeat_area_ha'].fillna(0) == 0

out = DATA_DIR / 'recovery_per_fire.csv'
recovery.round(4).to_csv(out, index=False)
print(f'\n[1] Сохранено: {out}  ({len(recovery)} строк, {len(recovery.columns)} колонок)')

def compare(clean, dirty, col, label, fmt='{:.4f}'):
    c_m = clean[col].mean() if col in clean.columns else float('nan')
    d_m = dirty[col].mean() if col in dirty.columns else float('nan')
    c_s = fmt.format(c_m)
    d_s = fmt.format(d_m)
    diff = ((d_m - c_m) / abs(c_m) * 100) if c_m and not np.isnan(c_m) else float('nan')
    diff_s = f'{diff:+.1f}%' if not np.isnan(diff) else 'N/A'
    print(f'  {label:<22} {c_s:>10}  {d_s:>10}  diff: {diff_s:>8}')

clean = recovery[recovery.is_clean == True]
dirty = recovery[recovery.is_clean == False]
print(f'\n=== Восстановление: чистые (n={len(clean)}) vs повторные (n={len(dirty)}) ===')

for idx in INDICES:
    print(f'\n{idx}:')
    print(f'{"":>24} {"Clean":>10}  {"Dirty":>10}  {"Diff":>15}')
    compare(clean, dirty, f'{idx}_baseline',       'baseline')
    compare(clean, dirty, f'{idx}_post_min',       'post-fire min')
    compare(clean, dirty, f'{idx}_drop',           'drop magnitude')
    compare(clean, dirty, f'{idx}_recovery_share', 'recovery share')
    compare(clean, dirty, f'{idx}_slope',          'recovery slope')
    compare(clean, dirty, f'{idx}_time_to_90pct',  'years to 90%', '{:.1f}')

if 'veg_name' in recovery.columns:
    print('\n=== Восстановление по типам растительности ===')
    for idx in INDICES:
        slope_col = f'{idx}_slope'
        share_col = f'{idx}_recovery_share'
        t90_col = f'{idx}_time_to_90pct'
        if slope_col not in recovery.columns:
            continue
        by_veg = recovery.groupby('veg_name').agg(
            n=('fire_id', 'count'),
            baseline=(f'{idx}_baseline', 'mean'),
            drop=(f'{idx}_drop', 'mean'),
            slope=(slope_col, 'mean'),
            recovery=(share_col, 'mean'),
            t90=(t90_col, 'mean'),
        ).round(4)
        print(f'\n{idx}:')
        print(by_veg.to_string())

print('\n=== Корреляции recovery_slope vs атрибуты ===')
for idx in INDICES:
    slope_col = f'{idx}_slope'
    if slope_col not in recovery.columns:
        continue
    print(f'\n{idx}_slope vs:')
    for attr in [
        'fire_area_polygon_ha',
        'forest_pre_total_ha',
        'burn_rbr_forest_total_ha',
        'damage_share_forest_with_repeats',
    ]:
        if attr not in recovery.columns:
            continue
        pair = recovery[[slope_col, attr]].dropna()
        if len(pair) >= 3:
            r_pearson = pair.corr('pearson').iloc[0, 1]
            r_spearman = pair.corr('spearman').iloc[0, 1]
            print(f'  {attr:<40} Pearson: {r_pearson:+.3f}  Spearman: {r_spearman:+.3f}  n={len(pair)}')

print(f'\n=== Готово ===')
print(f'  {out}')
