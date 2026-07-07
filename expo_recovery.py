"""
Неделя 3, шаг 2: экспоненциальная модель восстановления.

Заменяет линейную регрессию из шага 1 на нелинейную модель:
    VI(t) = A − delta · exp(−k · t)
где:
    t     = годы с 2006 (post-fire minimum)
    A     = асимптотическое значение восстановления (потолок)
    delta = глубина начального падения (A − VI_min)
    k     = константа скорости восстановления (1/годы)

Что даёт физический смысл каждого параметра:
    - A:     до какого уровня VI восстановится в пределе
    - k:     как быстро, larger k → faster recovery
    - time_to_half = ln(2)/k     — время закрыть половину gap
    - time_to_90%  = ln(10)/k    — время закрыть 90% gap
    - A / baseline:               доля восстановления от pre-fire уровня
                                  (< 1 = permanent damage, ≈ 1 = full recovery)

Запуск: python week3_step2_expo_recovery.py
"""
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

DATA_DIR = Path('data')

STUDY_YEAR = 2005
FIRE_MIN_YEAR = 2006
BASELINE_YEARS = [2002, 2004, 2005]
POST_START = 2006
POST_END = 2024
INDICES = ['NDVI', 'NBR']

MIN_POINTS = 4
K_INIT = 0.15

def recovery_model(t, A, delta, k):
    """VI(t) = A − delta · exp(−k · t)"""
    return A - delta * np.exp(-k * t)

def fit_recovery(fire_data: pd.DataFrame, index: str) -> dict:
    col = f'{index}_median'
    if col not in fire_data.columns:
        return {}

    yearly = fire_data.set_index('year')[col].dropna()
    if yearly.empty:
        return {}

    pre = yearly.reindex(BASELINE_YEARS).dropna()
    baseline = pre.mean() if len(pre) > 0 else np.nan

    post = yearly[(yearly.index >= POST_START) & (yearly.index <= POST_END)]
    if len(post) < MIN_POINTS:
        return {f'{index}_baseline': baseline, f'{index}_fit_status': 'too_few_points'}

    t_data = (post.index - FIRE_MIN_YEAR).values.astype(float)
    y_data = post.values.astype(float)

    A_init = baseline if not np.isnan(baseline) else float(y_data.max())
    delta_init = max(A_init - float(y_data.min()), 0.01)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            popt, _ = curve_fit(
                recovery_model,
                t_data, y_data,
                p0=[A_init, delta_init, K_INIT],
                bounds=(
                    [-0.5, 0.0,  0.001],
                    [ 1.5, 2.0,  5.0],
                ),
                maxfev=5000,
            )
        A_fit, delta_fit, k_fit = popt

        y_pred = recovery_model(t_data, *popt)
        ss_res = ((y_data - y_pred) ** 2).sum()
        ss_tot = ((y_data - y_data.mean()) ** 2).sum()
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        time_to_half = float(np.log(2) / k_fit) if k_fit > 0 else np.nan
        time_to_90 = float(np.log(10) / k_fit) if k_fit > 0 else np.nan
        recovery_ratio = A_fit / baseline if (baseline and not np.isnan(baseline)) else np.nan

        return {
            f'{index}_baseline':         baseline,
            f'{index}_A':                A_fit,
            f'{index}_delta':            delta_fit,
            f'{index}_k':                k_fit,
            f'{index}_time_to_half':     time_to_half,
            f'{index}_time_to_90pct':    time_to_90,
            f'{index}_recovery_ratio':   recovery_ratio,
            f'{index}_r_squared_expo':   r_squared,
            f'{index}_fit_status':       'ok',
        }
    except (RuntimeError, ValueError) as e:
        return {
            f'{index}_baseline': baseline,
            f'{index}_fit_status': f'failed: {type(e).__name__}',
        }

vi = pd.read_csv(DATA_DIR / 'vi_wide_format.csv')
damage = pd.read_csv(DATA_DIR / 'damage_per_fire.csv')
recovery_lin = pd.read_csv(DATA_DIR / 'recovery_per_fire.csv')
print(f'Загружено: {vi.fire_id.nunique()} пожаров')

print(f'\nФит экспоненциальной модели...')
records = []
for fire_id, group in vi.groupby('fire_id'):
    row = {'fire_id': fire_id}
    for idx in INDICES:
        row.update(fit_recovery(group, idx))
    if 'veg_name' in group.columns:
        row['veg_name'] = group['veg_name'].iloc[0]
    if 'is_clean' in group.columns:
        row['is_clean'] = bool(group['is_clean'].iloc[0])
    records.append(row)

expo = pd.DataFrame(records)

merge_cols = [c for c in [
    'fire_id',
    'fire_area_polygon_ha', 'clean_area_ha', 'repeat_area_ha',
    'forest_pre_total_ha', 'burn_rbr_forest_total_ha',
    'damage_share_forest_with_repeats',
] if c in damage.columns]
expo = expo.merge(damage[merge_cols], on='fire_id', how='left')

out = DATA_DIR / 'recovery_expo_per_fire.csv'
expo.round(4).to_csv(out, index=False)
print(f'\n[1] Сохранено: {out}  ({len(expo)} строк, {len(expo.columns)} колонок)')

print('\n=== Статус фитов ===')
for idx in INDICES:
    status_col = f'{idx}_fit_status'
    if status_col in expo.columns:
        print(f'\n{idx}:')
        print(expo[status_col].value_counts().to_string())

print('\n=== R²: линейная vs экспоненциальная ===')
for idx in INDICES:
    lin_col = f'{idx}_r_squared'
    exp_col = f'{idx}_r_squared_expo'

    if exp_col not in expo.columns or lin_col not in recovery_lin.columns:
        continue

    m = recovery_lin[['fire_id', lin_col]].merge(expo[['fire_id', exp_col]], on='fire_id')
    m = m.dropna()

    print(f'\n{idx} (n={len(m)}):')
    print(f'  Линейная:      mean R² = {m[lin_col].mean():.3f}, median = {m[lin_col].median():.3f}')
    print(f'  Экспоненц.:    mean R² = {m[exp_col].mean():.3f}, median = {m[exp_col].median():.3f}')
    improved = (m[exp_col] > m[lin_col]).mean() * 100
    print(f'  Улучшилось в:  {improved:.0f}% пожаров')

print('\n=== Экспо-метрики: clean vs dirty ===')
clean = expo[expo.is_clean == True]
dirty = expo[expo.is_clean == False]

for idx in INDICES:
    print(f'\n{idx}:')
    print(f'  {"":<28} {"Clean":>10}  {"Dirty":>10}')
    for col, label in [
        (f'{idx}_A',              'asymptote A'),
        (f'{idx}_k',              'rate constant k'),
        (f'{idx}_time_to_half',   'years to half'),
        (f'{idx}_time_to_90pct',  'years to 90%'),
        (f'{idx}_recovery_ratio', 'A / baseline'),
        (f'{idx}_r_squared_expo', 'R² expo'),
    ]:
        if col not in expo.columns:
            continue
        c_m = clean[col].mean()
        d_m = dirty[col].mean()
        print(f'  {label:<28} {c_m:>10.4f}  {d_m:>10.4f}')

if 'veg_name' in expo.columns:
    print('\n=== Экспо-метрики по типам растительности ===')
    for idx in INDICES:
        cols_needed = [f'{idx}_A', f'{idx}_k', f'{idx}_time_to_half', f'{idx}_recovery_ratio']
        if not all(c in expo.columns for c in cols_needed):
            continue
        by_veg = expo.groupby('veg_name').agg(
            n=('fire_id', 'count'),
            A=(f'{idx}_A', 'mean'),
            k=(f'{idx}_k', 'mean'),
            t_half=(f'{idx}_time_to_half', 'mean'),
            ratio=(f'{idx}_recovery_ratio', 'mean'),
        ).round(3)
        print(f'\n{idx}:')
        print(by_veg.to_string())

print(f'\n=== Готово ===')
print(f'  {out}')
