"""
Неделя 3, шаг 4: расширенные корреляции recovery с severity и типом растительности.

Что делает:
- Считает Spearman-корреляции между recovery-метриками (NDVI_slope, NBR_slope,
  recovery_share, time_to_90pct) и атрибутами пожаров: area, forest, burn area,
  damage share, severity (RBR, dNBR), pre-fire baseline, drop magnitude.
- Для каждой пары выдаёт r, p-value, N, маркеры значимости (* p<0.05, ** p<0.01).
- Разделяет анализ:
  * всех пожаров вместе
  * по типам растительности (там, где n >= 10)
- Сохраняет полную матрицу в data/correlation_matrix.csv.

Запуск: python week3_step4_correlations.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

DATA = Path('data')

rec = pd.read_csv(DATA / 'recovery_per_fire.csv')
dnbr = pd.read_csv(DATA / 'dnbr_extended_stats.csv')
sev_cols = ['fire_id', 'nbr_pre_total', 'nbr_post_total', 'dnbr_total', 'rbr_total']
rec = rec.merge(dnbr[sev_cols], on='fire_id', how='left')
print(f'Загружено: {len(rec)} пожаров, {len(rec.columns)} колонок после мержа')

METRICS = [
    'NDVI_slope', 'NBR_slope',
    'NDVI_recovery_share', 'NBR_recovery_share',
    'NDVI_time_to_90pct',
]

ATTRIBUTES = [
    'fire_area_polygon_ha',
    'forest_pre_total_ha',
    'burn_rbr_forest_total_ha',
    'damage_share_forest_with_repeats',
    'rbr_total',
    'dnbr_total',
    'NDVI_baseline', 'NDVI_drop',
    'NBR_baseline',  'NBR_drop',
]

def corr(x, y):
    """Возвращает (r, p, n) для Spearman."""
    pair = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(pair) < 3:
        return np.nan, np.nan, len(pair)
    r, p = stats.spearmanr(pair['x'], pair['y'])
    return r, p, len(pair)

def build_corr_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in METRICS:
        if m not in df.columns:
            continue
        for a in ATTRIBUTES:
            if a not in df.columns:
                continue
            r, p, n = corr(df[m], df[a])
            if not np.isnan(r):
                sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
                rows.append({
                    'metric': m, 'attribute': a,
                    'r': round(r, 3), 'p': round(p, 4), 'n': n, 'sig': sig,
                })
    return pd.DataFrame(rows)

all_c = build_corr_long(rec)
sig = all_c[all_c.p < 0.05].sort_values('r', key=lambda x: x.abs(), ascending=False)
print(f'\n=== Spearman, все пожары ===')
print(f'Значимых связей (p<0.05): {len(sig)} из {len(all_c)}')
print(sig.to_string(index=False))

print(f'\n=== По типам растительности (только n>=10) ===')
if 'veg_name' in rec.columns:
    for veg, g in rec.groupby('veg_name'):
        if len(g) < 10:
            continue
        v_c = build_corr_long(g)
        sig_v = v_c[v_c.p < 0.05].sort_values('r', key=lambda x: x.abs(), ascending=False)
        print(f'\n{veg} (n={len(g)}, значимых связей: {len(sig_v)}):')
        if len(sig_v):
            print(sig_v.head(5).to_string(index=False))
        else:
            print('  Значимых связей нет')

out = DATA / 'correlation_matrix.csv'
all_c.sort_values(['metric', 'attribute']).to_csv(out, index=False)
print(f'\n=== Сохранено: {out} ({len(all_c)} пар всего) ===')
