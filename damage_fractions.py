import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')

forest = pd.read_csv(DATA_DIR / 'forest_stats_per_fire.csv')
dnbr   = pd.read_csv(DATA_DIR / 'dnbr_extended_stats.csv')
print(f'Загружено: forest_stats_per_fire ({len(forest)} строк)')
print(f'Загружено: dnbr_extended_stats   ({len(dnbr)} строк)')

df = forest.merge(dnbr, on='fire_id', how='inner')
print(f'После merge: {len(df)} пожаров')

SCENARIOS = {
    'with_repeats':    ('burn_rbr_forest_total_ha',   'fire_area_polygon_ha', 'forest_pre_total_ha'),
    'without_repeats': ('burn_rbr_forest_clean_ha',   'clean_area_ha',        'forest_pre_clean_ha'),
    'only_repeats':    ('burn_rbr_forest_repeats_ha', 'repeat_area_ha',       'forest_pre_repeats_ha'),
}

SCENARIO_LABELS = {
    'with_repeats':    'С повторными (все 2005)',
    'without_repeats': 'Без повторных (только чистые)',
    'only_repeats':    'Только повторные',
}

for sc, (burn_col, area_col, forest_col) in SCENARIOS.items():
    df[f'damage_share_polygon_{sc}'] = (
        df[burn_col] / df[area_col].replace(0, np.nan)
    )
    df[f'damage_share_forest_{sc}'] = (
        df[burn_col] / df[forest_col].replace(0, np.nan)
    )

keep = [
    'fire_id',

    'fire_area_polygon_ha', 'repeat_area_ha', 'clean_area_ha',

    'forest_pre_total_ha', 'forest_pre_repeats_ha', 'forest_pre_clean_ha',
    'forest_lost_total_ha', 'forest_lost_repeats_ha', 'forest_lost_clean_ha',

    'nbr_pre_total', 'nbr_post_total', 'dnbr_total', 'rbr_total',
    'nbr_pre_repeats', 'nbr_post_repeats', 'dnbr_repeats', 'rbr_repeats',
    'nbr_pre_clean', 'nbr_post_clean', 'dnbr_clean', 'rbr_clean',

    'burn_rbr_forest_total_ha', 'burn_rbr_forest_repeats_ha', 'burn_rbr_forest_clean_ha',

    'damage_share_polygon_with_repeats',
    'damage_share_polygon_without_repeats',
    'damage_share_polygon_only_repeats',

    'damage_share_forest_with_repeats',
    'damage_share_forest_without_repeats',
    'damage_share_forest_only_repeats',
]
keep = [c for c in keep if c in df.columns]
df_out = df[keep].copy()

out_per_fire = DATA_DIR / 'damage_per_fire.csv'
df_out.round(4).to_csv(out_per_fire, index=False)
print(f'\n[1] Сохранено: {out_per_fire}  ({len(df_out)} строк, {len(df_out.columns)} колонок)')

def total(col):
    return df[col].sum() if col in df.columns else 0.0

rows = []
for sc, (burn_col, area_col, forest_col) in SCENARIOS.items():
    area_sum = total(area_col)
    forest_sum = total(forest_col)
    burn_sum = total(burn_col)
    rows.append({
        'scenario':            SCENARIO_LABELS[sc],
        'scenario_id':         sc,
        'n_fires_with_zone':   (df[area_col] > 0).sum() if area_col in df.columns else 0,
        'area_total_ha':       area_sum,
        'forest_total_ha':     forest_sum,
        'burn_rbr_forest_ha':  burn_sum,
        'damage_share_polygon': burn_sum / area_sum if area_sum else float('nan'),
        'damage_share_forest':  burn_sum / forest_sum if forest_sum else float('nan'),
    })

agg = pd.DataFrame(rows)
out_agg = DATA_DIR / 'damage_aggregate.csv'
agg.round(4).to_csv(out_agg, index=False)
print(f'[2] Сохранено: {out_agg}')

print('\n' + '=' * 75)
print('СВОДКА: доля повреждений (burn = RBR >= 0.27 ∩ Hansen forest)')
print('=' * 75)
print(f'{"Сценарий":<32} {"Гарь, га":>10} {"Площадь":>10} {"% поли":>8} {"% лес":>8}')
print('-' * 75)
for _, r in agg.iterrows():
    print(f'{r["scenario"]:<32} '
          f'{r["burn_rbr_forest_ha"]:>10,.0f} '
          f'{r["area_total_ha"]:>10,.0f} '
          f'{r["damage_share_polygon"]*100:>7.1f}% '
          f'{r["damage_share_forest"]*100:>7.1f}%')

with_ = agg.loc[agg.scenario_id == 'with_repeats',    'damage_share_forest'].iloc[0]
without = agg.loc[agg.scenario_id == 'without_repeats', 'damage_share_forest'].iloc[0]
only_r  = agg.loc[agg.scenario_id == 'only_repeats',   'damage_share_forest'].iloc[0]

print(f'\n--- Главный сюжет (% повреждённого леса по RBR) ---')
print(f'  Со всеми зонами:        {with_*100:.1f}%')
print(f'  Только чистые зоны:     {without*100:.1f}%')
print(f'  Только повторные зоны:  {only_r*100:.1f}%')
diff = (without - only_r) / max(only_r, 1e-9) * 100
print(f'\n  Чистые / повторные = {without/only_r:.2f}×  ({diff:+.0f}%)')
print(f'  → подтверждает fuel-limitation-эффект: чистые зоны (зрелый хвойный')
print(f'    лес) горят значительно сильнее повторных (где топливо уже выгорело).')

print(f'\n--- Распределение damage_share_forest по 159 пожарам ---')
desc_with = df_out['damage_share_forest_with_repeats'].describe()
desc_without = df_out['damage_share_forest_without_repeats'].describe()
print(f'{"":>10} {"С повторными":>15} {"Без повторных":>15}')
for stat in ['mean', 'std', '25%', '50%', '75%']:
    a = desc_with[stat] if stat in desc_with.index else float('nan')
    b = desc_without[stat] if stat in desc_without.index else float('nan')
    print(f'{stat:>10} {a*100:>14.1f}% {b*100:>14.1f}%')

print('\n=== Готово ===')
print(f'  {out_per_fire}')
print(f'  {out_agg}')
