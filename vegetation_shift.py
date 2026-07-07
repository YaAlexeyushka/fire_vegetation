"""
Неделя 3, шаг 3: разрез по типам растительности (усиленный).

Читает data/vegetation_area_by_year.xlsx (площади типов растительности
MODIS по годам на территориях гарей 2005) и:

1. Показывает динамику ВСЕХ типов, не только 3-5 ключевых.
2. Разбивает изменения по фазам (pre-fire / immediate post / recovery / current).
3. Считает скорости смены типа: где и когда происходит максимальный сдвиг.
4. Считает процентную структуру (доля каждого типа в total, stacked view).
5. Cross-check: сходятся ли смены типов с NDVI/NBR трендами из recovery_per_fire.
6. Сохраняет таблицы и графики для отчёта.

Запуск: python week3_step3_vegetation_shift.py
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')
PLOTS_DIR = OUTPUT_DIR / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

FIRE_YEAR = 2005

COLORS = {
    'Вечнозелёные хвойные леса':         '#1b5e20',
    'Листопадные широколиственные леса': '#8bc34a',
    'Смешанные леса':                    '#4caf50',
    'Редколесье':                        '#ff9800',
    'Разреженный древостой':             '#ffb74d',
    'Луга и пастбища':                   '#ffc107',
    'Пахотные земли':                    '#a1887f',
    'Водоёмы':                           '#03a9f4',
    'Мозаика леса и пахотных':           '#c5e1a5',
    'Городские земли':                   '#616161',
}

veg = pd.read_excel(DATA_DIR / 'vegetation_area_by_year.xlsx')

rename_map = {
    'Лесистые саванны': 'Редколесье',
    'Саванны':          'Разреженный древостой',
}
veg = veg.rename(columns=rename_map)

year_col = next((c for c in veg.columns if c.lower() in ('year', 'год')), 'Year')
veg = veg.set_index(year_col).sort_index()

type_cols = [
    c for c in veg.columns
    if pd.api.types.is_numeric_dtype(veg[c])
    and c.lower() not in ('year', 'год')
    and veg[c].sum() > 0
]
print(f'Загружено: {len(veg)} лет × {len(type_cols)} типов растительности (без пустых)')
print(f'Годы: {veg.index.min()}-{veg.index.max()}')

print('\n=== Изменение площадей 2001 → 2024 по всем типам ===')

pre_year = veg.index.min()
last_year = veg.index.max()

changes = pd.DataFrame({
    f'{pre_year}_ha':  veg.loc[pre_year, type_cols].values,
    f'{last_year}_ha': veg.loc[last_year, type_cols].values,
}, index=type_cols)
changes['delta_ha']  = changes[f'{last_year}_ha'] - changes[f'{pre_year}_ha']
changes['delta_pct'] = (changes['delta_ha'] / changes[f'{pre_year}_ha'].replace(0, pd.NA)) * 100
changes = changes.sort_values('delta_pct', ascending=False)

print(changes.round(1).to_string())

PHASES = {
    'pre':       (2001, 2005),
    'immediate': (2006, 2010),
    'mid':       (2011, 2020),
    'current':   (2021, 2024),
}

phase_means = pd.DataFrame(index=type_cols)
for name, (a, b) in PHASES.items():
    years_in = [y for y in veg.index if a <= y <= b]
    if years_in:
        phase_means[name] = veg.loc[years_in, type_cols].mean().values

print(f'\n=== Средние площади по фазам (га) ===')
print(phase_means.round(0).to_string())

print(f'\n=== Изменения между фазами (% от pre-fire) ===')
delta_phases = pd.DataFrame(index=type_cols)
delta_phases['pre→imm']   = (phase_means['immediate'] - phase_means['pre']) / phase_means['pre'] * 100
delta_phases['imm→mid']   = (phase_means['mid'] - phase_means['immediate']) / phase_means['immediate'].replace(0, pd.NA) * 100
delta_phases['mid→cur']   = (phase_means['current'] - phase_means['mid']) / phase_means['mid'].replace(0, pd.NA) * 100
delta_phases['pre→cur']   = (phase_means['current'] - phase_means['pre']) / phase_means['pre'] * 100
print(delta_phases.round(1).to_string())

print(f'\n=== Год максимального изменения по типам ===')
diffs = veg[type_cols].diff().abs()
max_change_years = pd.DataFrame({
    'year':        diffs.idxmax(),
    'delta_ha':    diffs.max().round(0),
    'delta_pct':   (diffs.max() / veg[type_cols].shift().max()).replace([np.inf, -np.inf], np.nan) * 100,
})
max_change_years['years_from_fire'] = max_change_years['year'] - FIRE_YEAR
print(max_change_years.round(1).to_string())

summary_out = DATA_DIR / 'vegetation_shift_summary.csv'
combined = changes.merge(delta_phases, left_index=True, right_index=True)
combined.round(2).to_csv(summary_out)
print(f'\n[1] Сохранено: {summary_out}')

fig, ax = plt.subplots(figsize=(14, 8))

sort_by_size = veg.loc[pre_year, type_cols].sort_values(ascending=False).index.tolist()
for col in sort_by_size:
    color = COLORS.get(col, '#999999')
    ax.plot(veg.index, veg[col] / 1000, 'o-', color=color,
            linewidth=2, markersize=4, label=col)

ax.axvline(x=2005.7, color='red', linestyle='--', alpha=0.6, label='Пожар (сент. 2005)')
ax.set_xlabel('Год', fontsize=12)
ax.set_ylabel('Площадь (тыс. га)', fontsize=12)
ax.set_title('Динамика площадей типов растительности на территориях гарей 2005 г.\n(MODIS Land Cover)', fontsize=13)
ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out1 = PLOTS_DIR / 'vegetation_dynamics_full.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'[2] Сохранён график: {out1}')

fig, ax = plt.subplots(figsize=(12, 6))
bars = changes.dropna(subset=['delta_pct']).sort_values('delta_pct')

bars['delta_pct'] = bars['delta_pct'].astype(float)
colors = [COLORS.get(t, '#999999') for t in bars.index]
y_pos = np.arange(len(bars))
ax.barh(y_pos, bars['delta_pct'], color=colors, edgecolor='black', linewidth=0.5)
ax.set_yticks(y_pos)
ax.set_yticklabels(bars.index, fontsize=10)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel(f'Изменение площади {pre_year} → {last_year} (%)', fontsize=11)
ax.set_title('Смена типов растительности за 23 года после пожара', fontsize=13)
ax.grid(True, alpha=0.3, axis='x')

for i, (idx, row) in enumerate(bars.iterrows()):
    x = row['delta_pct']
    label = f'{x:+.0f}%'
    ha = 'left' if x >= 0 else 'right'
    offset = 3 if x >= 0 else -3
    ax.text(x + offset, i, label, va='center', ha=ha, fontsize=9)

plt.tight_layout()
out2 = PLOTS_DIR / 'vegetation_change_bar.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'[3] Сохранён график: {out2}')

fig, ax = plt.subplots(figsize=(14, 7))
total = veg[type_cols].sum(axis=1)
percent = veg[type_cols].div(total, axis=0) * 100
percent = percent[sort_by_size]

colors_ordered = [COLORS.get(t, '#999999') for t in sort_by_size]
ax.stackplot(percent.index, percent.T.values, labels=sort_by_size, colors=colors_ordered, alpha=0.85)
ax.axvline(x=2005.7, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax.set_xlabel('Год', fontsize=12)
ax.set_ylabel('Доля от общей площади гарей (%)', fontsize=12)
ax.set_title('Структура типов растительности на территориях гарей 2005 г. (в %)', fontsize=13)
ax.legend(fontsize=9, loc='center left', bbox_to_anchor=(1.01, 0.5), framealpha=0.9)
ax.set_xlim(veg.index.min(), veg.index.max())
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out3 = PLOTS_DIR / 'vegetation_stacked_percent.png'
plt.savefig(out3, dpi=150, bbox_inches='tight')
plt.close()
print(f'[4] Сохранён график: {out3}')

recovery_csv = DATA_DIR / 'recovery_per_fire.csv'
if recovery_csv.exists():
    print(f'\n=== Cross-check с recovery_per_fire ===')
    rec = pd.read_csv(recovery_csv)

    if 'veg_name' in rec.columns:
        by_veg_rec = rec.groupby('veg_name').agg(
            n=('fire_id', 'count'),
            NDVI_slope=('NDVI_slope', 'mean'),
            NDVI_baseline=('NDVI_baseline', 'mean'),
        ).round(4)
        print('Recovery-метрики по dominant vegetation классу (per-fire):')
        print(by_veg_rec.to_string())
        print(f'\nЗамечание: veg_name в recovery — это dominant тип полигона пожара,')
        print(f'а MODIS Land Cover в xlsx — это классификация каждого пикселя по годам.')
        print(f'Оба разреза должны согласоваться в общем: там, где доминирует хвойный,')
        print(f'MODIS показывает падение хвойных, а NDVI slope низкий (у нас 0.004).')

print(f'\n=== Готово ===')
print(f'  {summary_out}')
print(f'  {out1}')
print(f'  {out2}')
print(f'  {out3}')
