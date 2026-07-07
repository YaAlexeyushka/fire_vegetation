"""
Неделя 2, шаг 1 (исправленная версия): Layer 4 + Hansen mask + forest stats.

Изменения по сравнению с первой версией:
1. Обновлён ID Hansen: v1.12 → v1.13 (покрытие 2000–2025). Логика lossyear не меняется.
2. Исправлен знаменатель: теперь fire_area_polygon_ha (геометрическая площадь
   полигона), а не Area из MODIS (фактически выгоревшие гектары).
   Колонка Area сохранена отдельно как fire_area_modis_ha.
3. Добавлены площади зон (repeat_area, clean_area) для расчёта долей внутри зон.
4. Расширенный набор долей: по полигону, по MODIS, внутри каждой зоны отдельно.

Запуск: python week2_step1_hansen.py
Требования: ee.Authenticate() уже выполнен, output/fires_*.geojson от Недели 1 есть.
"""
import ee
import geemap
import geopandas as gpd
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('output')
DATA_DIR = Path('data')

METRIC_CRS = 'EPSG:32648'
WGS84 = 'EPSG:4326'

HANSEN_ID = 'UMD/hansen/global_forest_change_2025_v1_13'
FOREST_THRESHOLD = 30
STUDY_YEAR = 2005
AREA_THRESHOLD_HA = 500

GEE_PROJECT = 'bubbly-operator-479705-v3'

ee.Initialize(project=GEE_PROJECT)
print(f'GEE инициализирован: {GEE_PROJECT}')

print('\n=== ШАГ 1: Layer 4 — чистые зоны 2005 ===')

fires_by_year = gpd.read_file(OUTPUT_DIR / 'fires_by_year.geojson')
non_2005 = gpd.read_file(OUTPUT_DIR / 'fires_non2005_dissolved.geojson')

fires_2005 = fires_by_year[fires_by_year['year'] == STUDY_YEAR].copy()
if 'Area' in fires_2005.columns:
    fires_2005 = fires_2005[fires_2005['Area'] >= AREA_THRESHOLD_HA].copy()
print(f'  Пожаров 2005 с Area >= {AREA_THRESHOLD_HA} га: {len(fires_2005)}')

fires_2005_m = fires_2005.to_crs(METRIC_CRS).copy()
non_2005_m = non_2005.to_crs(METRIC_CRS).copy()
fires_2005_m['geometry'] = fires_2005_m.geometry.buffer(0)
non_2005_m['geometry'] = non_2005_m.geometry.buffer(0)

fires_2005_m['fire_area_polygon_ha'] = fires_2005_m.geometry.area / 10_000

fires_2005_clean = gpd.overlay(fires_2005_m, non_2005_m, how='difference')
fires_2005_clean['clean_area_ha'] = fires_2005_clean.geometry.area / 10_000
fires_2005_clean = fires_2005_clean[fires_2005_clean['clean_area_ha'] > 0].copy()

out_layer4 = OUTPUT_DIR / 'fires_2005_clean.geojson'
fires_2005_clean.to_crs(WGS84).to_file(out_layer4, driver='GeoJSON')
print(f'  Сохранено: {out_layer4}')
print(f'  Полигонов: {len(fires_2005_clean)}')
print(f'  Общая площадь чистых зон: {fires_2005_clean.clean_area_ha.sum():,.0f} га')

total_polygon = fires_2005_m['fire_area_polygon_ha'].sum()
total_modis = fires_2005['Area'].sum() if 'Area' in fires_2005.columns else None
print(f'\n  --- Площади для контроля ---')
print(f'  Полигон 2005 (геометрическая):  {total_polygon:>10,.0f} га')
if total_modis is not None:
    print(f'  MODIS «выгорело» (Area):        {total_modis:>10,.0f} га '
          f'({total_modis / total_polygon * 100:.1f}% полигона)')

print('\n=== ШАГ 2: Hansen healthy forest mask ===')
print(f'  Asset: {HANSEN_ID}')

hansen = ee.Image(HANSEN_ID)
treecover = hansen.select('treecover2000')
lossyear = hansen.select('lossyear')
datamask = hansen.select('datamask')

healthy_forest_pre2005 = (
    treecover.gte(FOREST_THRESHOLD)
    .And(lossyear.eq(0).Or(lossyear.gte(5)))
    .And(datamask.eq(1))
    .rename('healthy')
)

forest_m2 = healthy_forest_pre2005.multiply(ee.Image.pixelArea()).rename('m2_pre')

forest_lost_2005_m2 = (
    treecover.gte(FOREST_THRESHOLD)
    .And(lossyear.eq(5))
    .And(datamask.eq(1))
    .multiply(ee.Image.pixelArea())
    .rename('m2_lost')
)

stack = forest_m2.addBands(forest_lost_2005_m2)
print(f'  Маска: treecover>={FOREST_THRESHOLD}%, lossyear in {{0, >=5}}, datamask=1')

print('\n=== ШАГ 3: forest_area по зонам ===')

def compute_forest_area(gdf: gpd.GeoDataFrame, zone_name: str, suffix: str) -> pd.DataFrame:
    """reduceRegions по полигонам gdf, возврат DataFrame с колонками fire_id + площади."""
    print(f'  Зона «{zone_name}»: {len(gdf)} полигонов...')

    gdf_wgs = gdf.to_crs(WGS84).copy()
    keep = [c for c in ['fire_id', 'geometry'] if c in gdf_wgs.columns]
    gdf_wgs = gdf_wgs[keep]
    gdf_wgs = gdf_wgs[~gdf_wgs.geometry.is_empty & gdf_wgs.geometry.is_valid]

    fc = geemap.geopandas_to_ee(gdf_wgs)

    reduced = stack.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.sum(),
        scale=30,
        crs=WGS84,
    )

    features = reduced.getInfo()['features']
    rows = []
    for f in features:
        p = f['properties']
        rows.append({
            'fire_id': p.get('fire_id'),
            f'forest_pre_{suffix}_ha': (p.get('m2_pre') or 0) / 10_000,
            f'forest_lost_{suffix}_ha': (p.get('m2_lost') or 0) / 10_000,
        })
    df = pd.DataFrame(rows)

    df = df.groupby('fire_id', as_index=False).sum()
    return df

df_total = compute_forest_area(fires_2005, 'вся гарь 2005', 'total')

repeats = gpd.read_file(OUTPUT_DIR / 'fires_2005_repeats.geojson')
df_repeats = compute_forest_area(repeats, 'повторные', 'repeats')

df_clean = compute_forest_area(fires_2005_clean, 'чистые', 'clean')

print('\n=== ШАГ 4: Сводная таблица ===')

fires_summary = fires_2005_m[['fire_id', 'fire_area_polygon_ha']].copy()
if 'Area' in fires_2005.columns:
    fires_summary = fires_summary.merge(
        fires_2005[['fire_id', 'Area']].rename(columns={'Area': 'fire_area_modis_ha'}),
        on='fire_id', how='left',
    )

repeat_areas = (
    repeats.groupby('fire_id', as_index=False)['repeat_area_ha']
    .sum()
)

clean_areas = (
    fires_2005_clean.groupby('fire_id', as_index=False)['clean_area_ha']
    .sum()
)

df = (
    fires_summary
    .merge(repeat_areas, on='fire_id', how='left')
    .merge(clean_areas, on='fire_id', how='left')
    .merge(df_total, on='fire_id', how='left')
    .merge(df_repeats, on='fire_id', how='left')
    .merge(df_clean, on='fire_id', how='left')
    .fillna(0)
)

df['forest_share_polygon'] = df['forest_pre_total_ha'] / df['fire_area_polygon_ha']

if 'fire_area_modis_ha' in df.columns:
    df['forest_share_modis'] = (
        df['forest_pre_total_ha'] / df['fire_area_modis_ha'].replace(0, pd.NA)
    )

df['hansen_loss_share_total'] = (
    df['forest_lost_total_ha'] / df['forest_pre_total_ha'].replace(0, pd.NA)
)

df['hansen_loss_share_repeats'] = (
    df['forest_lost_repeats_ha'] / df['forest_pre_repeats_ha'].replace(0, pd.NA)
)
df['hansen_loss_share_clean'] = (
    df['forest_lost_clean_ha'] / df['forest_pre_clean_ha'].replace(0, pd.NA)
)

df['forest_share_in_repeats'] = (
    df['forest_pre_repeats_ha'] / df['repeat_area_ha'].replace(0, pd.NA)
)
df['forest_share_in_clean'] = (
    df['forest_pre_clean_ha'] / df['clean_area_ha'].replace(0, pd.NA)
)

DATA_DIR.mkdir(exist_ok=True)
out_csv = DATA_DIR / 'forest_stats_per_fire.csv'
df.round(3).to_csv(out_csv, index=False)

def pct(num, den):
    if not den or pd.isna(den):
        return 'N/A'
    return f'{num / den * 100:.1f}%'

print(f'\n  Сохранено: {out_csv}')
print(f'  Колонок: {len(df.columns)}')

print(f'\n  --- Площади ---')
print(f'  Полигоны 2005 (geom):        {df.fire_area_polygon_ha.sum():>10,.0f} га')
if 'fire_area_modis_ha' in df.columns:
    print(f'  MODIS «выгорело»:            {df.fire_area_modis_ha.sum():>10,.0f} га')
print(f'  Повторные зоны:              {df.repeat_area_ha.sum():>10,.0f} га')
print(f'  Чистые зоны:                 {df.clean_area_ha.sum():>10,.0f} га')

print(f'\n  --- Лес до 2005 (Hansen, treecover>=30%) ---')
total_pre = df.forest_pre_total_ha.sum()
print(f'  В полигонах 2005:            {total_pre:>10,.0f} га '
      f'({pct(total_pre, df.fire_area_polygon_ha.sum())} от полигона)')
print(f'    в повторных зонах:         {df.forest_pre_repeats_ha.sum():>10,.0f} га '
      f'({pct(df.forest_pre_repeats_ha.sum(), df.repeat_area_ha.sum())} от площади повторных)')
print(f'    в чистых зонах:            {df.forest_pre_clean_ha.sum():>10,.0f} га '
      f'({pct(df.forest_pre_clean_ha.sum(), df.clean_area_ha.sum())} от площади чистых)')

print(f'\n  --- Лес, потерянный в 2005 (Hansen, lossyear=5) ---')
total_lost = df.forest_lost_total_ha.sum()
print(f'  Всего:                       {total_lost:>10,.0f} га '
      f'({pct(total_lost, total_pre)} от исходного леса)')
lost_rep = df.forest_lost_repeats_ha.sum()
pre_rep = df.forest_pre_repeats_ha.sum()
print(f'    в повторных:               {lost_rep:>10,.0f} га '
      f'({pct(lost_rep, pre_rep)} от леса в повторных)')
lost_clean = df.forest_lost_clean_ha.sum()
pre_clean = df.forest_pre_clean_ha.sum()
print(f'    в чистых:                  {lost_clean:>10,.0f} га '
      f'({pct(lost_clean, pre_clean)} от леса в чистых)')

print(f'\n  ⚠ Hansen lossyear фиксирует ПЕРВОЕ событие. В повторных зонах часть')
print(f'    потерь приписана прежним годам (2001–2004), поэтому здесь Hansen')
print(f'    систематически недооценивает ущерб 2005. На следующем шаге dNBR')
print(f'    даст несмещённую оценку — ждём, что severity в repeats ≥ clean.')

print('\n=== Готово ===')
print(f'  1. {out_layer4}')
print(f'  2. {out_csv}')
