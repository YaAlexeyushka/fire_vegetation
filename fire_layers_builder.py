import glob
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)
METRIC_CRS = 'EPSG:32648'
OUTPUT_CRS = 'EPSG:4326'

STUDY_YEAR = 2005

AREA_THRESHOLD_HA = 500

def extract_year(filepath) -> int | None:
    m = re.search(r'(\d{4})\.shp$', str(filepath), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None

irk_obl = gpd.read_file(DATA_DIR / 'irkutsk_region.geojson')
print(f'Иркутская область загружена. CRS: {irk_obl.crs}')

files = sorted(glob.glob(str(DATA_DIR / 'fires_*mc6mc6v3corr_*.shp')))
print(f'Найдено файлов: {len(files)}\n')

per_year = []
for f in files:
    year = extract_year(f)
    if year is None:
        continue

    gdf = gpd.read_file(f)
    if gdf.crs != irk_obl.crs:
        gdf = gdf.to_crs(irk_obl.crs)

    gdf = gpd.sjoin(gdf, irk_obl[['geometry']], predicate='intersects', how='inner')
    gdf = gdf.drop(columns=['index_right'], errors='ignore')
    gdf['year'] = year
    per_year.append(gdf)
    print(f'  {year}: {len(gdf):>6,} пожаров (в Иркутской области)')

fires_all = pd.concat(per_year, ignore_index=True)
fires_all = gpd.GeoDataFrame(fires_all, crs=irk_obl.crs)

for col in fires_all.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
    fires_all[col] = fires_all[col].dt.strftime('%Y-%m-%d')

print(f'\nВсего пожаров по всем годам: {len(fires_all):,}')

out1 = OUTPUT_DIR / 'fires_by_year.geojson'
fires_all.to_file(out1, driver='GeoJSON')
print(f'\n[1] Сохранено: {out1}  ({len(fires_all):,} пожаров)')

print('\n[2] Объединение пожаров не 2005 года...')
non_2005 = fires_all[fires_all['year'] != STUDY_YEAR].copy()
print(f'    Пожаров не из {STUDY_YEAR}: {len(non_2005):,}')

non_2005_m = non_2005.to_crs(METRIC_CRS)

non_2005_m['geometry'] = non_2005_m.geometry.buffer(0)

print('    unary_union (может занять несколько минут)...')
dissolved_geom = unary_union(list(non_2005_m.geometry.values))
print(f'    Тип результата: {dissolved_geom.geom_type}')

non_2005_dissolved = gpd.GeoDataFrame(
    {'description': [f'Все пожары 2001–2024 кроме {STUDY_YEAR}, объединены']},
    geometry=[dissolved_geom],
    crs=METRIC_CRS,
)

out2 = OUTPUT_DIR / 'fires_non2005_dissolved.geojson'
non_2005_dissolved.to_crs(OUTPUT_CRS).to_file(out2, driver='GeoJSON')

dissolved_area_ha = non_2005_dissolved.geometry.area.iloc[0] / 10_000
print(f'    Площадь объединённого слоя: {dissolved_area_ha:,.0f} га')
print(f'[2] Сохранено: {out2}')

print('\n[3] Геометрическое пересечение — территории повторных пожаров...')

fires_2005 = fires_all[fires_all['year'] == STUDY_YEAR].copy().to_crs(METRIC_CRS)
fires_2005['geometry'] = fires_2005.geometry.buffer(0)

if 'Area' in fires_2005.columns:
    fires_2005 = fires_2005[fires_2005['Area'] >= AREA_THRESHOLD_HA].copy()
    print(f'    Пожаров {STUDY_YEAR} с Area >= {AREA_THRESHOLD_HA} га: {len(fires_2005):,}')
else:
    print('    Колонка Area не найдена, без фильтра по площади.')

total_2005_ha = fires_2005.geometry.area.sum() / 10_000

repeats = gpd.overlay(
    fires_2005,
    non_2005_dissolved,
    how='intersection',
    keep_geom_type=False,
)
repeats = repeats[repeats.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
repeats['repeat_area_ha'] = repeats.geometry.area / 10_000

out3 = OUTPUT_DIR / 'fires_2005_repeats.geojson'
repeats.to_crs(OUTPUT_CRS).to_file(out3, driver='GeoJSON')

total_repeat_ha = repeats['repeat_area_ha'].sum()
share = total_repeat_ha / total_2005_ha * 100 if total_2005_ha else 0

print(f'    Полигонов после пересечения:    {len(repeats):,}')
print(f'    Уникальных fire_id с повтором:  {repeats["fire_id"].nunique():,}')
print(f'    Общая площадь 2005:             {total_2005_ha:,.0f} га')
print(f'    Площадь повторно горевших:      {total_repeat_ha:,.0f} га')
print(f'    Доля повторно горевших:         {share:.1f}%')
print(f'[3] Сохранено: {out3}')

print('\n=== Готово ===')
print(f'  1. {out1}')
print(f'  2. {out2}')
print(f'  3. {out3}')
