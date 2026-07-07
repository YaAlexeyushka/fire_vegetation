"""
Неделя 2, шаг 2: dNBR + строгий отбор «настоящих» гарей.

Что делает:
1. Строит pre-fire композит (Landsat 5, лето 2005, до сентябрьских пожаров)
   и post-fire композит (Landsat 5, лето 2006).
2. Считает NBR_pre, NBR_post и dNBR = NBR_pre - NBR_post.
3. Классифицирует по Key & Benson 2006:
       dNBR < 0.10        : unburned
       0.10 <= dNBR < 0.27 : low severity
       0.27 <= dNBR < 0.44 : moderate-low
       0.44 <= dNBR < 0.66 : moderate-high
       dNBR >= 0.66       : high
   «Настоящая гарь» = dNBR >= 0.27 (порог Key & Benson по умолчанию).
4. Через reduceRegions считает площадь по каждому классу + средневзвешенный
   dNBR в трёх зонах (total / repeats / clean) для каждого пожара.
5. Сохраняет data/dnbr_stats_per_fire.csv.

Опционально: экспорт регионального dNBR-растра в Drive (одним файлом для
всей коллекции пожаров) — для пиксельного анализа и дашборда на неделе 4.

Запуск: python week2_step2_dnbr.py
Требования: ee.Authenticate() выполнен; есть output/fires_by_year.geojson,
            output/fires_2005_repeats.geojson, output/fires_2005_clean.geojson.
"""
import ee
import geemap
import geopandas as gpd
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path('output')
DATA_DIR = Path('data')

WGS84 = 'EPSG:4326'

GEE_PROJECT = 'bubbly-operator-479705-v3'
STUDY_YEAR = 2005
AREA_THRESHOLD_HA = 500

PRE_YEAR = 2005
POST_YEAR = 2006
COMPOSITE_START = '06-15'
COMPOSITE_END = '08-15'
CLOUD_COVER_MAX = 30

BURN_THRESHOLD = 0.27

EXPORT_REGIONAL_RASTER = False
DRIVE_FOLDER = 'fire_dnbr'

ee.Initialize(project=GEE_PROJECT)
print(f'GEE инициализирован: {GEE_PROJECT}')

def mask_landsat_sr(image):
    """Cloud mask + scale factor для Landsat C2 L2 (SR)."""
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(int('11111', 2)).eq(0)
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(mask)

def get_l5_composite(year, region):
    """Median composite Landsat 5 за летний период с маскированием облаков."""
    col = (
        ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        .filterDate(f'{year}-{COMPOSITE_START}', f'{year}-{COMPOSITE_END}')
        .filterBounds(region)
        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
        .map(mask_landsat_sr)
    )
    return col.median(), col

def compute_nbr(image):
    """NBR = (NIR - SWIR2) / (NIR + SWIR2) для Landsat 5 (B4, B7)."""
    nir = image.select('SR_B4')
    swir2 = image.select('SR_B7')
    return nir.subtract(swir2).divide(nir.add(swir2))

print('\n=== Загрузка слоёв Недели 1+2 ===')
fires_by_year = gpd.read_file(OUTPUT_DIR / 'fires_by_year.geojson')
fires_2005 = fires_by_year[fires_by_year['year'] == STUDY_YEAR].copy()
if 'Area' in fires_2005.columns:
    fires_2005 = fires_2005[fires_2005['Area'] >= AREA_THRESHOLD_HA].copy()
print(f'  Пожаров 2005 (Area >= {AREA_THRESHOLD_HA} га): {len(fires_2005)}')

repeats = gpd.read_file(OUTPUT_DIR / 'fires_2005_repeats.geojson')
clean = gpd.read_file(OUTPUT_DIR / 'fires_2005_clean.geojson')
print(f'  Повторных полигонов: {len(repeats)}')
print(f'  Чистых полигонов:    {len(clean)}')

fires_fc_full = geemap.geopandas_to_ee(fires_2005[['fire_id', 'geometry']])
all_bbox = ee.FeatureCollection(fires_fc_full).geometry().bounds()

print(f'\n=== Композиты Landsat 5 ===')
pre_img, pre_col = get_l5_composite(PRE_YEAR, all_bbox)
post_img, post_col = get_l5_composite(POST_YEAR, all_bbox)

pre_count = pre_col.size().getInfo()
post_count = post_col.size().getInfo()
print(f'  Pre  ({PRE_YEAR} {COMPOSITE_START}–{COMPOSITE_END}): {pre_count} снимков')
print(f'  Post ({POST_YEAR} {COMPOSITE_START}–{COMPOSITE_END}): {post_count} снимков')
if pre_count == 0 or post_count == 0:
    raise RuntimeError('Не хватает Landsat 5 снимков в одном из периодов! '
                       'Попробуйте увеличить CLOUD_COVER_MAX или расширить окно.')

nbr_pre = compute_nbr(pre_img).rename('NBR_pre')
nbr_post = compute_nbr(post_img).rename('NBR_post')
dnbr = nbr_pre.subtract(nbr_post).rename('dNBR')

unburned = dnbr.lt(0.10)
low      = dnbr.gte(0.10).And(dnbr.lt(0.27))
mod_low  = dnbr.gte(0.27).And(dnbr.lt(0.44))
mod_high = dnbr.gte(0.44).And(dnbr.lt(0.66))
high     = dnbr.gte(0.66)
real_burn = dnbr.gte(BURN_THRESHOLD)

area = ee.Image.pixelArea()
area_stack = ee.Image.cat([
    unburned.multiply(area).rename('m2_unburned'),
    low.multiply(area).rename('m2_low'),
    mod_low.multiply(area).rename('m2_mod_low'),
    mod_high.multiply(area).rename('m2_mod_high'),
    high.multiply(area).rename('m2_high'),
    real_burn.multiply(area).rename('m2_real_burn'),
])

def compute_zone_stats(gdf: gpd.GeoDataFrame, zone_name: str, suffix: str) -> pd.DataFrame:
    """Площади по классам severity + средневзвешенный dNBR для каждого fire_id."""
    print(f'  Зона «{zone_name}»: {len(gdf)} полигонов...')

    gdf_wgs = gdf.to_crs(WGS84).copy()
    keep = [c for c in ['fire_id', 'geometry'] if c in gdf_wgs.columns]
    gdf_wgs = gdf_wgs[keep]
    gdf_wgs = gdf_wgs[~gdf_wgs.geometry.is_empty & gdf_wgs.geometry.is_valid]
    fc = geemap.geopandas_to_ee(gdf_wgs)

    areas_fc = area_stack.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.sum(),
        scale=30,
        crs=WGS84,
        tileScale=4,
    )

    dnbr_fc = dnbr.reduceRegions(
        collection=fc,
        reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
        scale=30,
        crs=WGS84,
        tileScale=4,
    )

    rows_a = []
    for f in areas_fc.getInfo()['features']:
        p = f['properties']
        rows_a.append({
            'fire_id': p.get('fire_id'),
            f'unburned_{suffix}_ha':  (p.get('m2_unburned')  or 0) / 10_000,
            f'low_{suffix}_ha':       (p.get('m2_low')       or 0) / 10_000,
            f'mod_low_{suffix}_ha':   (p.get('m2_mod_low')   or 0) / 10_000,
            f'mod_high_{suffix}_ha':  (p.get('m2_mod_high')  or 0) / 10_000,
            f'high_{suffix}_ha':      (p.get('m2_high')      or 0) / 10_000,
            f'real_burn_{suffix}_ha': (p.get('m2_real_burn') or 0) / 10_000,
        })
    df_a = pd.DataFrame(rows_a).groupby('fire_id', as_index=False).sum()

    rows_d = []
    for f in dnbr_fc.getInfo()['features']:
        p = f['properties']
        rows_d.append({
            'fire_id': p.get('fire_id'),
            'mean': p.get('mean'),
            'count': p.get('count') or 0,
        })
    df_d = pd.DataFrame(rows_d)

    df_d['mean'] = df_d['mean'].fillna(0)
    df_d['_w'] = df_d['mean'] * df_d['count']
    agg = df_d.groupby('fire_id', as_index=False).agg(_w=('_w', 'sum'), _c=('count', 'sum'))
    agg[f'dnbr_mean_{suffix}'] = agg['_w'] / agg['_c'].replace(0, pd.NA)
    agg[f'valid_pixels_{suffix}'] = agg['_c']
    df_d = agg[['fire_id', f'dnbr_mean_{suffix}', f'valid_pixels_{suffix}']]

    return df_a.merge(df_d, on='fire_id', how='left')

print('\n=== reduceRegions по зонам ===')
df_total   = compute_zone_stats(fires_2005, 'вся гарь 2005', 'total')
df_repeats = compute_zone_stats(repeats,    'повторные',     'repeats')
df_clean   = compute_zone_stats(clean,      'чистые',        'clean')

print('\n=== Сводная таблица ===')

df = (
    df_total
    .merge(df_repeats, on='fire_id', how='left')
    .merge(df_clean,   on='fire_id', how='left')
    .fillna(0)
)

DATA_DIR.mkdir(exist_ok=True)
out_csv = DATA_DIR / 'dnbr_stats_per_fire.csv'
df.round(4).to_csv(out_csv, index=False)
print(f'  Сохранено: {out_csv}')

print(f'\n--- Площади по классам severity (га, суммарно по всем пожарам) ---')
print(f'{"Класс":<14} {"Total":>14} {"Repeats":>14} {"Clean":>14}')
print('-' * 60)
for cls, label in [
    ('unburned',  'Unburned'),
    ('low',       'Low'),
    ('mod_low',   'Mod-low'),
    ('mod_high',  'Mod-high'),
    ('high',      'High'),
    ('real_burn', 'REAL BURN'),
]:
    t = df.get(f'{cls}_total_ha',   pd.Series([0])).sum()
    r = df.get(f'{cls}_repeats_ha', pd.Series([0])).sum()
    c = df.get(f'{cls}_clean_ha',   pd.Series([0])).sum()
    print(f'{label:<14} {t:>14,.0f} {r:>14,.0f} {c:>14,.0f}')

repeat_area_total = repeats['repeat_area_ha'].sum() if 'repeat_area_ha' in repeats.columns else None
clean_area_total = clean['clean_area_ha'].sum() if 'clean_area_ha' in clean.columns else None

real_t = df['real_burn_total_ha'].sum()
real_r = df['real_burn_repeats_ha'].sum()
real_c = df['real_burn_clean_ha'].sum()

print(f'\n--- «Настоящая гарь» (dNBR >= {BURN_THRESHOLD}) ---')
print(f'  Всего:       {real_t:>10,.0f} га')
if repeat_area_total:
    print(f'  В повторных: {real_r:>10,.0f} га '
          f'({real_r / repeat_area_total * 100:.1f}% площади повторных зон)')
if clean_area_total:
    print(f'  В чистых:    {real_c:>10,.0f} га '
          f'({real_c / clean_area_total * 100:.1f}% площади чистых зон)')

def weighted_mean(col_mean, col_count):
    s = (df[col_mean] * df[col_count]).sum()
    n = df[col_count].sum()
    return s / n if n else float('nan')

dnbr_t = weighted_mean('dnbr_mean_total',   'valid_pixels_total')
dnbr_r = weighted_mean('dnbr_mean_repeats', 'valid_pixels_repeats')
dnbr_c = weighted_mean('dnbr_mean_clean',   'valid_pixels_clean')
print(f'\n--- Средневзвешенный dNBR ---')
print(f'  Всего:       {dnbr_t:.3f}')
print(f'  В повторных: {dnbr_r:.3f}')
print(f'  В чистых:    {dnbr_c:.3f}')
diff = (dnbr_r - dnbr_c) / max(abs(dnbr_c), 1e-6) * 100
print(f'  Разница repeats vs clean: {diff:+.1f}%  ← ожидается > 0')

hansen_csv = DATA_DIR / 'forest_stats_per_fire.csv'
if hansen_csv.exists():
    print(f'\n--- Cross-check с Hansen ---')
    h = pd.read_csv(hansen_csv)
    hansen_lost_total = h['forest_lost_total_ha'].sum()
    print(f'  dNBR real_burn (>={BURN_THRESHOLD}): {real_t:>10,.0f} га')
    print(f'  Hansen forest_lost (lossyear=5):    {hansen_lost_total:>10,.0f} га')
    ratio = real_t / hansen_lost_total if hansen_lost_total else float('inf')
    print(f'  dNBR / Hansen = {ratio:.2f}×')
    print(f'  (dNBR обычно >= Hansen, потому что покрывает low-severity ожоги')
    print(f'   и не привязан к «первому» событию)')

if EXPORT_REGIONAL_RASTER:
    print(f'\n=== Экспорт регионального dNBR-растра в Drive ===')
    fires_geom = ee.FeatureCollection(fires_fc_full).geometry()
    fires_mask = ee.Image.constant(1).clip(fires_geom)
    dnbr_export = dnbr.updateMask(fires_mask).toFloat()

    task = ee.batch.Export.image.toDrive(
        image=dnbr_export,
        description='dnbr_2005_irkutsk',
        folder=DRIVE_FOLDER,
        fileNamePrefix='dnbr_2005_irkutsk',
        region=fires_geom.bounds(),
        scale=30,
        crs='EPSG:4326',
        maxPixels=1e10,
        fileFormat='GeoTIFF',
    )
    task.start()
    print(f'  Задача запущена: id={task.id}')
    print(f'  Проверка статуса: ee.batch.Task.list()')

print('\n=== Готово ===')
print(f'  {out_csv}')
