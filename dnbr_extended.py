"""
Неделя 2, шаг 2b: расширенный dNBR-анализ.

Что добавлено к шагу 2:
1. NBR_pre и NBR_post по каждой зоне — диагностика «нечему гореть» в повторных.
2. RBR (Relativized Burn Ratio, Parks 2014): RBR = dNBR / (NBR_pre + 1.001).
   Нормализован относительно исходного состояния, корректнее для повторных пожаров.
3. dNBR ∩ Hansen forest mask — честный знаменатель для cross-check с Hansen.
4. Расширенный CSV: dnbr_extended_stats.csv (старый dnbr_stats_per_fire.csv не трогаем).

Запуск: python week2_step2b_dnbr_extended.py
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

PRE_YEAR, POST_YEAR = 2005, 2006
COMPOSITE_START, COMPOSITE_END = '06-15', '08-15'
CLOUD_COVER_MAX = 30
BURN_THRESHOLD = 0.27

HANSEN_ID = 'UMD/hansen/global_forest_change_2025_v1_13'
FOREST_THRESHOLD = 30

ee.Initialize(project=GEE_PROJECT)
print(f'GEE инициализирован: {GEE_PROJECT}')

def mask_landsat_sr(image):
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(int('11111', 2)).eq(0)
    optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(optical, None, True).updateMask(mask)

def get_l5_composite(year, region):
    return (
        ee.ImageCollection('LANDSAT/LT05/C02/T1_L2')
        .filterDate(f'{year}-{COMPOSITE_START}', f'{year}-{COMPOSITE_END}')
        .filterBounds(region)
        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
        .map(mask_landsat_sr)
        .median()
    )

def compute_nbr(image):
    nir = image.select('SR_B4')
    swir2 = image.select('SR_B7')
    return nir.subtract(swir2).divide(nir.add(swir2))

print('\n=== Загрузка слоёв ===')
fires_by_year = gpd.read_file(OUTPUT_DIR / 'fires_by_year.geojson')
fires_2005 = fires_by_year[fires_by_year['year'] == 2005]
fires_2005 = fires_2005[fires_2005['Area'] >= 500].copy()
repeats = gpd.read_file(OUTPUT_DIR / 'fires_2005_repeats.geojson')
clean = gpd.read_file(OUTPUT_DIR / 'fires_2005_clean.geojson')
print(f'  2005: {len(fires_2005)}, repeats: {len(repeats)}, clean: {len(clean)}')

fires_fc_full = geemap.geopandas_to_ee(fires_2005[['fire_id', 'geometry']])
all_bbox = ee.FeatureCollection(fires_fc_full).geometry().bounds()

print('\n=== Композиты Landsat 5 ===')
pre_img = get_l5_composite(PRE_YEAR, all_bbox)
post_img = get_l5_composite(POST_YEAR, all_bbox)

nbr_pre = compute_nbr(pre_img)
nbr_post = compute_nbr(post_img)
dnbr = nbr_pre.subtract(nbr_post)

rbr = dnbr.divide(nbr_pre.add(1.001))

common_mask = dnbr.mask()
nbr_pre = nbr_pre.updateMask(common_mask).rename('NBR_pre')
nbr_post = nbr_post.updateMask(common_mask).rename('NBR_post')
dnbr = dnbr.rename('dNBR')
rbr = rbr.updateMask(common_mask).rename('RBR')

print('\n=== Hansen forest mask ===')
hansen = ee.Image(HANSEN_ID)
forest_mask = (
    hansen.select('treecover2000').gte(FOREST_THRESHOLD)
    .And(hansen.select('lossyear').eq(0).Or(hansen.select('lossyear').gte(5)))
    .And(hansen.select('datamask').eq(1))
)
print(f'  Asset: {HANSEN_ID}')

area = ee.Image.pixelArea()

real_burn_dnbr        = dnbr.gte(BURN_THRESHOLD)
real_burn_dnbr_forest = real_burn_dnbr.And(forest_mask)
real_burn_rbr         = rbr.gte(BURN_THRESHOLD)
real_burn_rbr_forest  = real_burn_rbr.And(forest_mask)

area_stack = ee.Image.cat([
    real_burn_dnbr       .multiply(area).rename('m2_dnbr'),
    real_burn_dnbr_forest.multiply(area).rename('m2_dnbr_forest'),
    real_burn_rbr        .multiply(area).rename('m2_rbr'),
    real_burn_rbr_forest .multiply(area).rename('m2_rbr_forest'),
])

mean_stack = ee.Image.cat([nbr_pre, nbr_post, dnbr, rbr])

M2_COLS = ['m2_dnbr', 'm2_dnbr_forest', 'm2_rbr', 'm2_rbr_forest']
MEAN_COLS = ['NBR_pre', 'NBR_post', 'dNBR', 'RBR']

def _to_df(fc_result, fields):
    """Парсинг features → DataFrame с fire_id и заданными полями."""
    rows = []
    for f in fc_result.getInfo()['features']:
        p = f['properties']
        row = {'fire_id': p.get('fire_id')}
        for k in fields:
            v = p.get(k)
            row[k] = float(v) if v is not None else 0.0
        rows.append(row)
    return pd.DataFrame(rows)

def compute_zone(gdf: gpd.GeoDataFrame, zone_name: str, suffix: str) -> pd.DataFrame:
    print(f'  Зона «{zone_name}»: {len(gdf)} полигонов...')

    gdf_wgs = gdf.to_crs(WGS84).copy()
    keep = [c for c in ['fire_id', 'geometry'] if c in gdf_wgs.columns]
    gdf_wgs = gdf_wgs[keep]
    gdf_wgs = gdf_wgs[~gdf_wgs.geometry.is_empty & gdf_wgs.geometry.is_valid]
    fc = geemap.geopandas_to_ee(gdf_wgs)

    areas_fc = area_stack.reduceRegions(
        collection=fc, reducer=ee.Reducer.sum(),
        scale=30, crs=WGS84, tileScale=4,
    )

    means_fc = mean_stack.reduceRegions(
        collection=fc, reducer=ee.Reducer.mean(),
        scale=30, crs=WGS84, tileScale=4,
    )

    count_fc = dnbr.reduceRegions(
        collection=fc, reducer=ee.Reducer.count(),
        scale=30, crs=WGS84, tileScale=4,
    )

    df_a = _to_df(areas_fc, M2_COLS)
    df_m = _to_df(means_fc, MEAN_COLS)
    df_c = _to_df(count_fc, ['count'])

    df_a = df_a.groupby('fire_id', as_index=False)[M2_COLS].sum()

    df_mc = df_m.merge(df_c, on='fire_id', how='left')
    df_mc['count'] = df_mc['count'].fillna(0)
    for col in MEAN_COLS:
        df_mc[f'_w_{col}'] = df_mc[col] * df_mc['count']

    agg = df_mc.groupby('fire_id', as_index=False).agg(
        **{f'_w_{c}': (f'_w_{c}', 'sum') for c in MEAN_COLS},
        _count=('count', 'sum'),
    )
    for col in MEAN_COLS:
        agg[f'{col}_mean'] = agg[f'_w_{col}'] / agg['_count'].replace(0, pd.NA)

    result = df_a.merge(
        agg[['fire_id', '_count'] + [f'{c}_mean' for c in MEAN_COLS]],
        on='fire_id', how='outer',
    )

    rename = {
        'm2_dnbr':        f'burn_dnbr_{suffix}_ha',
        'm2_dnbr_forest': f'burn_dnbr_forest_{suffix}_ha',
        'm2_rbr':         f'burn_rbr_{suffix}_ha',
        'm2_rbr_forest':  f'burn_rbr_forest_{suffix}_ha',
    }
    for old, new in rename.items():
        if old in result.columns:
            result[new] = result[old] / 10_000
    result = result.drop(columns=list(rename.keys()), errors='ignore')
    result = result.rename(columns={
        'NBR_pre_mean':  f'nbr_pre_{suffix}',
        'NBR_post_mean': f'nbr_post_{suffix}',
        'dNBR_mean':     f'dnbr_{suffix}',
        'RBR_mean':      f'rbr_{suffix}',
        '_count':        f'pixels_{suffix}',
    })
    return result

print('\n=== reduceRegions по зонам ===')
df_total   = compute_zone(fires_2005, 'вся гарь 2005', 'total')
df_repeats = compute_zone(repeats,    'повторные',     'repeats')
df_clean   = compute_zone(clean,      'чистые',        'clean')

df = (
    df_total
    .merge(df_repeats, on='fire_id', how='left')
    .merge(df_clean,   on='fire_id', how='left')
)
df = df.fillna(0).infer_objects(copy=False)

DATA_DIR.mkdir(exist_ok=True)
out_csv = DATA_DIR / 'dnbr_extended_stats.csv'
df.round(4).to_csv(out_csv, index=False)
print(f'\n  Сохранено: {out_csv}')

def wmean(df, prefix, suffix):
    """Взвешенное среднее prefix_suffix по pixels_suffix."""
    vals = df[f'{prefix}_{suffix}']
    weights = df[f'pixels_{suffix}']
    total = (vals * weights).sum()
    n = weights.sum()
    return total / n if n else float('nan')

print('\n--- Средние индексы по зонам (взвешено по площади) ---')
print(f'{"":>10} {"NBR_pre":>10} {"NBR_post":>10} {"dNBR":>10} {"RBR":>10}')
print('-' * 54)
for suffix, label in [('total', 'Total'), ('repeats', 'Repeats'), ('clean', 'Clean')]:
    print(f'{label:>10} '
          f'{wmean(df, "nbr_pre",  suffix):>10.3f} '
          f'{wmean(df, "nbr_post", suffix):>10.3f} '
          f'{wmean(df, "dnbr",     suffix):>10.3f} '
          f'{wmean(df, "rbr",      suffix):>10.3f}')

print('\n--- Площади «настоящей гари» (га, суммарно) ---')
print(f'{"":>10} {"dNBR all":>12} {"dNBR forest":>12} {"RBR all":>12} {"RBR forest":>12}')
print('-' * 62)
for suffix, label in [('total', 'Total'), ('repeats', 'Repeats'), ('clean', 'Clean')]:
    d  = df[f'burn_dnbr_{suffix}_ha'].sum()
    df_ = df[f'burn_dnbr_forest_{suffix}_ha'].sum()
    r  = df[f'burn_rbr_{suffix}_ha'].sum()
    rf = df[f'burn_rbr_forest_{suffix}_ha'].sum()
    print(f'{label:>10} {d:>12,.0f} {df_:>12,.0f} {r:>12,.0f} {rf:>12,.0f}')

print('\n--- Repeats vs Clean (отношение) ---')
for prefix, label in [('nbr_pre', 'NBR_pre'), ('nbr_post', 'NBR_post'),
                      ('dnbr', 'dNBR'), ('rbr', 'RBR')]:
    r = wmean(df, prefix, 'repeats')
    c = wmean(df, prefix, 'clean')
    diff = (r - c) / max(abs(c), 1e-6) * 100
    note = ''
    if prefix == 'rbr':
        note = ' ← ожидается > 0 при правильной нормализации'
    print(f'  {label:>10}: repeats={r:+.3f}, clean={c:+.3f}, diff={diff:+.1f}%{note}')

hansen_csv = DATA_DIR / 'forest_stats_per_fire.csv'
if hansen_csv.exists():
    h = pd.read_csv(hansen_csv)
    hansen_lost = h['forest_lost_total_ha'].sum()
    print(f'\n--- Cross-check с Hansen (forest_lost = {hansen_lost:,.0f} га) ---')
    for col, label in [
        ('burn_dnbr_total_ha',        'dNBR, все пиксели'),
        ('burn_dnbr_forest_total_ha', 'dNBR, только лес (Hansen-mask)'),
        ('burn_rbr_total_ha',         'RBR,  все пиксели'),
        ('burn_rbr_forest_total_ha',  'RBR,  только лес'),
    ]:
        v = df[col].sum()
        ratio = v / hansen_lost if hansen_lost else float('inf')
        print(f'  {label:<35} {v:>10,.0f} га   → {ratio:.2f}×')

print(f'\n=== Готово ===')
print(f'  {out_csv}')
