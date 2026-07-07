"""
Быстрая проверка: есть ли плато в post-fire восстановлении?

Разбивает 2006-2024 на 5-летние клины (плюс pre-fire baseline).
Если VI в поздних клинах ≈ равны — плато подтверждено.
"""
import pandas as pd
from pathlib import Path

vi = pd.read_csv(Path('data') / 'vi_wide_format.csv')

BINS = {
    'pre':     [2002, 2004, 2005],
    '2006-10': list(range(2006, 2011)),
    '2011-15': list(range(2011, 2016)),
    '2016-20': list(range(2016, 2021)),
    '2021-24': list(range(2021, 2025)),
}
BIN_COLS = list(BINS.keys())

for idx in ['NDVI', 'NBR']:
    col = f'{idx}_median'
    result = {b: vi[vi.year.isin(y)].groupby('fire_id')[col].mean() for b, y in BINS.items()}
    df = pd.DataFrame(result).reset_index()
    meta = vi[['fire_id', 'veg_name', 'is_clean']].drop_duplicates('fire_id')
    df = df.merge(meta, on='fire_id', how='left')

    print(f'\n=== {idx}: средние по 5-летним клинам ===')
    by_veg = df.groupby('veg_name').agg(n=('fire_id', 'count'), **{b: (b, 'mean') for b in BIN_COLS})
    print(by_veg.round(3).to_string())

    df['change_late'] = (df['2021-24'] - df['2016-20']) / df['2016-20'].replace(0, pd.NA)
    df['change_mid']  = (df['2016-20'] - df['2011-15']) / df['2011-15'].replace(0, pd.NA)

    print(f'\n{idx}: относительное изменение поздних клинов (%)')
    stats = df.groupby('veg_name').agg(
        n=('fire_id', 'count'),
        change_2011_to_2016=('change_mid', lambda x: x.mean() * 100),
        change_2016_to_2021=('change_late', lambda x: x.mean() * 100),
    ).round(1)
    print(stats.to_string())

    print(f'\n{idx}: средние по клинам, clean vs dirty')
    cd = df.groupby('is_clean').agg(**{b: (b, 'mean') for b in BIN_COLS}).round(3)
    print(cd.to_string())

    plateau = df['change_late'].abs() < 0.03
    print(f'\n{idx}: пожаров на плато к 2024 (|Δ| < 3%): '
          f'{plateau.sum()}/{plateau.notna().sum()} ({plateau.mean() * 100:.0f}%)')
