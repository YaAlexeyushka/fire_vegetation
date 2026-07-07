"""
Быстрая проверка: нужен ли LandTrendr?

Смотрит на R² линейных регрессий восстановления. Правило:
  R² > 0.7        — сильный линейный тренд, LandTrendr не нужен
  R² 0.4–0.7     — умеренный, LandTrendr косметически улучшит
  R² < 0.4        — слабый, LandTrendr или полезен, или тренд нелинейный
"""
import pandas as pd
from pathlib import Path

df = pd.read_csv(Path('data') / 'recovery_per_fire.csv')

for idx in ['NDVI', 'NBR']:
    col = f'{idx}_r_squared'
    r2 = df[col].dropna()
    print(f'\n=== {idx} R² ===')
    print(f'  n:      {len(r2)}')
    print(f'  mean:   {r2.mean():.3f}')
    print(f'  median: {r2.median():.3f}')
    print(f'  min:    {r2.min():.3f}')
    print(f'  max:    {r2.max():.3f}')
    print(f'\n  Распределение по бинам:')
    print(f'    R² > 0.7 (сильный тренд):    {(r2 > 0.7).sum():>3} ({(r2 > 0.7).mean()*100:.0f}%)')
    print(f'    R² 0.4–0.7 (умеренный):      {((r2 >= 0.4) & (r2 <= 0.7)).sum():>3} ({((r2 >= 0.4) & (r2 <= 0.7)).mean()*100:.0f}%)')
    print(f'    R² < 0.4 (слабый):           {(r2 < 0.4).sum():>3} ({(r2 < 0.4).mean()*100:.0f}%)')

    if 'is_clean' in df.columns:
        clean_r2 = df[df.is_clean == True][col].dropna()
        dirty_r2 = df[df.is_clean == False][col].dropna()
        print(f'\n  Clean (n={len(clean_r2)}):  mean R² = {clean_r2.mean():.3f}')
        print(f'  Dirty (n={len(dirty_r2)}): mean R² = {dirty_r2.mean():.3f}')

ndvi_med = df['NDVI_r_squared'].median()
nbr_med = df['NBR_r_squared'].median()
avg = (ndvi_med + nbr_med) / 2

print(f'\n{"="*60}')
print(f'Вердикт (медиана R² по обоим индексам: {avg:.3f}):')
if avg > 0.7:
    print(f'  LandTrendr НЕ НУЖЕН. Линейная регрессия отражает динамику')
    print(f'  корректно, доп. сложность не оправдана.')
elif avg > 0.4:
    print(f'  LandTrendr ПОЛЕЗЕН, но опционально. Косметическое улучшение,')
    print(f'  особенно для выявления года перелома тренда.')
else:
    print(f'  LandTrendr НУЖЕН. Линейный тренд слабый — либо шум,')
    print(f'  либо нелинейное восстановление (полка → рост / плато).')
