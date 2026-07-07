"""
Дашборд: пожары в Иркутской области 2005 г. и восстановление растительности.

Запуск: streamlit run app.py

Структура:
    1. Обзор — ключевые цифры
    2. Карта пожаров — год-слайдер + слои повторных/чистых; отдельный
       режим — раскраска гарей 2005 по годам восстановления
    3. Ущерб — три сценария (все / без повторных / только повторные) +
       сравнение Hansen/dNBR/RBR
    4. Восстановление — выбор пожара, timeline индексов, метрики
    5. Смена типов (MODIS Land Cover) — динамика типов растительности
    6. Скорость восстановления — корреляция с площадью гари
"""
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

DATA_DIR = Path('data')
OUTPUT_DIR = Path('output')
STUDY_YEAR_MAP = 2005

st.set_page_config(
    page_title='Иркутск: пожары 2005 и восстановление',
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🔥',
)

@st.cache_data
def load_csv(fname):
    return pd.read_csv(DATA_DIR / fname)

@st.cache_data
def load_excel(fname):
    return pd.read_excel(DATA_DIR / fname)

@st.cache_data
def load_geojson(fname):
    return gpd.read_file(OUTPUT_DIR / fname)

def to_folium_json(gdf):
    """Сериализация GDF в JSON для folium — обходит проблемы с Timestamp."""
    g = gdf.copy()
    for col in g.select_dtypes(include=['datetime64[ns]', 'datetime64', 'datetimetz']).columns:
        g[col] = g[col].astype(str)

    for col in g.columns:
        if col == 'geometry':
            continue
        if g[col].dtype == 'object':

            sample = g[col].dropna().head(1)
            if len(sample) and hasattr(sample.iloc[0], 'isoformat'):
                g[col] = g[col].astype(str)
    return g.to_json()

def try_load(loader, fname):
    try:
        return loader(fname)
    except Exception as e:
        st.warning(f'Не удалось загрузить {fname}: {e}')
        return None

damage      = try_load(load_csv,   'damage_per_fire.csv')
damage_agg  = try_load(load_csv,   'damage_aggregate.csv')
recovery    = try_load(load_csv,   'recovery_per_fire.csv')
forest      = try_load(load_csv,   'forest_stats_per_fire.csv')
dnbr_ext    = try_load(load_csv,   'dnbr_extended_stats.csv')
vi          = try_load(load_csv,   'vi_wide_format.csv')
veg_year    = try_load(load_excel, 'vegetation_area_by_year.xlsx')
veg_shift   = try_load(load_csv,   'vegetation_shift_summary.csv')
corr        = try_load(load_csv,   'correlation_matrix.csv')

st.sidebar.markdown('# 🔥 Дашборд')
st.sidebar.markdown('Пожары 2005 г. в Иркутской области и восстановление растительности')
st.sidebar.markdown('---')

page = st.sidebar.radio(
    'Раздел',
    [
        '📊 Обзор',
        '🗺️ Карта пожаров',
        '💥 Ущерб',
        '🌱 Восстановление',
        '🌲 Смена типов',
        '📈 Корреляции',
    ],
)

st.sidebar.markdown('---')
st.sidebar.caption('Данные: MODIS MCD64A1, Landsat 5, Hansen GFC v1.13, MODIS Land Cover')

if page.startswith('📊'):
    st.title('Пожары 2005 г. в Иркутской области')
    st.markdown('Анализ восстановления растительности после катастрофических пожаров сентября 2005 г.')

    st.markdown('## Ключевые показатели')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Пожаров 2005', '159', '≥ 500 га')
    col2.metric('Общая площадь', '379 420 га', 'полигоны MODIS')
    col3.metric('Повторные зоны', '67.3%', '255 235 га')
    col4.metric('Чистые зоны', '32.7%', '124 185 га')

    col5, col6, col7, col8 = st.columns(4)
    col5.metric('Лес до пожара', '21.4%', '81 144 га (Hansen)')
    col6.metric('Настоящая гарь', '17 129 га', 'RBR ∩ forest')
    col7.metric('Fuel limitation', '1.61×', 'clean > repeats')
    col8.metric('Смена типов', '−67% / +355%', 'хвойные / лиственные')

    st.markdown('---')

elif page.startswith('🗺️'):
    st.title('Карта пожаров')

    map_mode = st.radio(
        'Режим карты:',
        ['По годам (все пожары)', 'Восстановление (гари 2005)'],
        horizontal=True,
    )

    fires_by_year = try_load(load_geojson, 'fires_by_year.geojson')
    if fires_by_year is None:
        st.error('fires_by_year.geojson не найден')
        st.stop()

    if map_mode == 'По годам (все пожары)':
        col1, col2 = st.columns([2, 1])
        with col1:
            year = st.slider('Год', 2001, 2024, 2005, key='map_year')
        with col2:
            show_repeats = st.checkbox('Повторные зоны 2005', True)
            show_clean = st.checkbox('Чистые зоны 2005', True)
            show_all_2005 = st.checkbox('Все пожары 2005 фоном', year != 2005)

        fires_year = fires_by_year[fires_by_year['year'] == year]
        st.markdown(f'**Пожаров в {year} году:** {len(fires_year)}')

        m = folium.Map(location=[55, 105], zoom_start=6, tiles='OpenStreetMap')

        if show_all_2005 and year != 2005:
            fires_2005 = fires_by_year[fires_by_year['year'] == 2005]
            folium.GeoJson(
                to_folium_json(fires_2005),
                name='Пожары 2005 (фон)',
                style_function=lambda x: {
                    'fillColor': '#ffcc80', 'color': '#ff9800',
                    'weight': 0.5, 'fillOpacity': 0.3,
                },
            ).add_to(m)

        year_color = '#e53935' if year == 2005 else '#fb8c00'
        tooltip_fields = [c for c in ['fire_id', 'Area'] if c in fires_year.columns]
        folium.GeoJson(
            to_folium_json(fires_year),
            name=f'Пожары {year}',
            style_function=lambda x, c=year_color: {
                'fillColor': c, 'color': c,
                'weight': 1, 'fillOpacity': 0.5,
            },
            tooltip=folium.GeoJsonTooltip(fields=tooltip_fields) if tooltip_fields else None,
        ).add_to(m)

        if show_repeats:
            repeats = try_load(load_geojson, 'fires_2005_repeats.geojson')
            if repeats is not None:
                folium.GeoJson(
                    to_folium_json(repeats),
                    name='Повторные зоны',
                    style_function=lambda x: {
                        'fillColor': '#d32f2f', 'color': '#b71c1c',
                        'weight': 1, 'fillOpacity': 0.6,
                    },
                ).add_to(m)

        if show_clean:
            clean = try_load(load_geojson, 'fires_2005_clean.geojson')
            if clean is not None:
                folium.GeoJson(
                    to_folium_json(clean),
                    name='Чистые зоны',
                    style_function=lambda x: {
                        'fillColor': '#43a047', 'color': '#1b5e20',
                        'weight': 1, 'fillOpacity': 0.6,
                    },
                ).add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(
            m, width=1200, height=600, returned_objects=[],
            key=f'map_by_year_{year}_{show_repeats}_{show_clean}_{show_all_2005}',
        )

    else:
        if recovery is None:
            st.error('recovery_per_fire.csv не найден')
            st.stop()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            index_choice = st.selectbox('Индекс:', ['NDVI', 'NBR'])
        with col2:
            threshold_pct = st.slider('Порог восстановления, %', 50, 100, 90, step=5)
        with col3:
            scope = st.radio(
                'Учитывать повторные пожары:',
                ['Все пожары', 'Только чистые', 'Только повторные'],
                index=0,
            )

        rec = recovery.copy()
        if scope == 'Только чистые' and 'is_clean' in rec.columns:
            rec = rec[rec.is_clean == True]
        elif scope == 'Только повторные' and 'is_clean' in rec.columns:
            rec = rec[rec.is_clean == False]

        base_col  = f'{index_choice}_baseline'
        min_col   = f'{index_choice}_post_min'
        year_col  = f'{index_choice}_min_year'
        slope_col = f'{index_choice}_slope'
        cur_col   = f'{index_choice}_current'

        needed_cols = [base_col, min_col, year_col, slope_col]
        missing = [c for c in needed_cols if c not in rec.columns]
        if missing:
            st.error(f'В recovery_per_fire.csv нет колонок: {missing}')
            st.stop()

        target = rec[base_col] * (threshold_pct / 100.0)
        with np.errstate(divide='ignore', invalid='ignore'):
            year_reach = np.where(
                rec[slope_col] > 0,
                rec[year_col] + (target - rec[min_col]) / rec[slope_col],
                np.nan,
            )
        rec['years_to_threshold'] = year_reach - STUDY_YEAR_MAP

        rec['years_to_threshold'] = rec['years_to_threshold'].clip(lower=0)

        HORIZON_YEARS = 2 * (2024 - STUDY_YEAR_MAP)
        rec.loc[rec['years_to_threshold'] > HORIZON_YEARS, 'years_to_threshold'] = np.nan

        if cur_col in rec.columns:
            rec['loss_now_pct'] = (
                (1 - rec[cur_col] / rec[base_col]).clip(lower=0) * 100
            )
        else:
            rec['loss_now_pct'] = np.nan

        fires_2005_geom = fires_by_year[fires_by_year['year'] == 2005].copy()
        if 'Area' in fires_2005_geom.columns:
            fires_2005_geom = fires_2005_geom[fires_2005_geom['Area'] >= 500]

        merged = fires_2005_geom.merge(
            rec[['fire_id', 'years_to_threshold', 'loss_now_pct']],
            on='fire_id', how='inner',
        )
        st.markdown(f'**Пожаров в выборке ({scope}):** {len(merged)}')

        if merged.empty:
            st.warning('Нет пожаров в этой выборке.')
        else:

            CAP_YEARS = 15.0

            def years_to_hex(y, cap=CAP_YEARS):
                if pd.isna(y):
                    return '#9e9e9e'
                t = min(max(y / cap, 0), 1)

                if t < 0.5:
                    tt = t / 0.5
                    r = int(0x2e + (0xfd - 0x2e) * tt)
                    g = int(0x7d + (0xd8 - 0x7d) * tt)
                    b = int(0x32 + (0x35 - 0x32) * tt)
                else:
                    tt = (t - 0.5) / 0.5
                    r = int(0xfd + (0xc6 - 0xfd) * tt)
                    g = int(0xd8 + (0x28 - 0xd8) * tt)
                    b = int(0x35 + (0x28 - 0x35) * tt)
                return f'#{r:02x}{g:02x}{b:02x}'

            merged['map_color'] = merged['years_to_threshold'].apply(years_to_hex)
            merged['years_label'] = merged['years_to_threshold'].apply(
                lambda y: f'{y:.1f} лет' if pd.notna(y)
                else f'не достигнут за {HORIZON_YEARS:.0f} лет'
            )
            merged['loss_label'] = merged['loss_now_pct'].apply(
                lambda v: f'{v:.1f}%' if pd.notna(v) else 'N/A'
            )

            m = folium.Map(location=[55, 105], zoom_start=6, tiles='OpenStreetMap')
            gdf_json = to_folium_json(merged)
            folium.GeoJson(
                gdf_json,
                name='Восстановление гарей 2005',
                style_function=lambda feat: {
                    'fillColor': feat['properties']['map_color'],
                    'color': '#333333', 'weight': 0.5, 'fillOpacity': 0.75,
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['fire_id', 'years_label', 'loss_label'],
                    aliases=['Пожар', f'Лет до {threshold_pct}%', 'Потеря сейчас'],
                ),
            ).add_to(m)
            st_folium(
                m, width=1200, height=550, returned_objects=[],
                key=f'map_recovery_{index_choice}_{threshold_pct}_{scope}',
            )

            st.caption(
                f'Цвет: зелёный (0 лет) → жёлтый → красный ({CAP_YEARS:.0f}+ лет, фиксированная шкала). '
                f'Серый — порог {threshold_pct}% не достигается ни при отсутствии роста тренда, '
                f'ни в пределах {HORIZON_YEARS:.0f} лет (2× период наблюдений 2006–2024) — '
                f'более длинная экстраполяция линейного тренда считается ненадёжной.'
            )

            st.markdown('---')

            st.markdown('### Сводная статистика')
            reached = merged['years_to_threshold'].dropna()
            not_reached_n = merged['years_to_threshold'].isna().sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric('Медиана лет до порога', f'{reached.median():.1f}' if len(reached) else 'N/A')
            c2.metric('Среднее лет до порога', f'{reached.mean():.1f}' if len(reached) else 'N/A')
            c3.metric(f'Не достигли {threshold_pct}%', f'{not_reached_n} из {len(merged)}')
            c4.metric('Средняя потеря сейчас', f'{merged["loss_now_pct"].mean():.1f}%')

            st.markdown('---')

            st.markdown('### Распределение времени восстановления')
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=reached, nbinsx=15,
                marker_color='#43a047',
                name=f'{scope}',
            ))
            fig.update_layout(
                xaxis_title=f'Лет до достижения {threshold_pct}% baseline ({index_choice})',
                yaxis_title='Число пожаров',
                height=400,
                title=f'{scope}: n={len(reached)} восстановились, {not_reached_n} нет',
            )
            st.plotly_chart(fig, use_container_width=True)

elif page.startswith('💥'):
    st.title('Ущерб от пожаров 2005 г.')
    st.markdown('Оценка через три метрики: Hansen forest_lost (независимый бенчмарк), dNBR, RBR.')

    if damage_agg is None:
        st.error('damage_aggregate.csv не найден')
        st.stop()

    scenario = st.radio(
        'Сценарий:',
        ['Со всеми зонами', 'Без повторных (только чистые)', 'Только повторные'],
        horizontal=True,
    )
    scenario_map = {
        'Со всеми зонами':                'with_repeats',
        'Без повторных (только чистые)':  'without_repeats',
        'Только повторные':                'only_repeats',
    }
    sc_id = scenario_map[scenario]
    row = damage_agg[damage_agg.scenario_id == sc_id].iloc[0]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Пожаров в сценарии', int(row['n_fires_with_zone']))
    col2.metric('Площадь', f'{row["area_total_ha"]:,.0f} га')
    col3.metric('Гарь (RBR ∩ forest)', f'{row["burn_rbr_forest_ha"]:,.0f} га')
    col4.metric('% повреждённого леса', f'{row["damage_share_forest"] * 100:.1f}%')

    st.markdown('---')

    st.markdown('### Три оценки ущерба (суммарно по всем пожарам)')

    if forest is not None and dnbr_ext is not None:
        metrics_df = pd.DataFrame({
            'Метрика': [
                'Hansen forest_lost',
                'dNBR (все пиксели)',
                'dNBR ∩ forest',
                'RBR (все пиксели)',
                'RBR ∩ forest',
            ],
            'Площадь, га': [
                forest['forest_lost_total_ha'].sum(),
                dnbr_ext['burn_dnbr_total_ha'].sum(),
                dnbr_ext['burn_dnbr_forest_total_ha'].sum(),
                dnbr_ext['burn_rbr_total_ha'].sum(),
                dnbr_ext['burn_rbr_forest_total_ha'].sum(),
            ],
        })
        colors = ['#795548', '#ff7043', '#e64a19', '#42a5f5', '#1976d2']
        fig = go.Figure(go.Bar(
            x=metrics_df['Метрика'], y=metrics_df['Площадь, га'],
            marker_color=colors,
            text=metrics_df['Площадь, га'].apply(lambda x: f'{x:,.0f}'),
            textposition='outside',
        ))
        fig.update_layout(height=400, yaxis_title='Площадь, га')
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            '**RBR ∩ forest (17 129 га)** — основная оценка ущерба. '
            'Отношение к Hansen (12 636 га) = 1.36×, что согласуется с ожидаемым '
            'диапазоном 1.3–2.0× для независимо валидированной методики.'
        )

    st.markdown('---')

    if damage is not None:
        st.markdown('### Распределение доли повреждений по пожарам')
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            for col, name, color in [
                ('damage_share_forest_with_repeats',    'С повторными',  '#ff5722'),
                ('damage_share_forest_without_repeats', 'Без повторных', '#4caf50'),
            ]:
                if col in damage.columns:
                    fig.add_trace(go.Histogram(
                        x=damage[col].dropna() * 100,
                        name=name, opacity=0.6, nbinsx=20,
                        marker_color=color,
                    ))
            fig.update_layout(
                title='Гистограмма % повреждённого леса',
                xaxis_title='% повреждённого леса',
                yaxis_title='Число пожаров',
                barmode='overlay',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('**Статистика по сценариям**')
            stats_rows = []
            for label, sc_key in scenario_map.items():
                col = f'damage_share_forest_{sc_key}'
                if col in damage.columns:
                    s = damage[col].dropna() * 100
                    if len(s):
                        stats_rows.append({
                            'Сценарий': label,
                            'Медиана %': round(s.median(), 1),
                            'Средн. %':  round(s.mean(), 1),
                            'Std %':     round(s.std(), 1),
                            'n':         len(s),
                        })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows), hide_index=True, use_container_width=True)

elif page.startswith('🌱'):
    st.title('Восстановление растительности')
    st.markdown('Временные ряды индексов растительности для пожаров 2005 г.')

    if recovery is None or vi is None:
        st.error('recovery_per_fire.csv или vi_wide_format.csv не найден')
        st.stop()

    col1, col2 = st.columns([1, 2])
    with col1:
        fire_ids = sorted(recovery['fire_id'].dropna().unique().tolist())
        selected = st.selectbox('Пожар (fire_id):', fire_ids)
    with col2:
        available_indices = [
            i for i in ['NDVI', 'NBR', 'EVI', 'SAVI', 'NBR2', 'BAI', 'NDWI']
            if f'{i}_median' in vi.columns
        ]
        default_ind = [i for i in ['NDVI', 'NBR'] if i in available_indices]
        selected_indices = st.multiselect(
            'Индексы (можно несколько):',
            options=available_indices,
            default=default_ind or available_indices[:2],
        )

    y_min = int(vi.year.min()) if 'year' in vi.columns else 2000
    y_max = int(vi.year.max()) if 'year' in vi.columns else 2024
    year_range = st.slider('Диапазон лет:', y_min, y_max, (y_min, y_max))

    st.markdown('**Слои для отображения** (можно комбинировать):')
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1:
        show_fire = st.checkbox(f'Пожар {selected}', True)
    with lc2:
        show_all = st.checkbox('Все пожары (среднее)', False)
    with lc3:
        show_clean = st.checkbox('Чистые (среднее)', False)
    with lc4:
        show_dirty = st.checkbox('Повторные (среднее)', False)

    available_veg = (
        sorted(vi['veg_name'].dropna().unique().tolist())
        if 'veg_name' in vi.columns else []
    )
    selected_veg = st.multiselect(
        'По типам растительности (можно выбрать несколько):',
        options=available_veg, default=[],
    )

    fire_info = recovery[recovery.fire_id == selected].iloc[0]
    is_clean = bool(fire_info.get('is_clean', False))

    st.markdown(f'### Пожар {selected}')
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Тип растительности', str(fire_info.get('veg_name', 'N/A')))
    c2.metric('Площадь', f'{fire_info.get("fire_area_polygon_ha", 0):,.0f} га')
    c3.metric('Тип', 'Чистый' if is_clean else 'Повторный')
    slope = fire_info.get('NDVI_slope', np.nan)
    c4.metric('NDVI slope', f'{slope:+.4f}' if pd.notna(slope) else 'N/A')

    fire_vi = vi[vi.fire_id == selected].sort_values('year')
    fire_vi_f = fire_vi[
        (fire_vi.year >= year_range[0]) & (fire_vi.year <= year_range[1])
    ]

    def _range_filter(s):
        """Обрезка Series по годам из year_range."""
        return s[(s.index >= year_range[0]) & (s.index <= year_range[1])]

    INDEX_COLORS = {
        'NDVI': '#2e7d32', 'NBR':  '#5d4037', 'EVI':  '#388e3c',
        'SAVI': '#7cb342', 'NBR2': '#8d6e63', 'BAI':  '#e65100',
        'NDWI': '#0288d1',
    }
    VEG_COLORS = {
        'Хвойный лес':    '#1b5e20',
        'Лиственный лес': '#8bc34a',
        'Смешанный лес':  '#4caf50',
        'Редколесье':     '#ff9800',
        'Поля':           '#a1887f',
        'Пастбища':       '#ffc107',
    }

    def make_plot(idx):
        col_name = f'{idx}_median'
        if col_name not in vi.columns:
            return None
        fig = go.Figure()
        idx_color = INDEX_COLORS.get(idx, '#333333')

        if show_fire and len(fire_vi_f):
            fig.add_trace(go.Scatter(
                x=fire_vi_f.year, y=fire_vi_f[col_name],
                name=f'Пожар {selected}', mode='lines+markers',
                line=dict(color=idx_color, width=3),
            ))

        if show_all:
            agg = _range_filter(vi.groupby('year')[col_name].mean())
            if len(agg):
                fig.add_trace(go.Scatter(
                    x=agg.index, y=agg.values, mode='lines',
                    name=f'Все пожары (n={vi.fire_id.nunique()})',
                    line=dict(color='#616161', dash='dash', width=2),
                ))

        if show_clean and 'is_clean' in vi.columns:
            sub = vi[vi.is_clean == True]
            agg = _range_filter(sub.groupby('year')[col_name].mean())
            if len(agg):
                fig.add_trace(go.Scatter(
                    x=agg.index, y=agg.values, mode='lines',
                    name=f'Чистые (n={sub.fire_id.nunique()})',
                    line=dict(color='#2e7d32', dash='dash', width=2),
                ))

        if show_dirty and 'is_clean' in vi.columns:
            sub = vi[vi.is_clean == False]
            agg = _range_filter(sub.groupby('year')[col_name].mean())
            if len(agg):
                fig.add_trace(go.Scatter(
                    x=agg.index, y=agg.values, mode='lines',
                    name=f'Повторные (n={sub.fire_id.nunique()})',
                    line=dict(color='#c62828', dash='dash', width=2),
                ))

        if selected_veg and 'veg_name' in vi.columns:
            for veg in selected_veg:
                sub = vi[vi.veg_name == veg]
                if sub.empty:
                    continue
                agg = _range_filter(sub.groupby('year')[col_name].mean())
                if len(agg):
                    fig.add_trace(go.Scatter(
                        x=agg.index, y=agg.values, mode='lines',
                        name=f'{veg} (n={sub.fire_id.nunique()})',
                        line=dict(color=VEG_COLORS.get(veg, None),
                                  dash='dot', width=2),
                    ))

        if year_range[0] <= 2005 <= year_range[1]:
            fig.add_vline(x=2005.7, line_dash='dash', line_color='red',
                          annotation_text='Пожар', annotation_position='top')

        fig.update_layout(
            title=f'{idx} за {year_range[0]}–{year_range[1]}',
            xaxis_title='Год',
            yaxis_title=f'{idx} (median)',
            height=400,
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1),
        )
        return fig

    if not selected_indices:
        st.warning('Выберите хотя бы один индекс')
    else:
        for i in range(0, len(selected_indices), 2):
            chunk = selected_indices[i:i + 2]
            cols = st.columns(len(chunk))
            for c, idx in zip(cols, chunk):
                fig = make_plot(idx)
                if fig is not None:
                    c.plotly_chart(fig, use_container_width=True)

    st.markdown('### Метрики восстановления (NDVI и NBR)')
    metric_map = [
        ('NDVI baseline (2002–2005)',    'NDVI_baseline'),
        ('NDVI min (обычно 2006)',       'NDVI_post_min'),
        ('NDVI slope (год/год)',         'NDVI_slope'),
        ('NDVI recovery share',          'NDVI_recovery_share'),
        ('NDVI years to 90%',            'NDVI_time_to_90pct'),
        ('NDVI R² линейной модели',      'NDVI_r_squared'),
        ('NBR baseline',                 'NBR_baseline'),
        ('NBR min',                      'NBR_post_min'),
        ('NBR slope',                    'NBR_slope'),
        ('NBR recovery share',           'NBR_recovery_share'),
        ('NBR years to 90%',             'NBR_time_to_90pct'),
        ('NBR R²',                       'NBR_r_squared'),
    ]
    rows = []
    for label, key in metric_map:
        v = fire_info.get(key)
        rows.append({'Метрика': label, 'Значение': f'{v:.4f}' if pd.notna(v) else 'N/A'})
    ncols = 3
    per_col = len(rows) // ncols + (1 if len(rows) % ncols else 0)
    cols = st.columns(ncols)
    for i, c in enumerate(cols):
        chunk = rows[i * per_col:(i + 1) * per_col]
        if chunk:
            c.dataframe(pd.DataFrame(chunk), hide_index=True, use_container_width=True)

elif page.startswith('🌲'):
    st.title('Смена типов растительности (MODIS Land Cover)')
    st.markdown('Динамика площадей типов растительности на территориях гарей 2005 г. за 2001–2024 гг.')

    if veg_year is None:
        st.error('vegetation_area_by_year.xlsx не найден')
        st.stop()

    v = veg_year.rename(columns={
        'Лесистые саванны': 'Редколесье',
        'Саванны':          'Разреженный древостой',
    })
    year_col_name = next(
        (c for c in v.columns if c.lower() in ('year', 'год')),
        v.columns[0],
    )
    v = v.set_index(year_col_name).sort_index()
    v = v.drop(columns=[c for c in v.columns if c.lower() in ('year', 'год')], errors='ignore')
    type_cols = [
        c for c in v.columns
        if pd.api.types.is_numeric_dtype(v[c]) and v[c].sum() > 0
    ]

    default_types = [
        'Вечнозелёные хвойные леса',
        'Листопадные широколиственные леса',
        'Смешанные леса',
        'Редколесье',
        'Луга и пастбища',
    ]
    default_types = [t for t in default_types if t in type_cols]

    selected_types = st.multiselect(
        'Типы растительности:',
        options=type_cols,
        default=default_types,
    )

    if not selected_types:
        st.warning('Выберите хотя бы один тип')
    else:

        fig = go.Figure()
        for t in selected_types:
            fig.add_trace(go.Scatter(
                x=v.index, y=v[t] / 1000,
                name=t, mode='lines+markers',
            ))
        fig.add_vline(x=2005.7, line_dash='dash', line_color='red',
                      annotation_text='Пожар')
        fig.update_layout(
            title='Динамика площадей',
            xaxis_title='Год',
            yaxis_title='Площадь (тыс. га)',
            height=500,
            hovermode='x unified',
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Изменения 2001 → 2024')
        pre = v.iloc[0][selected_types]
        last = v.iloc[-1][selected_types]
        delta_ha = last - pre
        delta_pct = (delta_ha / pre.replace(0, np.nan)) * 100

        change_df = pd.DataFrame({
            '2001 (га)': pre.round(0),
            '2024 (га)': last.round(0),
            'Δ (га)':    delta_ha.round(0),
            'Δ (%)':     delta_pct.round(1),
        }).sort_values('Δ (%)', ascending=False)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(change_df, use_container_width=True)
        with col2:
            bar_df = change_df.reset_index().rename(columns={'index': 'Тип'})
            fig = px.bar(
                bar_df.sort_values('Δ (%)'),
                x='Δ (%)', y='Тип', orientation='h',
                color='Δ (%)', color_continuous_scale='RdYlGn',
                color_continuous_midpoint=0,
                title='Изменение площади, %',
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown('### Процентная структура по годам')
        v_sel = v[selected_types].copy()
        v_pct = v_sel.div(v_sel.sum(axis=1), axis=0) * 100
        fig = go.Figure()
        for t in selected_types:
            fig.add_trace(go.Scatter(
                x=v_pct.index, y=v_pct[t],
                name=t, mode='lines',
                stackgroup='one',
            ))
        fig.add_vline(x=2005.7, line_dash='dash', line_color='red')
        fig.update_layout(
            title='Структура типов растительности (%)',
            xaxis_title='Год',
            yaxis_title='Доля от суммарной площади (%)',
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

elif page.startswith('📈'):
    st.title('Скорость восстановления и площадь гари')
    st.markdown(
        'Взаимосвязь скорости восстановления растительности с площадью, '
        'пройденной огнём, и лесной площадью пожара (корреляция Spearman).'
    )

    if recovery is None:
        st.error('recovery_per_fire.csv не найден')
        st.stop()

    RECOVERY_METRICS = {
        'NDVI_slope': 'Скорость восстановления NDVI (slope)',
        'NBR_slope':  'Скорость восстановления NBR (slope)',
    }
    AREA_METRICS = {
        'fire_area_polygon_ha': 'Площадь, пройденная огнём (га)',
        'forest_pre_total_ha':  'Лесная площадь пожара (га, Hansen)',
    }

    available_metrics = {k: v for k, v in RECOVERY_METRICS.items() if k in recovery.columns}
    available_areas = {k: v for k, v in AREA_METRICS.items() if k in recovery.columns}

    if not available_metrics or not available_areas:
        st.error('В recovery_per_fire.csv нет нужных колонок (slope / площадь).')
        st.stop()

    st.markdown('### Корреляция скорости восстановления с площадью гари')

    from scipy import stats as _stats

    rows = []
    for m_key, m_label in available_metrics.items():
        for a_key, a_label in available_areas.items():
            pair = recovery[[m_key, a_key]].dropna()
            if len(pair) < 3:
                continue
            r, p = _stats.spearmanr(pair[m_key], pair[a_key])
            sig = '**' if p < 0.01 else ('*' if p < 0.05 else '')
            rows.append({
                'Скорость восстановления': m_label,
                'Показатель площади': a_label,
                'r (Spearman)': round(r, 3),
                'p-value': round(p, 4),
                'n': len(pair),
                'Значимость': sig,
            })
    corr_table = pd.DataFrame(rows)
    st.dataframe(corr_table, hide_index=True, use_container_width=True)
    st.caption('* p<0.05, ** p<0.01. Отрицательный r: чем больше площадь, тем медленнее восстановление.')

    st.markdown('---')

    st.markdown('### Диаграмма рассеяния')
    col1, col2, col3 = st.columns(3)
    with col1:
        m_key = st.selectbox(
            'Скорость восстановления:',
            options=list(available_metrics.keys()),
            format_func=lambda k: available_metrics[k],
        )
    with col2:
        a_key = st.selectbox(
            'Показатель площади:',
            options=list(available_areas.keys()),
            format_func=lambda k: available_areas[k],
        )
    with col3:
        color_options = ['Нет']
        if 'veg_name' in recovery.columns:
            color_options.append('veg_name')
        if 'is_clean' in recovery.columns:
            color_options.append('is_clean')
        color_by = st.selectbox('Раскраска:', color_options)

    plot_kw = dict(
        x=a_key, y=m_key,
        labels={a_key: available_areas[a_key], m_key: available_metrics[m_key]},
        title=f'{available_metrics[m_key]} vs {available_areas[a_key]}',
        height=500,
        trendline='ols',
    )
    if color_by != 'Нет':
        plot_kw['color'] = color_by
    try:
        fig = px.scatter(recovery, **plot_kw)
    except Exception:
        plot_kw.pop('trendline', None)
        fig = px.scatter(recovery, **plot_kw)
    st.plotly_chart(fig, use_container_width=True)
