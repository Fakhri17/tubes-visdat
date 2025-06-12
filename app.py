import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
import warnings
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import ipywidgets as widgets
from IPython.display import display, clear_output
import streamlit as st
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

st.set_page_config(
    page_title="Analisis Penjualan Video Game",
    page_icon="🎮",
    layout="wide"
)

st.title("Visualisasi dan Analisis Penjualan Video Game")
st.markdown("""
## Informasi Kelompok
1. Fakhri Alauddin ( 1203220131 )
2. M Riyan Akbari ( 1203220130 )
3. Elan Agum Wicaksono ( 1203220005 )
4. Ferry Oktariansyah ( 1203220006 )
""")

@st.cache_data
def load_data():
    df = pd.read_csv("vgsales.csv")
    return df

if 'preprocessed_df' not in st.session_state:
    st.session_state.preprocessed_df = None

st.sidebar.title("Menu Navigasi")

with st.sidebar.expander("About Dataset", expanded=False):
    st.markdown("""
    Dataset berisi data penjualan video game dengan:
    - 16,598 records
    - 11 kolom
    
    **Kolom Utama:**
    - Rank (int): Peringkat penjualan game
    - Name (str): Nama video game
    - Platform (str): Platform gaming (Wii, NES, GB, dll)
    - Year (float): Tahun rilis (beberapa nilai kosong)
    - Genre (str): Genre game (Sports, Racing, Role-Playing, dll)
    - Publisher (str): Publisher game (beberapa nilai kosong)
    - NA_Sales (float): Penjualan di Amerika Utara (juta unit)
    - EU_Sales (float): Penjualan di Eropa (juta unit)
    - JP_Sales (float): Penjualan di Jepang (juta unit)
    - Other_Sales (float): Penjualan di region lain (juta unit)
    - Global_Sales (float): Total penjualan global (juta unit)
    """)

page = st.sidebar.radio(
    "Pilih Halaman",
    ["Preprocessing", "Distribusi", "Tren", "Perbandingan", "Korelasi", "Geografis", "Hasil Model"]
)

if page == "Preprocessing":
    st.header("Preprocessing Data")
    
    df = load_data()
    
    st.subheader("Informasi Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Jumlah Data Kosong (Sebelum Pembersihan)")
    st.write(df.isna().sum())

    df_cleaned = df.dropna()

    st.subheader("Jumlah Data Kosong (Setelah Pembersihan)")
    st.write(df_cleaned.isna().sum())

    st.subheader("Jumlah Data Duplikat")
    st.write(f"Total duplikat: {df_cleaned.duplicated().sum()}")

    st.session_state.preprocessed_df = df_cleaned
    st.success("Data berhasil diproses dan siap digunakan!")

else:
    if st.session_state.preprocessed_df is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu di halaman Preprocessing!")
    else:
        df = st.session_state.preprocessed_df

if page == "Distribusi":
    st.header("Visualisasi Distribusi")

    st.subheader("Distribusi Tahun Rilis Game")
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider("Tahun Awal", 
                           int(df['Year'].min()), 
                           int(df['Year'].max())-1, 
                           int(df['Year'].min()))
    with col2:
        end_year = st.slider("Tahun Akhir", 
                         start_year+1, 
                         int(df['Year'].max()), 
                         int(df['Year'].max()))

    col3, col4 = st.columns(2)
    with col3:
        compare_year1 = st.number_input("Tahun Pembanding 1", 
                                    min_value=int(df['Year'].min()),
                                    max_value=int(df['Year'].max()),
                                    value=2008)
    with col4:
        compare_year2 = st.number_input("Tahun Pembanding 2", 
                                    min_value=int(df['Year'].min()),
                                    max_value=int(df['Year'].max()),
                                    value=2009)

    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(data=filtered_df, x="Year", bins=40, kde=False, ax=ax)
    
    plt.title(f'Distribusi Tahun Rilis Game ({start_year}-{end_year})', fontsize=12, fontweight='bold')
    plt.xlabel('Tahun Rilis', fontsize=10)
    plt.ylabel('Jumlah Game', fontsize=10)

    if not filtered_df.empty:
        peak_year = filtered_df['Year'].value_counts().idxmax()
        peak_year_count = filtered_df['Year'].value_counts().max()
        plt.text(start_year + (end_year-start_year)*0.6, plt.gca().get_ylim()[1]*0.9,
                f'Tahun dengan jumlah game terbanyak: {int(peak_year)} ({peak_year_count} game)',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))

    year1_count = df[df['Year'] == compare_year1].shape[0]
    year2_count = df[df['Year'] == compare_year2].shape[0]
    if year1_count > 0:
        perc_diff = ((year2_count - year1_count) / year1_count * 100)
        plt.text(start_year + (end_year-start_year)*0.1, plt.gca().get_ylim()[1]*0.7,
                f'Perbandingan {compare_year2} vs {compare_year1}:\n{compare_year2}: {year2_count} game\n{compare_year1}: {year1_count} game\nPerbedaan: {perc_diff:.1f}%',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.3))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Distribusi Genre Game")

    col1, col2, col3 = st.columns(3)
    with col1:
        max_genres = st.slider("Maksimum Genre", 5, len(df['Genre'].unique()), len(df['Genre'].unique()))
    with col2:
        min_percent = st.slider("Persentase Minimum (%)", 0.0, 5.0, 0.0, 0.1)
    with col3:
        highlight_genre = st.selectbox("Highlight Genre", ['None'] + sorted(df['Genre'].unique().tolist()))

    fig, ax = plt.subplots(figsize=(12, 8))

    genre_counts = df['Genre'].value_counts()
    genre_percent = (genre_counts / genre_counts.sum() * 100).round(1)

    if max_genres < len(genre_counts):
        main_genres = genre_counts.iloc[:max_genres-1]
        other_count = genre_counts.iloc[max_genres-1:].sum()
        genre_counts = pd.Series(list(main_genres) + [other_count],
                     index=list(main_genres.index) + ['Lainnya'])
        genre_percent = (genre_counts / genre_counts.sum() * 100).round(1)

    if min_percent > 0:
        small_genres_mask = genre_percent < min_percent
        if small_genres_mask.any():
            main_genres = genre_counts[~small_genres_mask]
            other_count = genre_counts[small_genres_mask].sum()
            genre_counts = pd.Series(list(main_genres) + [other_count],
                       index=list(main_genres.index) + ['Lainnya'])
            genre_percent = (genre_counts / genre_counts.sum() * 100).round(1)

    labels = [f'{genre}: {count} ({percent}%)' for genre, count, percent in 
            zip(genre_counts.index, genre_counts, genre_percent)]

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(genre_counts)))
    explode = [0.1 if genre == highlight_genre else 0 for genre in genre_counts.index]

    plt.pie(genre_counts, explode=explode, labels=None, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        shadow=True, startangle=90)

    plt.title('Distribusi Game Berdasarkan Genre', pad=20)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Distribusi Platform Game")

    col1, col2 = st.columns(2)
    with col1:
        max_platforms = st.slider("Jumlah Platform", 5, 15, 10)
    with col2:
        min_percent = st.slider("Persentase Minimum", 0.0, 5.0, 0.0, 0.1)

    top_platforms = df['Platform'].value_counts().head(10).index.tolist()
    top_platforms.sort()
    highlight_platform = st.selectbox("Highlight Platform", ['None'] + top_platforms + ['Others'])

    fig, ax = plt.subplots(figsize=(12, 8))

    platform_counts = df['Platform'].value_counts()
    platform_percent = (platform_counts / platform_counts.sum() * 100).round(1)

    if max_platforms < len(platform_counts):
        main_platforms = platform_counts.iloc[:max_platforms-1]
        other_count = platform_counts.iloc[max_platforms-1:].sum()
        platform_counts = pd.Series(list(main_platforms) + [other_count],
                                index=list(main_platforms.index) + ['Lainnya'])
        platform_percent = (platform_counts / platform_counts.sum() * 100).round(1)

    if min_percent > 0:
        small_platforms_mask = platform_percent < min_percent
        if small_platforms_mask.any():
            main_platforms = platform_counts[~small_platforms_mask]
            other_count = platform_counts[small_platforms_mask].sum()
            platform_counts = pd.Series(list(main_platforms) + [other_count],
                                    index=list(main_platforms.index) + ['Lainnya'])
            platform_percent = (platform_counts / platform_counts.sum() * 100).round(1)

    labels = [f'{platform}: {count} ({percent}%)' for platform, count, percent in
            zip(platform_counts.index, platform_counts, platform_percent)]

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(platform_counts)))
    explode = [0.1 if platform == highlight_platform else 0 for platform in platform_counts.index]

    plt.pie(platform_counts, explode=explode, labels=None, colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
            shadow=True, startangle=90)

    plt.title('Distribusi Game Berdasarkan Platform', pad=20)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    st.pyplot(fig)

elif page == "Tren":
    st.header("Visualisasi Tren")

    df_grouped = df.groupby('Year')['Global_Sales'].sum().reset_index()
    min_year = int(df_grouped['Year'].min())
    max_year = int(df_grouped['Year'].max())
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider('Tahun Awal', min_year, max_year-1, min_year)
    with col2:
        end_year = st.slider('Tahun Akhir', min_year+1, max_year, max_year)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    filtered_df = df_grouped[(df_grouped['Year'] >= start_year) & (df_grouped['Year'] <= end_year)]
    
    plt.plot(filtered_df['Year'], filtered_df['Global_Sales'],
             marker='o', color='#2c7bb6', linestyle='-',
             linewidth=3, markersize=8, markerfacecolor='#fdae61')

    plt.title(f'Tren Penjualan Global Video Game per Tahun ({start_year}-{end_year})\n',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Tahun', fontsize=12, labelpad=10)
    plt.ylabel('Total Penjualan Global (dalam Jutaan Unit)', fontsize=12, labelpad=10)
    plt.xticks(filtered_df['Year'], rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)

    if not filtered_df.empty:
        max_sales = filtered_df['Global_Sales'].max()
        max_year = filtered_df.loc[filtered_df['Global_Sales'].idxmax(), 'Year']
        plt.annotate(f'Puncak: {max_sales:.2f}Jt unit',
                     xy=(max_year, max_sales),
                     xytext=(max_year + 1, max_sales - 50),
                     arrowprops=dict(arrowstyle='->'),
                     fontsize=10,
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.3))

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Tren Penjualan per Region per Tahun")
    
    df_grouped_region = df.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
    min_year = int(df_grouped_region['Year'].min())
    max_year = int(df_grouped_region['Year'].max())
    
    col1, col2 = st.columns(2)
    with col1:
        start_year_region = st.slider('Tahun Awal', min_year, max_year-1, min_year, key='region_start')
    with col2:
        end_year_region = st.slider('Tahun Akhir', min_year+1, max_year, max_year, key='region_end')

    regions = {
        'NA_Sales': 'Amerika Utara',
        'EU_Sales': 'Eropa', 
        'JP_Sales': 'Jepang',
        'Other_Sales': 'Region Lainnya'
    }
    
    selected_regions = st.multiselect(
        'Pilih Region:',
        options=list(regions.keys()),
        default=list(regions.keys()),
        format_func=lambda x: regions[x]
    )

    if selected_regions:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        df_filtered = df_grouped_region[(df_grouped_region['Year'] >= start_year_region) & 
                                      (df_grouped_region['Year'] <= end_year_region)]

        colors = {
            'NA_Sales': '#1f77b4',
            'EU_Sales': '#ff7f0e',
            'JP_Sales': '#2ca02c',
            'Other_Sales': '#9467bd'
        }
        
        markers = {
            'NA_Sales': 'o',
            'EU_Sales': 's',
            'JP_Sales': '^',
            'Other_Sales': 'd'
        }

        for region in selected_regions:
            plt.plot(df_filtered['Year'], df_filtered[region],
                     marker=markers[region], label=regions[region],
                     color=colors[region], linestyle='-', linewidth=2.5,
                     markersize=7, markeredgecolor='white', markeredgewidth=0.7)

            if not df_filtered.empty:
                max_val = df_filtered[region].max()
                max_year = df_filtered.loc[df_filtered[region].idxmax(), 'Year']
                plt.annotate(f'{max_val:.1f}Jt', xy=(max_year, max_val),
                             xytext=(5, 5), textcoords='offset points',
                             fontsize=9, color=colors[region],
                             bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))

        plt.title(f'Perbandingan Tren Penjualan Video Game per Region ({start_year_region}-{end_year_region})\n',
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Tahun', fontsize=12, labelpad=10)
        plt.ylabel('Total Penjualan (dalam Jutaan Unit)', fontsize=12, labelpad=10)
        plt.xticks(range(start_year_region, end_year_region+1, 5))
        plt.grid(True, linestyle='--', alpha=0.6)

        legend = plt.legend(title='Region:', fontsize=11, title_fontsize=12,
                            bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Tren Genre Populer")

    df_grouped_genre = df.groupby('Year')['Genre'].value_counts().reset_index()
    min_year = int(df_grouped_genre['Year'].min())
    max_year = int(df_grouped_genre['Year'].max())
    
    year_range = st.slider(
        "Pilih Rentang Tahun",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    all_genres = sorted(df['Genre'].unique())
    selected_genres = st.multiselect(
        "Pilih Genre",
        options=all_genres,
        default=all_genres[:5]
    )
    
    if selected_genres:
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        df_grouped_genre = df_filtered.groupby(['Year', 'Genre']).size().unstack(fill_value=0)
        
        sns.heatmap(df_grouped_genre,
                   annot=True,
                   cmap='YlGnBu',
                   fmt='d',
                   linewidths=0.5,
                   annot_kws={'size': 8},
                   cbar_kws={'label': 'Jumlah Video Game'},
                   ax=ax)

        plt.title('Tren Genre Populer per Tahun', fontsize=16, pad=20)
        plt.xlabel('Genre', fontsize=12)
        plt.ylabel('Tahun', fontsize=12)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        year_labels = [int(year) if i % 2 == 0 else '' for i, year in enumerate(df_grouped_genre.index)]
        plt.yticks(np.arange(len(df_grouped_genre.index)),
                  labels=year_labels,
                  rotation=0,
                  fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")

elif page == "Perbandingan":
    st.header("Visualisasi Perbandingan")

    st.subheader("Top 10 Publisher dengan Penjualan Global Tertinggi")

    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_1")

    if not show_all_years:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1
        )
        
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"
    
    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_1")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5]
        )
    
    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
        
        top_publishers = df_filtered.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=top_publishers, x='Global_Sales', y='Publisher', hue='Publisher', palette='viridis', ax=ax)
        
        plt.title(f'Top 10 Publisher dengan Penjualan Global Tertinggi {title_year_range}', fontsize=14)
        plt.xlabel('Total Global Sales (juta unit)')
        plt.ylabel('Publisher')
        
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.1f}", (p.get_width()+0.3, p.get_y() + 0.5), va='center')
            
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Top 20 Game Terlaris Secara Global")

    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_2")

    if not show_all_years:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_2"
        )
        
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    top_games = df_filtered[['Name', 'Global_Sales']].sort_values(by='Global_Sales', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=top_games, x='Global_Sales', y='Name', hue='Name', palette='mako', ax=ax)
    
    plt.title(f'Top 20 Game Terlaris Secara Global {title_year_range}', fontsize=14)
    plt.xlabel('Total Global Sales (juta unit)')
    plt.ylabel('Nama Game')
    
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}", (p.get_width()+0.2, p.get_y() + 0.5), va='center')
        
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Total Penjualan Global per Platform")

    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_platform")

    if not show_all_years:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_platform"
        )
        
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_platform")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            key="genre_select_platform"
        )

    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
    else:
        title_genres = ""

    platform_sales = df_filtered.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        data=platform_sales,
        x='Platform',
        y='Global_Sales',
        hue='Platform',
        dodge=False,
        palette='crest',
        ax=ax
    )

    plt.title(f'Total Penjualan Global per Platform {title_year_range}', fontsize=14)
    plt.xlabel('Platform')
    plt.ylabel('Total Global Sales (juta unit)')
    plt.xticks(rotation=45)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.2), ha='center')

    plt.tight_layout()
    st.pyplot(fig)

elif page == "Korelasi":
    st.header("Visualisasi Korelasi")

    st.subheader("Korelasi antara Tahun Rilis dan Penjualan Global")

    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_correlation")

    if not show_all_years:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1
        )
        
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5]
        )

    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.scatterplot(
            x="Year",
            y="Global_Sales",
            data=df_filtered,
            hue="Genre",
            size="Global_Sales",
            sizes=(20, 200),
            alpha=0.7,
            palette="viridis",
            legend=False,
            ax=ax
        )

        sns.regplot(
            x="Year",
            y="Global_Sales",
            data=df_filtered,
            scatter=False,
            color="red",
            line_kws={"linestyle": "--", "linewidth": 2},
            ax=ax
        )

        top_games = df_filtered.nlargest(5, 'Global_Sales')
        for idx, row in top_games.iterrows():
            ax.annotate(
                row['Name'],
                xy=(row['Year'], row['Global_Sales']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
            )

        plt.title(f"Korelasi antara Tahun Rilis dan Penjualan Global {title_year_range}", fontsize=16)
        plt.xlabel("Tahun Rilis", fontsize=12)
        plt.ylabel("Penjualan Global (Juta Unit)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")

    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Korelasi antara Penjualan Global dan Penjualan di Region")

    fig, ax = plt.subplots(figsize=(14, 10))

    scatter = sns.scatterplot(
        x="NA_Sales",
        y="EU_Sales",
        data=df,
        hue="Genre",
        size="Global_Sales",
        sizes=(20, 200),
        alpha=0.7,
        palette="viridis",
        ax=ax
    )

    sns.regplot(
        x="NA_Sales",
        y="EU_Sales",
        data=df,
        scatter=False,
        color="red",
        line_kws={"linestyle": "--", "linewidth": 2},
        ax=ax
    )

    max_sales = max(df["NA_Sales"].max(), df["EU_Sales"].max())
    ax.plot([0, max_sales], [0, max_sales], 'k--', alpha=0.3, label="Penjualan Sama (y=x)")

    plt.title("Hubungan Penjualan Game antara Amerika Utara dan Eropa", fontsize=16)
    plt.xlabel("Penjualan Amerika Utara (Juta Unit)", fontsize=12)
    plt.ylabel("Penjualan Eropa (Juta Unit)", fontsize=12)

    top_games = df.nlargest(5, 'Global_Sales')
    for idx, row in top_games.iterrows():
        ax.annotate(
            row['Name'],
            xy=(row['NA_Sales'], row['EU_Sales']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7)
        )

    plt.legend(title="Genre Game", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.fill_between([0, max_sales], [0, 0], [0, max_sales], color='blue', alpha=0.05, label="NA < EU")
    plt.fill_between([0, max_sales], [0, max_sales], [max_sales, max_sales], color='red', alpha=0.05, label="NA > EU")

    plt.xlim(-0.5, max(df["NA_Sales"].max() * 1.1, 1))
    plt.ylim(-0.5, max(df["EU_Sales"].max() * 1.1, 1))

    plt.tight_layout()
    st.pyplot(fig)
    

elif page == "Geografis":
    st.header("Visualisasi Geografis")

    st.subheader("Total Penjualan Global per Negara")

    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_geo")

    if not show_all_years:
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_geo"
        )
        
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_geo")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5],
            key="genre_select_geo"
        )

    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        regional_sales = df_filtered[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        ax1.pie(regional_sales, labels=regional_sales.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title(f'Distribusi Penjualan Regional {title_year_range}', fontsize=14)
        
        yearly_sales = df_filtered.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
        
        for region, color in zip(yearly_sales.columns, colors):
            ax2.plot(yearly_sales.index, yearly_sales[region], label=region, color=color, marker='o')
        
        ax2.set_title(f'Tren Penjualan Regional per Tahun {title_year_range}', fontsize=14)
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Total Penjualan (Juta Unit)')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.subheader("Analisis Regional")
        
        regional_stats = pd.DataFrame({
            'Total Penjualan': regional_sales,
            'Persentase': (regional_sales / regional_sales.sum() * 100).round(1),
            'Rata-rata per Game': (regional_sales / len(df_filtered)).round(2)
        })
        
        st.dataframe(regional_stats, use_container_width=True)
        
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")

elif page == "Hasil Model":
    st.header("Hasil Model Prediksi Penjualan Game")
    
    def prepare_model_data(df):
        model_df = df.copy()
        
        le_platform = LabelEncoder()
        le_genre = LabelEncoder()
        le_publisher = LabelEncoder()
        
        model_df['Platform_Encoded'] = le_platform.fit_transform(model_df['Platform'])
        model_df['Genre_Encoded'] = le_genre.fit_transform(model_df['Genre'])
        model_df['Publisher_Encoded'] = le_publisher.fit_transform(model_df['Publisher'])
        
        features = ['Platform_Encoded', 'Year', 'Genre_Encoded', 'Publisher_Encoded', 
                    'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        X = model_df[features]
        y = model_df['Global_Sales']
        
        return X, y, le_platform, le_genre, le_publisher

    @st.cache_resource
    def train_model():
        X, y, le_platform, le_genre, le_publisher = prepare_model_data(df)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        y_pred = rf_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        feature_importance = pd.DataFrame({
            'Feature': ['Platform', 'Year', 'Genre', 'Publisher', 
                       'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return rf_model, rmse, r2, feature_importance, le_platform, le_genre, le_publisher

    model, rmse, r2, feature_importance, le_platform, le_genre, le_publisher = train_model()

    
    st.subheader("Analisis Model")
    
    st.markdown("### Analisis Performa Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("RMSE", f"{rmse:.4f} juta unit")
        st.markdown("""
        - Rata-rata kesalahan prediksi
        - Nilai relatif kecil dibanding range penjualan
        - Model dapat memprediksi dengan akurasi baik
        """)
    
    with col2:
        st.metric("R² Score", f"{r2:.4f}")
        st.markdown(f"""
        - Model dapat menjelaskan {r2*100:.1f}% variasi data
        - Menunjukkan kemampuan prediksi yang baik
        - Dapat menangkap pola dan hubungan signifikan
        """)

    st.markdown("### Analisis Feature Importance")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax)
        plt.title('Feature Importance dalam Model')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        top_features = feature_importance.head(3)
        for idx, row in top_features.iterrows():
            st.markdown(f"""
            **{row['Feature']}** ({row['Importance']:.2%})
            - Faktor kunci dalam penjualan game
            - Memiliki pengaruh signifikan
            """)

    st.markdown("### Insights Bisnis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Regional Sales Impact**
        - Penjualan regional memiliki pengaruh signifikan
        - Perlu strategi pemasaran berbeda per region
        
        **Platform Strategy**
        - Platform merupakan faktor penting
        - Pilih platform yang tepat untuk setiap game
        """)
    
    with col2:
        st.markdown("""
        **Genre Consideration**
        - Genre mempengaruhi potensi penjualan
        - Pahami preferensi pasar per genre
        
        **Publisher Influence**
        - Publisher berperan dalam kesuksesan
        - Reputasi dan pengalaman berpengaruh
        """)

    st.markdown("### Keterbatasan Model")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Data Historis**
        - Berdasarkan data historis
        - Perubahan industri dapat mempengaruhi akurasi
        
        **Faktor Eksternal**
        - Tidak mempertimbangkan:
          - Kondisi ekonomi
          - Perubahan teknologi
          - Perilaku konsumen
        """)
    
    with col2:
        st.markdown("""
        **Kualitas Data**
        - Beberapa data tidak lengkap
        - Nilai yang hilang dapat mempengaruhi akurasi
        
        **Perlu Diperhatikan**
        - Model perlu update berkala
        - Validasi dengan data terbaru
        """)

    st.markdown("---")

    st.subheader("Prediksi Penjualan Game")
    st.markdown("Masukkan detail game untuk memprediksi penjualan globalnya:")

    col1, col2 = st.columns(2)
    with col1:
        platform = st.selectbox("Platform", sorted(df['Platform'].unique()))
        year = st.number_input("Tahun", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=2020)
        genre = st.selectbox("Genre", sorted(df['Genre'].unique()))
        publisher = st.selectbox("Publisher", sorted(df['Publisher'].unique()))
    
    with col2:
        na_sales = st.number_input("NA Sales (juta unit)", min_value=0.0, max_value=float(df['NA_Sales'].max()), value=0.0)
        eu_sales = st.number_input("EU Sales (juta unit)", min_value=0.0, max_value=float(df['EU_Sales'].max()), value=0.0)
        jp_sales = st.number_input("JP Sales (juta unit)", min_value=0.0, max_value=float(df['JP_Sales'].max()), value=0.0)
        other_sales = st.number_input("Other Sales (juta unit)", min_value=0.0, max_value=float(df['Other_Sales'].max()), value=0.0)

    if st.button("Prediksi Penjualan"):
        input_data = pd.DataFrame({
            'Platform_Encoded': [le_platform.transform([platform])[0]],
            'Year': [year],
            'Genre_Encoded': [le_genre.transform([genre])[0]],
            'Publisher_Encoded': [le_publisher.transform([publisher])[0]],
            'NA_Sales': [na_sales],
            'EU_Sales': [eu_sales],
            'JP_Sales': [jp_sales],
            'Other_Sales': [other_sales]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
            <h2 style='color: #1f77b4;'>Prediksi Penjualan Global</h2>
            <h1 style='color: #2ecc71; font-size: 36px;'>{prediction:.2f} juta unit</h1>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Analisis Faktor yang Mempengaruhi Prediksi")
        
        input_importance = pd.DataFrame({
            'Feature': ['Platform', 'Year', 'Genre', 'Publisher', 
                       'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'],
            'Value': [platform, year, genre, publisher, 
                     na_sales, eu_sales, jp_sales, other_sales],
            'Importance': model.feature_importances_
        })
        
        input_importance['Weighted_Impact'] = 0.0
        for idx, row in input_importance.iterrows():
            if row['Feature'] in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Year']:
                input_importance.at[idx, 'Weighted_Impact'] = row['Importance'] * float(row['Value'])
            else:
                input_importance.at[idx, 'Weighted_Impact'] = row['Importance']
        
        input_importance = input_importance.sort_values('Weighted_Impact', ascending=False)
        
        st.markdown("#### Faktor-faktor yang Mempengaruhi Prediksi:")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            top_factors = input_importance.head(3)
            for idx, row in top_factors.iterrows():
                if row['Feature'] in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']:
                    st.markdown(f"""
                    **{row['Feature']}** ({row['Value']:.2f} juta unit)
                    - Pengaruh: {row['Weighted_Impact']:.4f}
                    - Penjualan regional yang tinggi memberikan kontribusi signifikan
                    """)
                elif row['Feature'] == 'Year':
                    st.markdown(f"""
                    **{row['Feature']}** ({int(row['Value'])})
                    - Pengaruh: {row['Weighted_Impact']:.4f}
                    - Tahun rilis mempengaruhi potensi penjualan
                    """)
                else:
                    st.markdown(f"""
                    **{row['Feature']}** ({row['Value']})
                    - Pengaruh: {row['Weighted_Impact']:.4f}
                    - Faktor penting dalam menentukan potensi penjualan
                    """)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 8))
            sns.barplot(data=input_importance, y='Feature', x='Weighted_Impact', ax=ax)
            plt.title('Pengaruh Faktor terhadap Prediksi')
            plt.xlabel('Pengaruh')
            plt.tight_layout()
            st.pyplot(fig)

        st.markdown("#### Analisis Keseluruhan:")
        total_regional = na_sales + eu_sales + jp_sales + other_sales
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Prediksi Penjualan Global", f"{prediction:.2f} juta unit")
            st.metric("Total Penjualan Regional", f"{total_regional:.2f} juta unit")
        
        with col2:
            if prediction > total_regional:
                st.markdown("""
                **Potensi Pertumbuhan Tinggi**
                - Prediksi lebih tinggi dari penjualan regional
                - Menunjukkan potensi pertumbuhan yang baik di pasar global
                """)
            else:
                st.markdown("""
                **Perlu Strategi Pasar**
                - Prediksi lebih rendah dari penjualan regional
                - Perlu strategi untuk meningkatkan penetrasi pasar global
                """)

        st.markdown("#### Potensi Pasar:")
        genre_avg = df[df['Genre'] == genre]['Global_Sales'].mean()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rata-rata Penjualan Genre", f"{genre_avg:.2f} juta unit")
        
        with col2:
            if prediction > genre_avg:
                st.markdown("""
                **Di Atas Rata-rata Genre**
                - Prediksi menunjukkan potensi yang baik
                - Kombinasi platform dan publisher yang tepat
                """)
            else:
                st.markdown("""
                **Perlu Diferensiasi**
                - Perlu fokus pada diferensiasi
                - Kembangkan keunggulan unik game
                """)

    