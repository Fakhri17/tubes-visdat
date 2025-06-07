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

st.set_page_config(
    page_title="Game Analytics Dashboard",
    page_icon="🎮",
    layout="wide"
)

st.title("Visualisasi Penjualan Video Game")
st.markdown("""
## Informasi Kelompok
1. Fakhri Alauddin ( 1203220131 )
2. M Riyan Akbari ( 1203220130 )
3. Elan Agum Wicaksono ( 1203220005 )
4. Ferry Oktariansyah ( 1203220006 )



### About Dataset
This dataset contains video game sales data across different platforms, genres, and regions, making it valuable for various analytical and business use cases.

- Total Records = 16,598
- Total Collumn = 11

- Rank (int): Rank of the game in terms of sales.
- Name (str): Name of the video game.
- Platform (str): Gaming platform (e.g., Wii, NES, GB).
- Year (float): Release year (some missing values).
- Genre (str): Game genre (e.g., Sports, Racing, Role-Playing).
- Publisher (str): Publisher of the game (some missing values).
- NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales (float): Sales in different regions (millions of copies). 
""")

st.markdown("""
Kemungkinan Visualisasi Data

1. Visualisasi Distribusi
- Histogram Tahun Rilis: Distribusi tahun rilis game untuk melihat tren produksi game
- Pie Chart Genre: Persentase game berdasarkan genre
- Pie Chart Platform: Distribusi game berdasarkan platform

2. Visualisasi Tren
- Line Chart Penjualan Global per Tahun: Tren penjualan global dari tahun ke tahun
- Line Chart Penjualan per Region per Tahun: Perbandingan tren penjualan di NA, EU, JP, dan Other

- Tren Genre Populer: Perkembangan genre populer dari waktu ke waktu

3. Visualisasi Perbandingan
- Bar Chart Publisher Teratas: 10 publisher dengan penjualan global tertinggi
- Bar Chart Game Terlaris: 20 game dengan penjualan global tertinggi
- Bar Chart Platform Terlaris: Perbandingan penjualan antar platform

4. Visualisasi Korelasi
- Scatter Plot Penjualan vs Tahun: korelasi antara tahun rilis dan penjualan
- Scatter Plot Penjualan Antar Region: Hubungan penjualan antara region berbeda

5. Visualisasi Geografis
- Bar Chart: Visualisasi distribusi daerah penjualan

""")

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("vgsales.csv")  # Pastikan file ini tersedia saat deploy
    return df

df = load_data()

# Display dataset info
st.subheader("Informasi Dataset")
buffer = io.StringIO()
df.info(buf=buffer)
st.text(buffer.getvalue())

# Display missing values
st.subheader("Jumlah Data Kosong (Sebelum Pembersihan)")
st.write(df.isna().sum())

# Remove missing values
df.dropna(inplace=True)

# Display missing values after cleaning
st.subheader("Jumlah Data Kosong (Setelah Pembersihan)")
st.write(df.isna().sum())

# Display duplicate count
st.subheader("Jumlah Data Duplikat")
st.write(f"Total duplikat: {df.duplicated().sum()}")

# margin top
st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Distribusi", "Tren", "Perbandingan", "Korelasi", "Geografis"]
)

# Main content based on selected page
if page == "Distribusi":
    st.header("Visualisasi Distribusi")

    # Visualisasi Distribusi Tahun Rilis Game
    st.subheader("Distribusi Tahun Rilis Game")
    
    # Buat sliders untuk rentang tahun
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

    # Tambah input untuk tahun perbandingan
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

    # Filter data berdasarkan rentang tahun
    filtered_df = df[(df['Year'] >= start_year) & (df['Year'] <= end_year)]

    # Buat plot dengan style asli
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.histplot(data=filtered_df, x="Year", bins=40, kde=False, ax=ax)
    
    plt.title(f'Distribusi Tahun Rilis Game ({start_year}-{end_year})', fontsize=12, fontweight='bold')
    plt.xlabel('Tahun Rilis', fontsize=10)
    plt.ylabel('Jumlah Game', fontsize=10)

    # Tambah informasi tahun puncak
    if not filtered_df.empty:
        peak_year = filtered_df['Year'].value_counts().idxmax()
        peak_year_count = filtered_df['Year'].value_counts().max()
        plt.text(start_year + (end_year-start_year)*0.6, plt.gca().get_ylim()[1]*0.9,
                f'Tahun dengan jumlah game terbanyak: {int(peak_year)} ({peak_year_count} game)',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))

    # Tambah perbandingan tahun
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

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    # Visualisasi Distribusi Genre
    st.subheader("Distribusi Genre Game")

    # Controls for genre visualization
    col1, col2, col3 = st.columns(3)
    with col1:
        max_genres = st.slider("Maksimum Genre", 5, len(df['Genre'].unique()), len(df['Genre'].unique()))
    with col2:
        min_percent = st.slider("Persentase Minimum (%)", 0.0, 5.0, 0.0, 0.1)
    with col3:
        highlight_genre = st.selectbox("Highlight Genre", ['None'] + sorted(df['Genre'].unique().tolist()))

    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate genre distribution
    genre_counts = df['Genre'].value_counts()
    genre_percent = (genre_counts / genre_counts.sum() * 100).round(1)

    # Apply filters
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

    # Create labels and colors 
    labels = [f'{genre}: {count} ({percent}%)' for genre, count, percent in 
            zip(genre_counts.index, genre_counts, genre_percent)]

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(genre_counts)))
    explode = [0.1 if genre == highlight_genre else 0 for genre in genre_counts.index]

    # Create pie chart
    plt.pie(genre_counts, explode=explode, labels=None, colors=colors,
        autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
        shadow=True, startangle=90)

    plt.title('Distribusi Game Berdasarkan Genre', pad=20)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    st.pyplot(fig)

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    # Visualisasi Distribusi Platform
    st.subheader("Distribusi Platform Game")

    # Create sliders and dropdown for platform distribution
    col1, col2 = st.columns(2)
    with col1:
        max_platforms = st.slider("Jumlah Platform", 5, 15, 10)
    with col2:
        min_percent = st.slider("Persentase Minimum", 0.0, 5.0, 0.0, 0.1)

    # Get top platforms for dropdown
    top_platforms = df['Platform'].value_counts().head(10).index.tolist()
    top_platforms.sort()
    highlight_platform = st.selectbox("Highlight Platform", ['None'] + top_platforms + ['Others'])

    # Create pie chart
    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate platform distribution
    platform_counts = df['Platform'].value_counts()
    platform_percent = (platform_counts / platform_counts.sum() * 100).round(1)

    # Apply filters
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

    # Create labels and colors
    labels = [f'{platform}: {count} ({percent}%)' for platform, count, percent in
            zip(platform_counts.index, platform_counts, platform_percent)]

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(platform_counts)))
    explode = [0.1 if platform == highlight_platform else 0 for platform in platform_counts.index]

    # Create pie chart
    plt.pie(platform_counts, explode=explode, labels=None, colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
            shadow=True, startangle=90)

    plt.title('Distribusi Game Berdasarkan Platform', pad=20)
    plt.legend(labels, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()
    st.pyplot(fig)

elif page == "Tren":
    st.header("Visualisasi Tren")

    # Create year range slider
    df_grouped = df.groupby('Year')['Global_Sales'].sum().reset_index()
    min_year = int(df_grouped['Year'].min())
    max_year = int(df_grouped['Year'].max())
    
    col1, col2 = st.columns(2)
    with col1:
        start_year = st.slider('Tahun Awal', min_year, max_year-1, min_year)
    with col2:
        end_year = st.slider('Tahun Akhir', min_year+1, max_year, max_year)

    # Create the plot
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

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    # Visualisasi Tren Penjualan per Region per Tahun
    st.subheader("Tren Penjualan per Region per Tahun")
    
    # Create year range slider
    df_grouped_region = df.groupby('Year')[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
    min_year = int(df_grouped_region['Year'].min())
    max_year = int(df_grouped_region['Year'].max())
    
    col1, col2 = st.columns(2)
    with col1:
        start_year_region = st.slider('Tahun Awal', min_year, max_year-1, min_year, key='region_start')
    with col2:
        end_year_region = st.slider('Tahun Akhir', min_year+1, max_year, max_year, key='region_end')

    # Region selection
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
        # Create the plot
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

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    # Visualisasi Tren Genre Populer
    st.subheader("Tren Genre Populer")

    # Create year range slider
    df_grouped_genre = df.groupby('Year')['Genre'].value_counts().reset_index()
    min_year = int(df_grouped_genre['Year'].min())
    max_year = int(df_grouped_genre['Year'].max())
    
    # Create year range slider
    year_range = st.slider(
        "Pilih Rentang Tahun",
        min_value=min_year,
        max_value=max_year,
        value=(min_year, max_year),
        step=1
    )
    
    # Create genre multiselect
    all_genres = sorted(df['Genre'].unique())
    selected_genres = st.multiselect(
        "Pilih Genre",
        options=all_genres,
        default=all_genres[:5]  # Default select first 5 genres
    )
    
    if selected_genres:
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        
        # Create heatmap
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

    # Add checkbox for all years
    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_1")

    if not show_all_years:
        # Create year range slider
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1
        )
        
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"
    
    # Add select all genres checkbox
    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_1")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        # Add multiselect for genres
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5]  # Default select first 5 genres
        )
    
    if selected_genres:
        # Filter data based on genres
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
        
        # Get top 10 publishers
        top_publishers = df_filtered.groupby('Publisher')['Global_Sales'].sum().sort_values(ascending=False).head(10).reset_index()
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=top_publishers, x='Global_Sales', y='Publisher', hue='Publisher', palette='viridis', ax=ax)
        
        plt.title(f'Top 10 Publisher dengan Penjualan Global Tertinggi {title_year_range}', fontsize=14)
        plt.xlabel('Total Global Sales (juta unit)')
        plt.ylabel('Publisher')
        
        # Add value labels on bars
        for p in ax.patches:
            ax.annotate(f"{p.get_width():.1f}", (p.get_width()+0.3, p.get_y() + 0.5), va='center')
            
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")


    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Top 20 Game Terlaris Secara Global")

    # Add checkbox for all years
    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_2")

    if not show_all_years:
        # Create year range slider
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_2"
        )
        
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    # Get top 20 games
    top_games = df_filtered[['Name', 'Global_Sales']].sort_values(by='Global_Sales', ascending=False).head(20)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.barplot(data=top_games, x='Global_Sales', y='Name', hue='Name', palette='mako', ax=ax)
    
    plt.title(f'Top 20 Game Terlaris Secara Global {title_year_range}', fontsize=14)
    plt.xlabel('Total Global Sales (juta unit)')
    plt.ylabel('Nama Game')
    
    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f"{p.get_width():.1f}", (p.get_width()+0.2, p.get_y() + 0.5), va='center')
        
    plt.tight_layout()
    st.pyplot(fig)

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Total Penjualan Global per Platform")

    # Add checkbox for all years
    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_platform")

    if not show_all_years:
        # Create year range slider
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_platform"
        )
        
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    # Add select all genres checkbox
    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_platform")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        # Add multiselect for genres
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            key="genre_select_platform"
        )

    # Filter by selected genres
    if selected_genres:
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
    else:
        title_genres = ""

    # Get platform sales data
    platform_sales = df_filtered.groupby('Platform')['Global_Sales'].sum().sort_values(ascending=False).reset_index()

    # Create bar plot
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

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2, p.get_height() + 0.2), ha='center')

    plt.tight_layout()
    st.pyplot(fig)


elif page == "Korelasi":
    st.header("Visualisasi Korelasi")

    st.subheader("Korelasi antara Tahun Rilis dan Penjualan Global")

    # Add checkbox for all years
    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_correlation")

    if not show_all_years:
        # Create year range slider
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1
        )
        
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    # Create genre multiselect with select all checkbox
    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5]  # Default select first 5 genres
        )

    if selected_genres:
        # Filter data based on genres
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"

        # Create scatter plot
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

        # Add regression line
        sns.regplot(
            x="Year",
            y="Global_Sales",
            data=df_filtered,
            scatter=False,
            color="red",
            line_kws={"linestyle": "--", "linewidth": 2},
            ax=ax
        )

        # Annotate top 5 games
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

    # margin top
    st.markdown("<hr style='margin-top: 5rem;'>", unsafe_allow_html=True)

    st.subheader("Korelasi antara Penjualan Global dan Penjualan di Region")

    # Create scatter plot
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

    # Add checkbox for all years
    show_all_years = st.checkbox("Tampilkan Semua Tahun", key="show_all_years_geo")

    if not show_all_years:
        # Create year range slider
        year_range = st.slider(
            "Pilih Rentang Tahun",
            min_value=int(df['Year'].min()),
            max_value=int(df['Year'].max()),
            value=(int(df['Year'].min()), int(df['Year'].max())),
            step=1,
            key="year_range_geo"
        )
        
        # Filter data based on selected year range
        df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]
        title_year_range = f"({year_range[0]}-{year_range[1]})"
    else:
        df_filtered = df
        title_year_range = "(Semua Tahun)"

    # Add select all genres checkbox
    all_genres = sorted(df['Genre'].unique())
    select_all_genres = st.checkbox("Pilih Semua Genre", key="select_all_genres_geo")
    
    if select_all_genres:
        selected_genres = all_genres
    else:
        # Add multiselect for genres
        selected_genres = st.multiselect(
            "Pilih Genre",
            options=all_genres,
            default=all_genres[:5],  # Default select first 5 genres
            key="genre_select_geo"
        )

    if selected_genres:
        # Filter data based on genres
        df_filtered = df_filtered[df_filtered['Genre'].isin(selected_genres)]
        title_genres = f" - Genre: {', '.join(selected_genres)}"
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Regional Sales Comparison
        regional_sales = df_filtered[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum()
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f1c40f']
        
        ax1.pie(regional_sales, labels=regional_sales.index, autopct='%1.1f%%', colors=colors)
        ax1.set_title(f'Distribusi Penjualan Regional {title_year_range}', fontsize=14)
        
        # Plot 2: Regional Sales Trend
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
        
        # Additional insights
        st.subheader("Analisis Regional")
        
        # Calculate regional statistics
        regional_stats = pd.DataFrame({
            'Total Penjualan': regional_sales,
            'Persentase': (regional_sales / regional_sales.sum() * 100).round(1),
            'Rata-rata per Game': (regional_sales / len(df_filtered)).round(2)
        })
        
        st.dataframe(regional_stats, use_container_width=True)
        
    else:
        st.warning("Pilih minimal satu genre untuk ditampilkan.")
