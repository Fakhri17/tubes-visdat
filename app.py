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

st.set_page_config(page_title="Visualisasi Data Game", layout="wide")

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
try:
    df = pd.read_csv("vgsales.csv")  # Pastikan file ini tersedia saat deploy
    st.subheader("Dataset Penjualan Video Game")
    st.dataframe(df)
except FileNotFoundError:
    st.error("File 'vgsales.csv' tidak ditemukan. Pastikan file ini tersedia di direktori yang sama dengan aplikasi.")
    st.stop()

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

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribusi", "Tren", "Perbandingan", "Korelasi", "Geografis"])

with tab1:
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

  # Buat plot
  fig, ax = plt.subplots(figsize=(12, 8))  # Reduced from (12, 6) to (8, 4)
  sns.histplot(data=filtered_df, x="Year", bins=40, kde=False, ax=ax)
  
  plt.title(f'Distribusi Tahun Rilis Game ({start_year}-{end_year})', fontsize=12, fontweight='bold')  # Reduced font size
  plt.xlabel('Tahun Rilis', fontsize=10)  # Reduced font size
  plt.ylabel('Jumlah Game', fontsize=10)  # Reduced font size

  # Tambah informasi tahun puncak
  if not filtered_df.empty:
      peak_year = filtered_df['Year'].value_counts().idxmax()
      peak_year_count = filtered_df['Year'].value_counts().max()
      plt.text(start_year + (end_year-start_year)*0.6, plt.gca().get_ylim()[1]*0.9,
              f'Tahun dengan jumlah game terbanyak: {int(peak_year)} ({peak_year_count} game)',
              fontsize=8,  # Added smaller font size
              bbox=dict(boxstyle="round,pad=0.3", fc='yellow', alpha=0.3))

  # Tambah perbandingan tahun
  year1_count = df[df['Year'] == compare_year1].shape[0]
  year2_count = df[df['Year'] == compare_year2].shape[0]
  if year1_count > 0:
      perc_diff = ((year2_count - year1_count) / year1_count * 100)
      plt.text(start_year + (end_year-start_year)*0.1, plt.gca().get_ylim()[1]*0.7,
              f'Perbandingan {compare_year2} vs {compare_year1}:\n{compare_year2}: {year2_count} game\n{compare_year1}: {year1_count} game\nPerbedaan: {perc_diff:.1f}%',
              fontsize=8,  # Added smaller font size
              bbox=dict(boxstyle="round,pad=0.3", fc='lightblue', alpha=0.3))

  plt.tight_layout()  # Added to prevent text overlap
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

with tab2:
    st.header("Visualisasi Tren")