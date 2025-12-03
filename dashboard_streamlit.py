import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# BACA DATASET
df_raw = pd.read_csv("chronic_disease_children_trend.csv")

# PREPROCESSING
df = df_raw.copy()
numeric_cols = df.select_dtypes(include='number').columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# STANDARDISASI
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[numeric_cols])

# CLUSTERING
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df['kmeans_cluster'] = km.fit_predict(X_scaled)

# DASHBOARD
st.title("Dashboard Clustering: Penyakit Kronis Anak")

st.write("Data sample:")
st.dataframe(df.head())

st.sidebar.header("Filter")
cluster_choice = st.sidebar.multiselect(
    "Pilih cluster",
    sorted(df['kmeans_cluster'].unique()),
    default=sorted(df['kmeans_cluster'].unique())
)

filtered = df[df['kmeans_cluster'].isin(cluster_choice)]

st.subheader("Ukuran cluster")
st.bar_chart(filtered['kmeans_cluster'].value_counts().sort_index())

st.subheader("Profil rata-rata fitur per cluster")
st.dataframe(filtered.groupby('kmeans_cluster')[numeric_cols].mean())

st.subheader("Scatter plot dua fitur")
x = st.selectbox("X", numeric_cols, index=0)
y = st.selectbox("Y", numeric_cols, index=1)

df_chart = filtered.copy()

chart = alt.Chart(df_chart).mark_circle(size=60).encode(
    x=x,
    y=y,
    color='kmeans_cluster:N',
).interactive()

st.altair_chart(chart, use_container_width=True)

# SPK BERDASARKAN PROVINSI (PROVINCE-BASED SPK)
st.subheader("🌍 Sistem Pendukung Keputusan Berdasarkan Provinsi")

# Pilih provinsi dari dataset
selected_province = st.selectbox(
    "Pilih Provinsi:",
    sorted(df['Province'].unique())
)

# Ambil data provinsi yang dipilih
prov_data = df[df['Province'] == selected_province].iloc[0]

# Tampilkan nilai prevalensi
st.write("### 📊 Data Kesehatan Provinsi")
st.write(f"- **Asma**: {prov_data['Asthma_Prevalence_pct']}%")
st.write(f"- **Pneumonia**: {prov_data['Pneumonia_Prevalence_pct']}%")
st.write(f"- **Anemia**: {prov_data['Anemia_Prevalence_pct']}%")

# Prediksi cluster provinsi (sudah dihitung di dataframe)
prov_cluster = prov_data['kmeans_cluster']

st.write(f"### 🏷️ Provinsi Ini Termasuk Cluster **{prov_cluster}**")

def spk_rekomendasi(cluster):
    rules = {
        0: [
            "Pantau peningkatan kasus pneumonia musiman.",
            "Tingkatkan edukasi kebersihan dan ventilasi rumah.",
            "Penguatan skrining pneumonia di Puskesmas.",
        ],
        1: [
            "Prioritaskan program pengendalian asma.",
            "Perbaiki kualitas udara (polusi, asap kendaraan).",
            "Edukasi orang tua terkait penggunaan inhaler & pencegahan serangan asma.",
        ],
        2: [
            "Provinsi berada di zona aman — prevalensi rendah.",
            "Tetap lakukan monitoring rutin kesehatan anak.",
            "Pertahankan program gizi dan imunisasi.",
        ],
        3: [
            "Perkuat program penanganan anemia (zat besi, edukasi gizi).",
            "Lakukan intervensi pneumonia seperti imunisasi PCV dan ventilasi rumah.",
            "Provinsi termasuk risiko tinggi — perlu menjadi prioritas pemerintah daerah.",
        ]
    }
    return rules.get(cluster, ["Cluster tidak ditemukan."])

st.write("Rekomendasi Kesehatan untuk Provinsi Ini:")

for rec in spk_rekomendasi(prov_cluster):
    st.write("- ", rec)
