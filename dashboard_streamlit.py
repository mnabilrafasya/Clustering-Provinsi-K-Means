import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Analisis Clustering Penyakit Kronis pada Anak (Per Provinsi)")

# BACA DATASET
df_raw = pd.read_csv("chronic_disease_children_trend.csv")
df = df_raw.copy()

features = ['Asthma_Prevalence_pct',
            'Pneumonia_Prevalence_pct',
            'Anemia_Prevalence_pct']

# Isi missing value
df[features] = df[features].fillna(df[features].median())

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# CLUSTERING (sesuai rekomendasi: K=4)
k = 4
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = km.fit_predict(X_scaled)

st.subheader("📌 Hasil Clustering per Provinsi")
st.dataframe(df)

# PROFIL CLUSTER
cluster_profile = df.groupby("Cluster")[features].mean().round(2)

st.subheader("📊 Profil Rata-rata Fitur per Cluster")
st.dataframe(cluster_profile)

# Evaluasi Clustering
sil = silhouette_score(X_scaled, df["Cluster"])
dbi = davies_bouldin_score(X_scaled, df["Cluster"])

st.write("### 🔎 Evaluasi Clustering")
st.write(f"- Silhouette Score: **{sil:.4f}**")
st.write(f"- Davies-Bouldin Index: **{dbi:.4f}**")

# SCATTER PLOT
st.subheader("🔍 Visualisasi Scatter Plot")
x = st.selectbox("Pilih fitur X:", features, index=0)
y = st.selectbox("Pilih fitur Y:", features, index=1)

chart = (
    alt.Chart(df)
    .mark_circle(size=80)
    .encode(
        x=x,
        y=y,
        color='Cluster:N',
        tooltip=['Province', x, y, 'Cluster']
    )
    .interactive()
)
st.altair_chart(chart, use_container_width=True)

# HEATMAP CLUSTER PROFILE
st.subheader("🔥 Heatmap Profil Cluster")

fig, ax = plt.subplots(figsize=(6,4))
sns.heatmap(cluster_profile.T, annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
st.pyplot(fig)

# INSIGHT CLUSTER
st.subheader("🧠 Insight Kesehatan per Cluster")

for c in sorted(df["Cluster"].unique()):
    st.markdown(f"### **Cluster {c}**")

    # Provinsi di cluster ini
    provs = df[df["Cluster"] == c]["Province"].unique().tolist()
    st.write("📌 **Provinsi:**", ", ".join(provs))

    # Karakteristik prevalensi
    st.write("**Karakteristik Prevalensi:**")
    for feat in features:
        st.write(f"- {feat.replace('_Prevalence_pct', '')}: {cluster_profile.loc[c, feat]}%")

# SPK - SISTEM PENDUKUNG KEPUTUSAN
st.subheader("🌍 SPK: Rekomendasi Berdasarkan Provinsi dan Tahun")

selected_prov = st.selectbox("Pilih Provinsi:", sorted(df["Province"].unique()))
selected_year = st.selectbox("Pilih Tahun:", sorted(df["Year"].unique()))

data_p = df[(df["Province"] == selected_prov) & (df["Year"] == selected_year)].iloc[0]


st.write("### 📊 Data Prevalensi")
st.write(f"- Asma: {data_p['Asthma_Prevalence_pct']:.2f}%")
st.write(f"- Pneumonia: {data_p['Pneumonia_Prevalence_pct']:.2f}%")
st.write(f"- Anemia: {data_p['Anemia_Prevalence_pct']:.2f}%")

cl = data_p["Cluster"]
st.write(f"### 🏷️ Masuk Cluster **{cl}**")

def rekomendasi(cluster):
    rules = {
        0: [
            "Lakukan skrining HB rutin setiap 6 bulan di posyandu.",
            "Perkuat edukasi konsumsi sumber zat besi hewani (daging, hati) dan sayuran hijau.",
            "Tingkatkan kebersihan & sanitasi untuk mencegah infeksi penyebab anemia.",
            "Program PMT (Pemberian Makanan Tambahan) ringan untuk anak rentan.",
        ],
        1: [
            "Fokus pada peningkatan asupan gizi mikro seperti zat besi, folat, dan vitamin B12.",
            "Selenggarakan edukasi gizi keluarga untuk mengurangi risiko anemia berulang.",
            "Perkuat monitoring pertumbuhan anak setiap bulan.",
            "Dorong diversifikasi pangan lokal yang kaya zat besi dan protein.",
        ],
        2: [
            "Lakukan skrining kualitas udara dalam rumah (ventilasi, asap rokok).",
            "Program PMT untuk anak berisiko.",
            "Edukasi manajemen asma (hindari pemicu, inhaler untuk pasien).",
            "Perbaikan gizi zat besi + vitamin C.",
        ],
        3: [
            "Prioritaskan suplementasi zat besi terjadwal bagi anak usia dini.",
            "Akselerasi imunisasi PCV untuk mencegah pneumonia berat.",
            "Intervensi kualitas udara (polusi, kepadatan, ventilasi).",
            "Deteksi dini ISPA melalui puskesmas/posyandu.",
        ]
    }
    return rules.get(cluster, ["Rekomendasi tidak tersedia."])

st.write("### 🩺 Rekomendasi Kesehatan:")
for r in rekomendasi(cl):
    st.write("- ", r)

st.subheader("🧪 SPK: Simulasi Prediksi dari Input Persentase Baru")

input_asma = st.number_input(
    "Masukkan Persentase Asma (%)", 
    min_value=0.0, max_value=100.0, value=5.0
)

input_pneumonia = st.number_input(
    "Masukkan Persentase Pneumonia (%)", 
    min_value=0.0, max_value=100.0, value=5.0
)

input_anemia = st.number_input(
    "Masukkan Persentase Anemia (%)", 
    min_value=0.0, max_value=100.0, value=10.0
)

# Gabungkan input ke array
input_data = [[input_asma, input_pneumonia, input_anemia]]

# Standarisasi menggunakan scaler yang sama
input_scaled = scaler.transform(input_data)

# Prediksi cluster dari KMeans
predicted_cluster = km.predict(input_scaled)[0]

st.write(f"### 🏷️ Prediksi Masuk ke Cluster **{predicted_cluster}**")

st.write("### 🩺 Rekomendasi Kesehatan (Berdasarkan Prediksi):")
for r in rekomendasi(predicted_cluster):
    st.write("- ", r)
