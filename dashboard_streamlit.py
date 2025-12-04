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

# CLUSTERING (sesuai rekomendasi: K=3)
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

    # Penyakit dominan
    dominant = cluster_profile.loc[c].idxmax().replace("_Prevalence_pct", "")
    value = cluster_profile.loc[c].max()
    st.write(f"🔥 **Penyakit dominan:** {dominant} ({value}%)")
    st.write("---")

# SPK - SISTEM PENDUKUNG KEPUTUSAN
st.subheader("🌍 SPK: Rekomendasi Berdasarkan Provinsi")

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
            "Perkuat skrining Hemoglobin (HB) secara berkala di posyandu.",
            "Tingkatkan edukasi konsumsi makanan kaya zat besi seperti daging, hati, telur, dan sayuran hijau.",
            "Perbaiki sanitasi dan kebersihan lingkungan untuk mencegah infeksi yang dapat memperburuk anemia.",
            "Program suplementasi zat besi untuk anak-anak dengan risiko tinggi.",
        ],
        1: [
            "Fokus pada peningkatan asupan gizi mikro seperti zat besi, folat, dan vitamin B12.",
            "Selenggarakan edukasi gizi keluarga untuk mengurangi risiko anemia berulang.",
            "Perkuat monitoring pertumbuhan anak setiap bulan.",
            "Dorong diversifikasi pangan lokal yang kaya zat besi dan protein.",
        ],
        2: [
            "Lakukan pemantauan rutin untuk memastikan prevalensi anemia tetap terkendali.",
            "Lanjutkan program imunisasi dan PMT sebagai pencegahan jangka panjang.",
            "Edukasi orang tua mengenai kombinasi makanan tinggi zat besi dan vitamin C.",
            "Tingkatkan akses makanan bergizi melalui program sekolah dan komunitas.",
        ],
        3: [
            "Prioritaskan suplementasi zat besi terjadwal bagi anak usia dini.",
            "Lakukan skrining HB lebih sering, terutama untuk kelompok berisiko.",
            "Perkuat intervensi gizi berbasis sekolah seperti PMT dan menu bergizi di PAUD.",
            "Koordinasi dengan puskesmas untuk penanganan cepat kasus anemia sedang–berat.",
        ]
    }
    return rules.get(cluster, ["Rekomendasi tidak tersedia."])

st.write("### 🩺 Rekomendasi Kesehatan:")
for r in rekomendasi(cl):
    st.write("- ", r)
