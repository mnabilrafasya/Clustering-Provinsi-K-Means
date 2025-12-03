import streamlit as st
import pandas as pd
import altair as alt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
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

# AGREGASI PER PROVINSI
df_prov = df.groupby("Province")[features].mean().reset_index()

# Standardisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_prov[features])

# CLUSTERING (sesuai rekomendasi: K=3)
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init=10)
df_prov["Cluster"] = km.fit_predict(X_scaled)

st.subheader("📌 Hasil Clustering per Provinsi")
st.dataframe(df_prov)

# PROFIL CLUSTER
cluster_profile = df_prov.groupby("Cluster")[features].mean().round(2)

st.subheader("📊 Profil Rata-rata Fitur per Cluster")
st.dataframe(cluster_profile)

# SCATTER PLOT
st.subheader("🔍 Visualisasi Scatter Plot")
x = st.selectbox("Pilih fitur X:", features, index=0)
y = st.selectbox("Pilih fitur Y:", features, index=1)

chart = (
    alt.Chart(df_prov)
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

for c in sorted(df_prov["Cluster"].unique()):
    st.markdown(f"### **Cluster {c}**")

    # Provinsi di cluster ini
    provs = df_prov[df_prov["Cluster"] == c]["Province"].tolist()
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

selected_prov = st.selectbox("Pilih Provinsi:", sorted(df_prov["Province"].unique()))
data_p = df_prov[df_prov["Province"] == selected_prov].iloc[0]

st.write("### 📊 Data Prevalensi")
st.write(f"- Asma: {data_p['Asthma_Prevalence_pct']:.2f}%")
st.write(f"- Pneumonia: {data_p['Pneumonia_Prevalence_pct']:.2f}%")
st.write(f"- Anemia: {data_p['Anemia_Prevalence_pct']:.2f}%")

cl = data_p["Cluster"]
st.write(f"### 🏷️ Masuk Cluster **{cl}**")

def rekomendasi(cluster):
    rules = {
        0: [
            "Pantau kemungkinan peningkatan kasus pneumonia.",
            "Perbaiki ventilasi rumah dan edukasi kebersihan.",
            "Perkuat skrining pneumonia anak di fasilitas kesehatan.",
        ],
        1: [
            "Prioritaskan penanganan asma (edukasi inhaler & kualitas udara).",
            "Kampanye pencegahan serangan asma secara rutin.",
        ],
        2: [
            "Prevalensi rendah — tetap lakukan monitoring anak.",
            "Pertahankan program imunisasi & gizi.",
        ]
    }
    return rules.get(cluster, ["Rekomendasi tidak tersedia."])

st.write("### 🩺 Rekomendasi Kesehatan:")
for r in rekomendasi(cl):
    st.write("- ", r)
