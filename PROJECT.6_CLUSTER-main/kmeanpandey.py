import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="Iris Flower Clustering",
    page_icon="🌸",
    layout="wide"
)

# st.title("🌸 Iris Flower Clustering using K-Means")
# st.markdown("Clustering Iris flowers using **Petal Length** and **Petal Width**.")
st.markdown("""
<h1 style='text-align:center;
color:white;
background:linear-gradient(90deg,#6a11cb,#2575fc);
padding:18px;
border-radius:15px;'>
🌸 Iris Flower Clustering using K-Means
</h1>
""", unsafe_allow_html=True)

# -------------------- Load Dataset --------------------
iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df = df[['petal length (cm)', 'petal width (cm)']]

st.subheader("📄 Dataset")
st.dataframe(df.head())

# -------------------- Scaling --------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# -------------------- Slider --------------------
k = st.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

# -------------------- KMeans --------------------
km = KMeans(
    n_clusters=k,
    random_state=0,
    n_init=10
)

clusters = km.fit_predict(scaled_data)

df["Cluster"] = clusters

# Convert centroid back to original scale
centroids = scaler.inverse_transform(km.cluster_centers_)

# -------------------- Cluster Plot --------------------
st.subheader("📊 Cluster Visualization")

fig, ax = plt.subplots(figsize=(7,5))

scatter = ax.scatter(
    df["petal length (cm)"],
    df["petal width (cm)"],
    c=df["Cluster"],
    cmap="viridis",
    s=70
)

ax.scatter(
    centroids[:,0],
    centroids[:,1],
    color="red",
    marker="*",
    s=350,
    label="Centroids"
)

ax.set_xlabel("Petal Length (cm)")
ax.set_ylabel("Petal Width (cm)")
ax.set_title(f"K-Means Clustering (K = {k})")
ax.legend()

st.pyplot(fig)

# -------------------- Elbow Method --------------------
st.subheader("📈 Elbow Method")

sse = []

for i in range(1,11):
    model = KMeans(
        n_clusters=i,
        random_state=0,
        n_init=10
    )
    model.fit(scaled_data)
    sse.append(model.inertia_)

fig2, ax2 = plt.subplots(figsize=(7,5))

ax2.plot(
    range(1,11),
    sse,
    marker="o",
    linewidth=2
)

ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("SSE")
ax2.set_title("Elbow Plot")

st.pyplot(fig2)

st.success("✅ Optimal number of clusters is approximately **3**.")

# -------------------- Sidebar --------------------
st.sidebar.title("👨‍💻 Developer")

st.sidebar.markdown("### Richeek Pandey")

st.sidebar.markdown(
"""
🔗 **LinkedIn**

https://www.linkedin.com/in/richeek-pandey-9954783a9
"""
)

st.sidebar.markdown(
"""
💻 **GitHub**

https://github.com/richeekpandey07
"""
)

st.sidebar.markdown("---")
st.sidebar.info(
"""
📌 **Project**

Iris Flower Clustering using K-Means

Built with ❤️ using Python, Scikit-Learn & Streamlit.
"""
)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
"""
<div style='text-align:center;color:gray;'>
Made with ❤️ by <b>Richeek Pandey</b><br>
Python • Scikit-Learn • Streamlit
</div>
""",
unsafe_allow_html=True
)
