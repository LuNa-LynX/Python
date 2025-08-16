# 📦 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
df = pd.read_csv(r"C:\Users\Swathi Krishna\Downloads\spotify dataset.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)

# Feature Engineering
df['beat_strength'] = df['energy'] * df['tempo']
df['mood_score'] = df['valence'] * df['danceability']
df['acoustic_energy_ratio'] = df['acousticness'] / (df['energy'] + 1e-6)

# Numerical Features
num_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
    'duration_ms', 'beat_strength', 'mood_score', 'acoustic_energy_ratio'
]

# Outlier Removal
df = df[(np.abs(stats.zscore(df[num_features])) < 3).all(axis=1)]

# Feature Scaling
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[num_features])

# Dimensionality Reduction
pca = PCA(n_components=2)
features_pca = pca.fit_transform(scaled_features)

# Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

#visualize

# 1. Distribution Plot
plt.figure(figsize=(8, 4))
sns.histplot(df['danceability'], kde=True, color='skyblue')
plt.title('Danceability Distribution')
plt.show()

# 2. Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df[num_features].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()

# 3. Cluster Visualization (PCA)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=features_pca[:, 0], y=features_pca[:, 1], hue=df['cluster'], palette='Set2')
plt.title('Song Clusters (PCA Reduced)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# 4. Cluster Distribution by Playlist Genre
plt.figure(figsize=(8, 5))
sns.countplot(x='playlist_genre', hue='cluster', data=df, palette='Set3')
plt.title('Cluster Distribution by Playlist Genre')
plt.xticks(rotation=45)
plt.show()

# 5. Recommendation System
similarity_matrix = cosine_similarity(scaled_features)

def recommend(index, top_n=5):
    scores = list(enumerate(similarity_matrix[index]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recommended_indices = [i[0] for i in sorted_scores[1:top_n+1]]
    return df.iloc[recommended_indices][['track_name', 'track_artist', 'playlist_genre', 'cluster']]


print("Recommended songs similar to index 42:")
print(recommend(42))
