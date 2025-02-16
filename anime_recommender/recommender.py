import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader, TensorDataset

df = pd.read_csv('cleaned_anime_info.csv')

df['genres'] = df['genres'].apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_)
df = pd.concat([df, genre_df], axis=1)

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['description'])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_recommendations(title, cosine_sim=cosine_sim):
    if title not in df['title_romaji'].values:
        print(f"Title '{title}' not found in the dataset.")
        return []

    idx = df.index[df['title_romaji'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return df['title_romaji'].iloc[sim_indices]

recommendations = get_recommendations('cowboy bebop')
print("TF-IDF Based Recommendations:")
print(recommendations)

class AnimeAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AnimeAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = tfidf_matrix.shape[1]
encoding_dim = 100

model = AnimeAutoencoder(input_dim, encoding_dim)

tfidf_tensor = torch.tensor(tfidf_matrix.toarray(), dtype=torch.float32)

dataset = TensorDataset(tfidf_tensor, tfidf_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    for data in dataloader:
        inputs, _ = data
        optimizer.zero_grad()
        _, outputs = model(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

with torch.no_grad():
    encoded_representations, _ = model(tfidf_tensor)

latent_cosine_sim = cosine_similarity(encoded_representations.numpy())

def get_recommendations_latent(title, latent_cosine_sim=latent_cosine_sim):
    idx = df.index[df['title_romaji'] == title].tolist()[0]
    sim_scores = list(enumerate(latent_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_indices = [i[0] for i in sim_scores[1:11]]
    return df['title_romaji'].iloc[sim_indices]

latent_recommendations = get_recommendations_latent('cowboy bebop')
print("Latent Feature-Based Recommendations:")
print(latent_recommendations)
