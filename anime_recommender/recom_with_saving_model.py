import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

MODEL_SAVE_PATH = "anime_recommender_model.pth"

df = pd.read_csv('cleaned_anime_info.csv')

df['genres'] = df['genres'].apply(lambda x: x.split(', '))
mlb = MultiLabelBinarizer()
genre_encoded = mlb.fit_transform(df['genres'])
genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)

scaler = MinMaxScaler()
normalized_score = scaler.fit_transform(df[['averageScore']])
score_df = pd.DataFrame(normalized_score, columns=['normalized_score'], index=df.index)

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['description'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=df.index)

feature_df = pd.concat([tfidf_df, genre_df, score_df], axis=1)
feature_matrix = feature_df.values
feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)

class AnimeAutoencoderCombined(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AnimeAutoencoderCombined, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

input_dim = feature_matrix.shape[1]
encoding_dim = 128

model = AnimeAutoencoderCombined(input_dim, encoding_dim).to(device)

try:
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    print(f"Loaded trained model from: {MODEL_SAVE_PATH}")
    model.eval()
except FileNotFoundError:
    print("No saved model found. Training model from scratch.")
    dataset = TensorDataset(feature_tensor, feature_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Trained model saved to: {MODEL_SAVE_PATH}")
    model.eval()

with torch.no_grad():
    encoded_representations, _ = model(feature_tensor)

latent_cosine_sim = cosine_similarity(encoded_representations.cpu().numpy())

def get_recommendations_latent(title, latent_cosine_sim=latent_cosine_sim):
    if title not in df['title_romaji'].values:
        print(f"Title '{title}' not found in the dataset.")
        return []
    idx = df.index[df['title_romaji'] == title].tolist()[0]
    sim_scores = list(enumerate(latent_cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    recommended_series = {}
    recommendations = []
    count = 0

    input_title_lower = title.lower()

    def get_series_identifier(anime_title):
        title_lower = anime_title.lower()
        words = title_lower.split()
        if "digimon" in title_lower:
            return "digimon"
        elif "pokemon" in title_lower or "pocket monsters" in title_lower:
            return "pokemon"
        elif len(words) >= 2:
            return " ".join(words[:2])
        elif words:
            return words[0]
        else:
            return anime_title.lower()

    input_series_id = get_series_identifier(title)

    for i in range(1, len(sim_scores)):
        sim_index = sim_scores[i][0]
        recommended_title = df['title_romaji'].iloc[sim_index]
        recommended_series_id = get_series_identifier(recommended_title)

        if recommended_series_id != input_series_id:
            if recommended_series_id not in recommended_series:
                recommended_series[recommended_series_id] = recommended_title
                recommendations.append(recommended_title)
                count += 1
                if count >= 10:
                    break

    return pd.Series(recommendations)

latent_recommendations = get_recommendations_latent('Cowboy Bebop')
print("Latent Feature-Based Recommendations (Combined Features - Genres, Score, Description):")
print(latent_recommendations)