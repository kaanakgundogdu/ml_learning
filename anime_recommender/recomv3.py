import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sentence_transformers import SentenceTransformer
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("GPU is not available. Using CPU.")

MODEL_SAVE_PATH = "anime_recommender_model_v2.pth"
FEATURE_SAVE_PATH = "anime_features_v2.npz"

df = pd.read_csv('cleaned_anime_info.csv')

description_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class AnimeAutoencoderCombinedV2(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(AnimeAutoencoderCombinedV2, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim // 4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim // 4, encoding_dim // 2),
            nn.ReLU(),
            nn.Linear(encoding_dim // 2, encoding_dim),
            nn.ReLU(),
            nn.Linear(encoding_dim, encoding_dim * 2),
            nn.ReLU(),
            nn.Linear(encoding_dim * 2, input_dim),
            nn.ReLU()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

feature_matrix = None
latent_cosine_sim_v2 = None

if os.path.exists(FEATURE_SAVE_PATH):
    print(f"Loading pre-computed features from: {FEATURE_SAVE_PATH}")
    features_data = np.load(FEATURE_SAVE_PATH)
    feature_matrix = features_data['feature_matrix']
    latent_cosine_sim_v2 = features_data['latent_cosine_sim_v2']
else:
    print("Pre-computed features not found. Processing features...")
    df['genres'] = df['genres'].apply(lambda x: x.split(', '))
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['genres'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index).astype('int64')

    scaler = MinMaxScaler()
    normalized_score = scaler.fit_transform(df[['averageScore']])
    score_df = pd.DataFrame(normalized_score, columns=['normalized_score'], index=df.index).astype('float32')

    description_embeddings = description_embedding_model.encode(df['description'].tolist(), convert_to_tensor=True)
    description_df = pd.DataFrame(description_embeddings.cpu().numpy(), index=df.index).astype('float32')

    format_df = pd.get_dummies(df['format'], prefix='format', dummy_na=False)
    format_df.index = df.index
    format_df = format_df.astype('int64')

    feature_df = pd.concat([description_df, format_df, genre_df, score_df], axis=1)
    feature_matrix = feature_df.values
    feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)

    input_dim = feature_matrix.shape[1]
    encoding_dim = 128
    model_v2 = AnimeAutoencoderCombinedV2(input_dim, encoding_dim).to(device)

    try:
        model_v2.load_state_dict(torch.load(MODEL_SAVE_PATH))
        print(f"Loaded trained model from: {MODEL_SAVE_PATH}")
        model_v2.eval()
    except FileNotFoundError:
        print("No saved model found. Training model from scratch (Version 2).")
        dataset = TensorDataset(feature_tensor, feature_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model_v2.parameters(), lr=0.001)
        epochs = 10
        for epoch in range(epochs):
            for data in dataloader:
                inputs, _ = data
                inputs = inputs.to(device)
                optimizer.zero_grad()
                _, outputs = model_v2(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        torch.save(model_v2.state_dict(), MODEL_SAVE_PATH)
        print(f"Trained model saved to: {MODEL_SAVE_PATH}")
        model_v2.eval()

    with torch.no_grad():
        encoded_representations_v2, _ = model_v2(feature_tensor)
    latent_cosine_sim_v2 = cosine_similarity(encoded_representations_v2.cpu().numpy())

    print(f"Saving features and cosine similarity to: {FEATURE_SAVE_PATH}")
    np.savez(FEATURE_SAVE_PATH, feature_matrix=feature_matrix, latent_cosine_sim_v2=latent_cosine_sim_v2)


def get_recommendations_latent_v2(title, latent_cosine_sim=latent_cosine_sim_v2):
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

latent_recommendations_v2 = get_recommendations_latent_v2('Naruto')
print("Latent Feature-Based Recommendations (V2 - Sentence Embeddings, Format, Deeper Model):")
print(latent_recommendations_v2)



# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
# from torch.utils.data import DataLoader, TensorDataset
# import numpy as np
# from sentence_transformers import SentenceTransformer

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("GPU is available. Using GPU:", torch.cuda.get_device_name(0))
# else:
#     device = torch.device("cpu")
#     print("GPU is not available. Using CPU.")

# MODEL_SAVE_PATH = "anime_recommender_model_v2.pth" # New model save path for version 2

# df = pd.read_csv('cleaned_anime_info.csv')

# # 1. Encode Genres
# df['genres'] = df['genres'].apply(lambda x: x.split(', '))
# mlb = MultiLabelBinarizer()
# genre_encoded = mlb.fit_transform(df['genres'])
# genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index).astype('int64') # Convert genre_df to int64
# print("\nData types of genre_df (after astype):")
# print(genre_df.dtypes)

# # 2. Normalize Average Score
# scaler = MinMaxScaler()
# normalized_score = scaler.fit_transform(df[['averageScore']])
# score_df = pd.DataFrame(normalized_score, columns=['normalized_score'], index=df.index).astype('float32') # Convert score_df to float32
# print("\nData types of score_df (after astype):")
# print(score_df.dtypes)

# # 3. Get Sentence Embeddings for Descriptions
# description_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
# description_embeddings = description_embedding_model.encode(df['description'].tolist(), convert_to_tensor=True)
# description_df = pd.DataFrame(description_embeddings.cpu().numpy(), index=df.index).astype('float32') # Convert description_df to float32
# print("\nData types of description_df (after astype):")
# print(description_df.dtypes)

# # 4. One-Hot Encode 'format'
# format_df = pd.get_dummies(df['format'], prefix='format', dummy_na=False)
# format_df.index = df.index
# format_df = format_df.astype('int64') # Convert format_df to int64
# print("\nData types of format_df (after astype):")
# print(format_df.dtypes)

# # 5. Concatenate Features (Now including format)
# feature_df = pd.concat([description_df, format_df, genre_df, score_df], axis=1)
# print("\nData types of feature_df (after concat and astype on components):") # Modified print statement
# print(feature_df.dtypes)
# feature_matrix = feature_df.values
# print(f"Data type of feature_matrix (after astype on components): {feature_matrix.dtype}") # Modified print statement for numpy array dtype
# feature_tensor = torch.tensor(feature_matrix, dtype=torch.float32).to(device)


# # 6. Define Deeper Autoencoder Model (Version 2)
# class AnimeAutoencoderCombinedV2(nn.Module):
#     def __init__(self, input_dim, encoding_dim):
#         super(AnimeAutoencoderCombinedV2, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, encoding_dim * 2),
#             nn.ReLU(),
#             nn.Linear(encoding_dim * 2, encoding_dim),
#             nn.ReLU(),
#             nn.Linear(encoding_dim, encoding_dim // 2),
#             nn.ReLU(),
#             nn.Linear(encoding_dim // 2, encoding_dim // 4),
#             nn.ReLU()
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(encoding_dim // 4, encoding_dim // 2),
#             nn.ReLU(),
#             nn.Linear(encoding_dim // 2, encoding_dim),
#             nn.ReLU(),
#             nn.Linear(encoding_dim, encoding_dim * 2),
#             nn.ReLU(),
#             nn.Linear(encoding_dim * 2, input_dim),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return encoded, decoded


# # 7. Initialize Model, DataLoader, Loss, Optimizer
# input_dim = feature_matrix.shape[1]
# encoding_dim = 128

# model_v2 = AnimeAutoencoderCombinedV2(input_dim, encoding_dim).to(device)

# try:
#     model_v2.load_state_dict(torch.load(MODEL_SAVE_PATH))
#     print(f"Loaded trained model from: {MODEL_SAVE_PATH}")
#     model_v2.eval()
# except FileNotFoundError:
#     print("No saved model found. Training model from scratch (Version 2).")
#     dataset = TensorDataset(feature_tensor, feature_tensor)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model_v2.parameters(), lr=0.001)
#     epochs = 10
#     for epoch in range(epochs):
#         for data in dataloader:
#             inputs, _ = data
#             inputs = inputs.to(device)
#             optimizer.zero_grad()
#             _, outputs = model_v2(inputs)
#             loss = criterion(outputs, inputs)
#             loss.backward()
#             optimizer.step()
#         print(f'Epoch {epoch+1}, Loss: {loss.item()}')
#     torch.save(model_v2.state_dict(), MODEL_SAVE_PATH)
#     print(f"Trained model saved to: {MODEL_SAVE_PATH}")
#     model_v2.eval()


# # 8. Obtain Encoded Representations (using Version 2 Model)
# with torch.no_grad():
#     encoded_representations_v2, _ = model_v2(feature_tensor)

# # 9. Compute Cosine Similarity in Latent Space (using Version 2 Representations)
# latent_cosine_sim_v2 = cosine_similarity(encoded_representations_v2.cpu().numpy())

# # 10. Recommendation Function (Latent Features - Series Filtering - using Version 2 Cosine Sim)
# def get_recommendations_latent_v2(title, latent_cosine_sim=latent_cosine_sim_v2):
#     if title not in df['title_romaji'].values:
#         print(f"Title '{title}' not found in the dataset.")
#         return []
#     idx = df.index[df['title_romaji'] == title].tolist()[0]
#     sim_scores = list(enumerate(latent_cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

#     recommended_series = {}
#     recommendations = []
#     count = 0

#     input_title_lower = title.lower()

#     def get_series_identifier(anime_title):
#         title_lower = anime_title.lower()
#         words = title_lower.split()
#         if "digimon" in title_lower:
#             return "digimon"
#         elif "pokemon" in title_lower or "pocket monsters" in title_lower:
#             return "pokemon"
#         elif len(words) >= 2:
#             return " ".join(words[:2])
#         elif words:
#             return words[0]
#         else:
#             return anime_title.lower()

#     input_series_id = get_series_identifier(title)

#     for i in range(1, len(sim_scores)):
#         sim_index = sim_scores[i][0]
#         recommended_title = df['title_romaji'].iloc[sim_index]
#         recommended_series_id = get_series_identifier(recommended_title)

#         if recommended_series_id != input_series_id:
#             if recommended_series_id not in recommended_series:
#                 recommended_series[recommended_series_id] = recommended_title
#                 recommendations.append(recommended_title)
#                 count += 1
#                 if count >= 10:
#                     break

#     return pd.Series(recommendations)

# # 11. Example Usage and Output (Version 2 Recommendations)
# latent_recommendations_v2 = get_recommendations_latent_v2('Naruto')
# print("Latent Feature-Based Recommendations (V2 - Sentence Embeddings, Format, Deeper Model):")
# print(latent_recommendations_v2)