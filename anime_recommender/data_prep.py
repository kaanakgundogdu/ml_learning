import pandas as pd
from bs4 import BeautifulSoup

df = pd.read_csv('anime_info.csv')

print("Initial Data Information:")
print(df.info())
print("\n")

threshold = len(df) * 0.5
df = df.dropna(thresh=threshold, axis=1)

df = df.dropna()

df = df.drop_duplicates()

def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()

df['description'] = df['description'].apply(remove_html_tags)

def standardize_genres(genre_str):
    return [genre.strip().title() for genre in genre_str.split(',')]

df['genres'] = df['genres'].apply(standardize_genres)

df['title_romaji'] = df['title_romaji'].str.title()

print("Cleaned Data Information:")
print(df.info())
print("\n")

df.to_csv('cleaned_anime_info.csv', index=False)
print("Cleaned data saved to 'cleaned_anime_info.csv'")
