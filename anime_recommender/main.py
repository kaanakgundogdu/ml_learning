import requests
import pandas as pd
import time

url = 'https://graphql.anilist.co'
query = '''
query ($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      total
      currentPage
      lastPage
      hasNextPage
      perPage
    }
    media(type: ANIME) {
      id
      title {
        romaji
        english
      }
      genres
      description
      averageScore
      format
    }
  }
}
'''

data_list = []
page = 1
perPage = 50
hasNextPage = True

while hasNextPage:
    variables = {"page": page, "perPage": perPage}
    try:
        response = requests.post(url, json={'query': query, 'variables': variables})
        
        if response.status_code == 429:
            print(f"Rate limit hit on page {page}. Waiting for 10 seconds before retrying...")
            time.sleep(10)
            continue
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}")
            break
        
        result = response.json()
        pageInfo = result['data']['Page']['pageInfo']
        media_list = result['data']['Page']['media']
        
        for media in media_list:
            entry = {
                "id": media['id'],
                "title_romaji": media['title']['romaji'],
                "title_english": media['title']['english'],
                "genres": ", ".join(media['genres']) if media['genres'] else None,
                "description": media['description'],
                "averageScore": media['averageScore'],
                "format": media['format']
            }
            data_list.append(entry)
        
        print(f"Fetched page {page} of {pageInfo['lastPage']}")
        hasNextPage = pageInfo['hasNextPage']
        page += 1
        time.sleep(2)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        break

df = pd.DataFrame(data_list)
csv_filename = 'anime_info.csv'
df.to_csv(csv_filename, index=False)
print(f"Data saved to {csv_filename}!")
