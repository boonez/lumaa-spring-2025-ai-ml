import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse


def preprocess_input(user_input):
    """Preprocess input by removing stop words and normalizing genre terms."""
    stop_words = {'i', 'like', 'love', 'want', 'prefer', 'looking for', 'set', 
                  'in', 'shows', 'show', 'movie', 'movies', 'the', 'a', 'an', 
                  'of', 'to', 'and', 'is', 'on', 'at', 'with'}
    filtered_words = [word for word in user_input.lower().split() if word not in stop_words]
    return ' '.join(normalize_genre(' '.join(filtered_words)).split())


def normalize_genre(genre):
    """Normalize genre terms for better matching."""
    genre_mapping = {
        "romantic": "romance", "sci-fi": "sci fi", "scifi": "sci fi", "science fiction": "sci fi",
        "comedic": "comedy", "dramedy": "comedy drama",
        "superhero": "action fantasy", "anime": "animation",
        "historical": "history"
    }
    return genre_mapping.get(genre.lower().strip(), genre)  # Return mapped genre or original


def load_dataset(filepath):
    """Load and preprocess IMDB dataset."""
    df = pd.read_csv(filepath)
    df = df[['Series_Title', 'Overview', 'Genre', 'IMDB_Rating', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']].fillna('')
    
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce').fillna(0)
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce').fillna(0)
    df['Genre'] = df['Genre'].apply(lambda x: ' '.join(set(normalize_genre(g) for g in x.split(','))))
    
    df['combined_text'] = (
        df['Overview'] + ' ' + (df['Genre'] + ' ') * 2 +
        (df['Director'] + ' ') + 
        (df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4'] + ' ')
    )
    
    return df


def filter_by_exact_names(data, names):
    """Filter dataset for exact matches in director or star columns."""
    if not names:
        return data  # No filtering if no names are provided

    mask = data['Director'].isin(names) | data[['Star1', 'Star2', 'Star3', 'Star4']].isin(names).any(axis=1)
    filtered_data = data[mask]
    
    return filtered_data if not filtered_data.empty else data


def compute_similarity(data, user_input, top_n=5):
    """Compute text similarity using TF-IDF and return top N recommendations."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=1, max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
    
    user_input = preprocess_input(user_input)
    user_tfidf = vectorizer.transform([user_input])
    
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Normalize and apply weighting factor using IMDB rating and Meta score
    normalized_scores = similarity_scores / similarity_scores.max()
    weighted_scores = normalized_scores * ((data['IMDB_Rating'] + (data['Meta_score'] / 10)) / 2)
    
    top_indices = np.argsort(weighted_scores)[-top_n:][::-1]
    recommendations = data.iloc[top_indices].reset_index(drop=True)
    scores = weighted_scores.iloc[top_indices]
    
    return recommendations, scores


def main(user_input, filepath):
    """Main function to load dataset, compute recommendations, and display results."""
    data = load_dataset(filepath)
    names = user_input.split(',')  # Extract names manually if needed
    filtered_data = filter_by_exact_names(data, names)
    recommendations, scores = compute_similarity(filtered_data, user_input)
    
    print("Top recommendations:")
    for i, (title, score) in enumerate(zip(recommendations['Series_Title'], scores)):
        print(f"{i+1}. {title} (Score: {score:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("user_input", type=str, help="User's preference description")
    parser.add_argument("filepath", type=str, help="Path to IMDB dataset CSV file")
    args = parser.parse_args()
    main(args.user_input, args.filepath)
