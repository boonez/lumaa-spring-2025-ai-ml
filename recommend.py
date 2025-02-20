import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import re


def preprocess_input(user_input):
    """Preprocess user input by removing common stop words and extracting names."""
    stop_words = {'i', 'like', 'want', 'prefer', 'looking for', 'set', 'in', 'shows', 'show', 'movie', 'movies', 'the', 'a', 'an', 'of', 'to', 'and', 'is', 'on', 'at', 'with'}
    words = user_input.lower().split()
    filtered_words = [word for word in words if word not in stop_words]
    normalized_words = [normalize_genre(word) for word in filtered_words]
    return ' '.join(normalized_words)



def extract_names(user_input):
    """Extract potential names from user input using regex (assumes capitalized words)."""
    name_pattern = re.compile(r'\b[A-Z][a-z]+\s[A-Z][a-z]+\b')
    return name_pattern.findall(user_input)


def normalize_genre(genre):
    """Normalize genre terms to improve matching with broader detection."""
    genre = genre.lower().strip()
    
    genre_mapping = {
        "romantic": "romance",
        "romance": "romance",
        "sci fi": "sci fi",
        "sci-fi": "sci fi",
        "scifi": "sci fi",
        "science fiction": "sci fi",
        "thriller": "thriller",
        "comedy": "comedy",
        "comedic": "comedy",
        "dramedy": "comedy drama",
        "action": "action",
        "adventure": "adventure",
        "action adventure": "action adventure",
        "horror": "horror",
        "mystery": "mystery",
        "animation": "animation",
        "anime": "animation",
        "historical": "history",
        "history": "history",
        "war": "war",
        "fantasy": "fantasy",
        "superhero": "action fantasy",
        "documentary": "documentary"
    }
    
    for key in genre_mapping:
        if key in genre:
            return genre_mapping[key]
    
    return genre  # Default to the original genre if no match is found




def load_dataset(filepath):
    """Load the IMDB dataset from a CSV file."""
    df = pd.read_csv(filepath)
    df = df[['Series_Title', 'Overview', 'Genre', 'IMDB_Rating', 'Meta_score', 'Director', 'Star1', 'Star2', 'Star3', 'Star4']].fillna('')  # Handle missing values
    
    # Convert ratings and scores to numeric values
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce').fillna(0)
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce').fillna(0)
    
    # Normalize genres
    df['Genre'] = df['Genre'].apply(lambda x: ' '.join(set(normalize_genre(g) for g in x.split(','))))

    # Boost metadata importance by repeating key features
    df['combined_text'] = (
        df['Overview'] + ' ' + (df['Genre'] + ' ') * 2 +
        (df['Director'] + ' ') + 
        (df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4'] + ' ')
    )
    
    return df


def filter_by_exact_names(data, names):
    """Filter dataset for exact name matches in cast or director fields."""
    if not names:
        return data  # No filtering needed if no names are provided
    
    mask = data.apply(lambda row: any(name == row['Director'] or name in {row['Star1'], row['Star2'], row['Star3'], row['Star4']} for name in names), axis=1)
    filtered_data = data[mask]
    
    return filtered_data if not filtered_data.empty else data  # Default to full dataset if no match


def compute_similarity(data, user_input, top_n=10):
    """Compute text similarity using TF-IDF and return top N recommendations."""
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), min_df=1, max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(data['combined_text'])
    
    user_input = preprocess_input(user_input)
    user_tfidf = vectorizer.transform([user_input])
    
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    # Normalize scores and apply weighting factor using IMDB rating and Meta score
    normalized_scores = similarity_scores / similarity_scores.max()
    weighted_scores = normalized_scores * ((data['IMDB_Rating'] + (data['Meta_score'] / 10)) / 2)
    
    top_indices = np.argsort(weighted_scores.values)[-top_n:][::-1]  # Ensure it's applied on the correct data structure
    recommendations = data.iloc[top_indices].reset_index(drop=True)  # Reset index to avoid mismatches
    scores = weighted_scores.iloc[top_indices].values  # Ensure correct indexing
    return recommendations, scores


def main(user_input, filepath):
    """Main function to load dataset, compute recommendations, and display results."""
    data = load_dataset(filepath)
    names = extract_names(user_input)
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
