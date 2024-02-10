#programme qui recommande des movies avec NLP

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Charger les données des films
df_movies = pd.read_csv("C:/Users/Admin/OneDrive/Pictures/Documents/GitHub/MovieRecommendation/tmdb_5000_movies.csv")
df_credits = pd.read_csv("C:/Users/Admin/OneDrive/Pictures/Documents/GitHub/MovieRecommendation/tmdb_5000_credits.csv")

# Créer un objet TfidfVectorizer pour convertir les aperçus de films en matrices TF-IDF
tfidf = TfidfVectorizer(stop_words='english')

# Remplir les valeurs manquantes dans la colonne 'overview' avec une chaîne vide
df_movies['overview'] = df_movies['overview'].fillna("")

# Transformer les aperçus de films en matrices TF-IDF
tfidf_matrix = tfidf.fit_transform(df_movies['overview'])

# Calculer la similarité cosinus entre les aperçus de films
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Créer un objet Series avec les index des films et les titres des films comme index
indices = pd.Series(df_movies.index, index=df_movies['original_title']).drop_duplicates()

# Définition de la fonction pour obtenir les recommandations de films similaires
def get_recommendations(title, cosine_sim=cosine_sim, indices=indices, df_movies=df_movies):
    # Obtenir l'index correspondant au titre du film donné
    idx = indices[title]
    
    # Obtenir les scores de similarité cosinus pour ce film
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Trier les scores de similarité dans l'ordre décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Ignorer le premier élément, car il s'agit du film lui-même (similarité de 1 avec lui-même)
    sim_scores = sim_scores[1:11]
    
    # Obtenir les index des films similaires
    sim_indices = [i[0] for i in sim_scores]
    
    # Obtenir les titres des films similaires et les afficher
    recommended_movies = df_movies['original_title'].iloc[sim_indices]
    print(recommended_movies)

# Appel de la fonction pour obtenir les recommandations de films similaires à "The Dark Knight Rises"
get_recommendations('The Dark Knight Rises')
