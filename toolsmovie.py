import pandas as pd 
import streamlit as st 
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt 
import category_encoders as ce
import pandas as pd
import streamlit as st
from datetime import datetime
from PIL import Image
import requests
from io import BytesIO
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import BinaryEncoder
from sklearn.neighbors import NearestNeighbors



#           NETTOYAGE

# Nettoyer les caracteres de la colonne genres
def clean_genre(genre):
    return genre.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")

# Fonction pour convertir la liste nettoyée en chaîne de caractères
def clean_list_to_string(data):
    """ Fonction pour convertir la liste nettoyée en chaîne de caractères et supprimer les chaines de caractères spéciaux"""
    if isinstance(data, list):
        return ', '.join(data)
    return data.replace("[", " ").replace("]", " ").replace("'", " ").replace(" ", " ").replace("{"," ").replace("}", " ")

# Fonction pour replacer les id du cast par leur noms
def clean_character_names(char_list):
    return [char.strip('"') for char in char_list]


def replace_keys_and_clean(characters, id_to_name):
    # Vérifier si characters est un dictionnaire
    if isinstance(characters, dict):
        cleaned_characters = {}
        for person_id, names in characters.items():
            if person_id in id_to_name:
                name = id_to_name[person_id]
                cleaned_characters[name] = clean_character_names(names)
        return cleaned_characters
    else:
        return {}


def replace_directors(directors, nom_cast):
    """ Fonction pour remplacer nm des réalisateurs par leur nom du cast"""
    for director in directors:
        director = director.replace("'","")
        for nm,value in nom_cast.items():
            if director == nm :
                director = value
                return director
            
def clean_dict_to_string(data):
    """ Fonction pour convertir un dictionnaire nettoyée en chaîne de caractères et supprimer les chaines de caractères spéciaux"""
    if isinstance(data, dict):
        data = str(data)
    return data.replace("[", "").replace("]", "").replace("'", " ").replace("{","").replace("}", "")

def clean_production(production_companies_name):
    return production_companies_name.replace("[", "").replace("]", "").replace("'", "").replace(" ", "")

def clean_base_knn_duplicates_list(base,connu) :
    """ Fonction pour retirer les doublons de la liste connu de la base de données"""
    for film in connu:
        base = base[base["tconst"] != film]
    return base



#           RECHERCHE

# définition de la fonction search_min pour forcer le texte de l'utilisateur en minuscule.
def search_min(search_text : str):
    " Arg : str, return str en minuscule "
    search_text.lower()
    result = search_text.lower()
    return result


#           PIPELINE



def create_pipeline(preprocessor):
    knn_model = NearestNeighbors(n_neighbors=7, algorithm="auto")
    pipeline_with_knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', knn_model)
    ])
    return pipeline_with_knn

def create_preprocessor(numeric_features, category_features, boolean_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
        ('scaler', StandardScaler())  # Standard scaling
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Impute missing values with 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encoding
    ])

    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())  # Binary encoding
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, category_features),
            ('bin', binary_transformer, boolean_features),
        ])
    return preprocessor


def fit_pipeline(pipeline_with_knn, movie):
    pipeline_with_knn.fit(movie.drop(columns=['title', 'title_minuscule', 'poster_path', 'tconst']))
    return pipeline_with_knn



#           KNN

def find_nearest_neighbors(movie_title : str, pipeline_with_knn : Pipeline, df : pd.DataFrame) -> pd.DataFrame :
    """Args : movie_title : str : The Product_ID of the query product
              pipeline_with_knn : Pipeline : The pipeline containing the preprocessor and KNN model
              df : pd.DataFrame : The DataFrame containing the product information
       Returns : pd.DataFrame : A DataFrame containing the nearest neighbors of the query product"""

    # Filter the DataFrame to get the features of the specified product
    query_product_features = df[df['tconst'] == movie_title].drop(columns=['title', 'tconst', 'overview_fr','casting', 'poster_path'])

    # Use the pipeline to preprocess the query product features
    query_product_features_processed = pipeline_with_knn.named_steps['preprocessor'].transform(query_product_features)

    # Use the KNN model to find the nearest neighbors for the query product
    nearest_neighbors_indices = pipeline_with_knn.named_steps['knn'].kneighbors(query_product_features_processed)[1][0]

    # Get the nearest neighbors' Product_IDs
    nearest_neighbors_product_ids = df.iloc[nearest_neighbors_indices]['tconst']

    # Create a DataFrame containing the nearest neighbors' information
    nearest_neighbors_df = df[df['tconst'].isin(nearest_neighbors_product_ids)]

    return nearest_neighbors_df[nearest_neighbors_df['tconst'] != movie_title]

def find_nearest_neighbors(movie_id, pipeline_with_knn, df):
    """Trouver les films les plus similaires à un film donné."""
    # Filtrer le DataFrame pour obtenir les caractéristiques du film spécifié
    query_product_features = df[df['tconst'] == movie_id].drop(columns=['title', 'title_minuscule', 'poster_path', 'tconst'])

    # Utiliser le pipeline pour prétraiter les caractéristiques du film spécifié
    query_product_features_processed = pipeline_with_knn.named_steps['preprocessor'].transform(query_product_features)

    # Utiliser le modèle KNN pour trouver les voisins les plus proches
    nearest_neighbors_indices = pipeline_with_knn.named_steps['knn'].kneighbors(query_product_features_processed)[1][0]

    # Obtenir les IDs des voisins les plus proches
    nearest_neighbors_product_ids = df.iloc[nearest_neighbors_indices]['tconst']

    # Créer un DataFrame contenant les informations des voisins les plus proches
    nearest_neighbors_df = df[df['tconst'].isin(nearest_neighbors_product_ids)]

    return nearest_neighbors_df[nearest_neighbors_df['tconst'] != movie_id]



def find_nearest_neighbors_cast(movie_title: str, pipeline_with_knn: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Trouver les films les plus similaires à partir du titre d'un film."""
    # Filtrer le DataFrame pour obtenir les features du film spécifié
    query_product_features = df[df['tconst'] == movie_title].drop(columns=['title', 'tconst', 'overview_fr','casting', 'poster_path'])

    # Utiliser le pipeline pour prétraiter les features du film spécifié
    query_product_features_processed = pipeline_with_knn.named_steps['preprocessor'].transform(query_product_features)

    # Utiliser le modèle KNN pour trouver les films les plus similaires
    nearest_neighbors_indices = pipeline_with_knn.named_steps['knn'].kneighbors(query_product_features_processed)[1][0]

    # Obtenir les IDs des films les plus similaires
    nearest_neighbors_product_ids = df.iloc[nearest_neighbors_indices]['tconst']

    # Créer un DataFrame contenant les informations des films les plus similaires
    nearest_neighbors_df = df[df['tconst'].isin(nearest_neighbors_product_ids)]

    return nearest_neighbors_df[nearest_neighbors_df['tconst'] != movie_title]




#           AFFICHAGE


# Définition de la taille des images à afficher
image_width = 100
image_height = 175
# Fonction pour afficher les images avec titre et note
def display_movie_with_image(title, poster_path):
    if poster_path:
        response = requests.get(poster_path)   # request pour recuperer l'image a partir d'url
        if response.status_code == 200:
            image = Image.open(BytesIO(response.content)) # Si image ok , on converti en format image
            st.image(image, caption=title, width=image_width)










# Fonction pour afficher les détails d'un film
def show_movie_details(movie, movie_id_value, cast):
    # Définir ligne du film défini par l'utilisateur
    movie_details = movie[movie['tconst'] == movie_id_value].iloc[0]
    # Définir dictionnaire nom cast (clé = nconst, valeur = noms du nconst)
    nom_cast = cast.set_index('nconst')['primaryName'].to_dict()
    # Définir variable réalisateur nettoyé
    cleaned_directors = replace_directors(movie_details['directors'], nom_cast)
    # Définir variable casting nettoyé
    cleaned_casting = replace_keys_and_clean(movie_details['casting'], nom_cast)
    cleaned_casting = clean_dict_to_string(cleaned_casting)
    cleaned_genre = clean_list_to_string(movie_details['genres'])
    cleaned_production = clean_list_to_string(movie_details['production_companies_name'])


    # Affichage du titre
    st.title(movie_details['title'])

    # Création de colonnes pour l'affichage de l'image et des détails
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        # Affichage de l'image du film
        st.image(movie_details['poster_path'], use_column_width=True)

    with col2:

        # Affichage du résumé
        st.markdown("<h3 style='text-align: center;'>Résumé 📋</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{movie_details['overview_fr']}</p>", unsafe_allow_html=True)
        # Affichage du casting
        st.markdown("<h3 style='text-align: center;'>Casting 🎭</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{cleaned_casting}</p>", unsafe_allow_html=True)



    with col3:
            # Centrer tout le contenu verticalement
        st.markdown(
        f"<div style='display: flex; justify-content: center; align-items: center; height: 100%;'>"
        # Affichage de la note centrée et plus grosse
        f"<h2 style='text-align: center;'>⭐ {movie_details['averageRating']}/10</h2>"
        f"</div>", 
        unsafe_allow_html=True
    )
        
        st.markdown("<h3 style='text-align: center;'>Genres 🎬</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{cleaned_genre}</p>", unsafe_allow_html=True)

        st.markdown("<h3 style='text-align: center;'>Production 🏢</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{cleaned_production}</p>", unsafe_allow_html=True)

         # Affichage du réalisateur
        st.markdown("<h3 style='text-align: center;'>Réalisateur 🎥</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>{cleaned_directors}</p>", unsafe_allow_html=True)
        

# Afficher details avec movie_id cast
def show_movie_details_cast(movie,movie_id,cast):
    # Définir ligne du film défini par l'utilisateur
    movie_row = movie[movie['tconst'] == movie_id].iloc[0]
    # Définir dictionnaire nom cast (clé = nm, valeur = noms du nm)
    nom_cast = cast.set_index('nconst')['primaryName'].to_dict()
    # Définir variable réalisateur nettoyé
    cleaned_directors = replace_directors(movie_row['directors'],nom_cast)
    # Définir variable casting nettoyé
    cleaned_casting = replace_keys_and_clean(movie_row['casting'], nom_cast)
    cleaned_casting = clean_dict_to_string(cleaned_casting)

    # Affichage du titre
    st.title(movie_row['title'])

    # Création de colonnes pour l'affichage de l'image et des détails 
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        # Affichage de l'image du film
        st.image(movie_row['poster_path'], use_column_width=True) 
    
    with col2:
        # Affichage du réalisateur
        st.markdown("### Réalisateur")
        st.write(cleaned_directors)

        # Affichage du casting
        st.markdown("### Casting")
        st.write(cleaned_casting)
        # Affichage des genres avec la mention "Genres :" centrée
        st.markdown("### Genres")
        st.write(f"{clean_list_to_string(movie_row['genres'])}")
        # Affichage de la production avec la mention "Production :" centrée
        st.markdown("### Production")
        st.write(f"{clean_list_to_string(movie_row['production_companies_name'])}")
        
    with col3:
        # Centrer tout le contenu verticalement
        st.markdown(
            f"<div style='display: flex; flex-direction: column; justify-content: left; align-items: left;'>"
            # Affichage de la note en premier, centrée et encore plus grosse
            f"<h2 style='text-align: left;'>⭐ {movie_row['averageRating']}/10</h2>"
            f"</div>", 
            unsafe_allow_html=True)
        
            
    st.markdown("### Résumé")
    st.write(f"{movie_row['overview_fr']}")
      



def display_recommendations(nearest_neighbors_df, movie, cast):
    """
    Affiche les recommandations sous forme de boutons avec une image et un titre.
    Permet d'afficher les détails d'un film lorsque le bouton est cliqué.

    Args:
        nearest_neighbors_df (DataFrame): DataFrame contenant les recommandations.
        movie (DataFrame): DataFrame contenant les détails des films.
        cast (DataFrame): DataFrame contenant les détails des acteurs.

    """
    # Initialisation de selected_movie_id à None
    st.session_state.selected_movie_id = None

    nombre_col = len(nearest_neighbors_df)
    cols = st.columns(nombre_col)

    # Afficher les résultats dans chaque colonne en gérant les erreurs d'index
    for i, (url, title, tconst) in enumerate(zip(nearest_neighbors_df['poster_path'][:nombre_col], nearest_neighbors_df['title'][:nombre_col], nearest_neighbors_df['tconst'][:nombre_col])):
        with cols[i % nombre_col]:
            if url:
                # Afficher l'image du film
                st.image(url, width=200, caption=title)
                # Créer un bouton pour chaque film avec image et titre
                if st.button(f"Voir les détails", key=f"{title}_{i}_button"):
                    # Stocker l'ID du film dans la session
                    st.session_state.selected_movie_id = tconst

    # Vérifier si un film a été sélectionné pour afficher les détails
    if 'selected_movie_id' in st.session_state and st.session_state.selected_movie_id is not None:
        # Afficher les détails du film sélectionné sur toute la largeur de la page
        show_movie_details(movie, st.session_state.selected_movie_id, cast)
    
    # Réinitialiser selected_movie_id
    st.session_state.selected_movie_id = None

    
