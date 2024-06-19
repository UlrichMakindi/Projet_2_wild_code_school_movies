import pandas as pd 
import streamlit as st 
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt 
import category_encoders as ce
import requests
from PIL import Image
from io import BytesIO
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from category_encoders import BinaryEncoder
from sklearn.neighbors import NearestNeighbors
from toolsmovie import *
from fuzzywuzzy import process
import requests



#           IMPORTATION ET INITIALISATION


# Importation de la base de données
movie = pd.read_pickle('DataSet\df_movieOK.pkl')
cast = pd.read_pickle('DataSet\df_castOK.pkl')




# Initialisation des variables d'état de session pour la navigation
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'movie_id' not in st.session_state:
    st.session_state.movie_id = None
if 'selected_recommendation' not in st.session_state:
    st.session_state.selected_recommendation = None
if 'selected_search_results' not in st.session_state:
    st.session_state.selected_search_results = {}






#           MISE EN PAGE ET FILTRE





# Changer le nom et l'icone de l'onglet
st.set_page_config(
        page_title="Recommandation de films",
        page_icon="📽️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

# Creer un bouton qui retourne a la page principal
col1, col2, col3 = st.columns([1, 2, 1])  # Création de colonnes pour centrer
with col3:
    if st.button("🏠 Accueil"):
        st.experimental_rerun()

# Utilisation HTML CSS pour centrer le titre
st.markdown(
    """
        <div style="text-align: center; font-size: 54px;"><strong>Ciné Plus</strong></div>

    """,
    unsafe_allow_html=True
)

# Suppprimer l'icone fullscreen en la masquant
# Utilisation de HTML pour cibler le bouton, CSS pour le rendre invisible
# unsafe_allow_html=True est l'autorisation pour streamlit de lire du code html
st.markdown("""
    <style>
        button[title="View fullscreen"] {
            display: none !important;
        }
    </style>
""", unsafe_allow_html=True)

# Ajouter des filtres interactifs pour startYear et averageRating
start_year_filter = st.slider("Filtrer par année de début :", min_value=1920, max_value=2024, value=(1920, 2024), step=1)
average_rating_filter = st.slider("Filtrer par note moyenne :", min_value=6.0, max_value=10.0, value=(6.0, 10.0), step=0.1)

# Appliquer les filtres
movie = movie[
    (movie['startYear'] >= start_year_filter[0]) & (movie['startYear'] <= start_year_filter[1]) &
    (movie['averageRating'] >= average_rating_filter[0]) & (movie['averageRating'] <= average_rating_filter[1])
]





#           BARRE DE RECHERCHE



# Assurez-vous que la colonne 'title_minuscule' est en minuscules
movie['title_minuscule'] = movie['title'].str.lower()

# Affichage de l'intitulé de l'application
st.header('Recommandations par film :')

# Définition de la variable texte à rechercher "user_text"
user_text = st.selectbox('', movie['title_minuscule'], index=None, placeholder='Select a Movie Here !')

# Gestion de la barre de recherche vide, des entrées inexistantes, des erreurs de frappe de l'utilisateur.
if user_text == "" or user_text is None:
    st.write('No movie selected')
else:
    # Utilisation de fuzzywuzzy pour trouver la meilleure correspondance
    best_match = process.extractOne(user_text, movie['title_minuscule'])

    if best_match[1] < 90:
        st.write('Pas de résultat trouvé')
    else:
        search_result = movie[movie['title_minuscule'] == best_match[0]]
        st.write('Meilleure correspondance :', best_match[0])




   
        #           AFFICHAGE RECHERCHE


    if not search_result.empty:
        # Créer une colonne pour afficher les résultats de la recherche
        col1, col2, col3 = st.columns(3)

        # Afficher les trois premiers résultats de recherche sur trois colonnes
        for i, (url, title, tconst) in enumerate(zip(search_result['poster_path'][:3], search_result['title'][:3], search_result['tconst'][:3])):
            if i % 3 == 0:
                with col1:
                    if url:
                        # Afficher l'image du film
                        st.image(url, width=200, caption=title)

                        # Créer un bouton pour chaque film avec image et titre
                        if st.button(f"Voir les détails", key=f"{tconst}_{i}_button"):
                            # Stocker les détails du film sélectionné à partir de la recherche dans st.session_state
                            st.session_state.selected_search_results = {'tconst': tconst, 'title': title}

            elif i % 3 == 1:
                with col2:
                    if url:
                        # Afficher l'image du film
                        st.image(url, width=200, caption=title)

                        # Créer un bouton pour chaque film avec image et titre
                        if st.button(f"Voir les détails", key=f"{title}_{i}_button"):
                            # Stocker les détails du film sélectionné à partir de la recherche dans st.session_state
                            st.session_state.selected_search_results = {'tconst': tconst, 'title': title}

            else:
                with col3:
                    if url:
                        # Afficher l'image du film
                        st.image(url, width=200, caption=title)

                        # Créer un bouton pour chaque film avec image et titre
                        if st.button(f"Voir les détails", key=f"{title}_{i}_button"):
                            # Stocker les détails du film sélectionné à partir de la recherche dans st.session_state
                            st.session_state.selected_search_results = {'tconst': tconst, 'title': title}


            # Afficher les détails du film sélectionné sur toute la largeur de la page
            selected_search_result = st.session_state['selected_search_results']
            if selected_search_result:
                st.write('Meilleure correspondance :', selected_search_result['title'])
                show_movie_details(movie, selected_search_result['tconst'], cast)

            # Réinitialiser selected_search_results à la fin de la boucle
            st.session_state.selected_search_results = {}



            #           PIPELINE ET KNN

            numeric_features = ['averageRating', 'numVotes', 'runtimeMinutes', 'revenue', 'startYear']
            category_features = ['production_countries', 'production_companies_name', 'genres']
            boolean_features = ['isAdult']

            preprocessor = create_preprocessor(numeric_features, category_features, boolean_features)

            pipeline_with_knn = create_pipeline(preprocessor)

            pipeline_with_knn = fit_pipeline(pipeline_with_knn, movie)

            # Filtre pour utilisation de la recommandation avec le 'tconst'

            query_product_id = search_result['tconst'].iloc[0]

            nearest_neighbors_df = find_nearest_neighbors(query_product_id, pipeline_with_knn, movie)

            # Vérifier si un film est sélectionné à partir des recommandations
            if st.session_state.selected_recommendation:
                # Afficher les détails du film sélectionné à partir des recommandations
                show_movie_details(movie, st.session_state.selected_recommendation, cast)



            st.subheader("Affichage des recommandations")

            display_recommendations(nearest_neighbors_df, movie, cast)




#           AFFICHAGE DU TOP 10 SUR LA PAGE


movie['genres'] = movie['genres'].apply(clean_genre)


# Séparation des genres en une liste et explosion de la colonne
df_exploded = movie.assign(genres=movie['genres'].str.split(',')).explode('genres')

# Compter le nombre de films par genre
genres_count = df_exploded['genres'].value_counts()

# Sélectionner les 10 genres les plus populaires
top_genres = genres_count.head(10).index

# Filtrer les films pour inclure uniquement les 10 genres les plus populaires
filtered_movies = df_exploded[df_exploded['genres'].isin(top_genres)]

# Groupement des films par genre parmi les 10 genres les plus populaires
grouped_movies = filtered_movies.groupby('genres')

# Compter le nombre de films par genre
genres_count = df_exploded['genres'].value_counts()

# Sélectionner les 10 genres les plus populaires
top_genres = genres_count.head(10).index

# Filtrer les films pour inclure uniquement les 10 genres les plus populaires
filtered_movies = df_exploded[df_exploded['genres'].isin(top_genres)]

# Groupement des films par genre parmi les 10 genres les plus populaires
grouped_movies = filtered_movies.groupby('genres')


if st.session_state.page == 'details' and st.session_state.movie_id:
    show_movie_details(movie, st.session_state.movie_id, cast)
    if st.button("Retour", key="return_button_details_page"):
        st.session_state.page = 'main'
        st.session_state.movie_id = None
        st.rerun()
else:
    # Afficher les 10 films avec le meilleur averageRating pour chaque genre parmi les 10 genres les plus populaires
    for genre, genre_movies in grouped_movies:
        st.subheader(genre.strip())
        top_genre_movies = genre_movies.nlargest(10, 'averageRating')

        # Créer une ligne pour afficher les images des films du même genre
        num_cols = 10
        col_index = 0
        cols = st.columns(num_cols)
        for idx, row in enumerate(top_genre_movies.itertuples()):
            with cols[col_index]:
                st.image(row.poster_path, use_column_width=True)
                if st.button(row.title, key=f"{genre}_{row.tconst}_{idx}"):
                    st.session_state.page = 'details'
                    st.session_state.movie_id = row.tconst
                    st.rerun()
                col_index = (col_index + 1) % num_cols
