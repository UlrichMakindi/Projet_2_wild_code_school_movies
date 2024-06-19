import pandas as pd
import streamlit as st
import requests
import numpy as np
import requests
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from category_encoders import BinaryEncoder
from sklearn.neighbors import NearestNeighbors
from toolsmovie import clean_list_to_string, clean_character_names, replace_keys_and_clean, replace_directors, clean_list_to_string, clean_genre, clean_dict_to_string
from toolsmovie import clean_base_knn_duplicates_list, show_movie_details_cast, find_nearest_neighbors_cast


#           IMPORTATION ET INITIALISATION


movie = pd.read_pickle('DataSet\df_movieOK.pkl')
cast = pd.read_pickle('DataSet\df_castOK.pkl')


# Initialisation de l'état de la session pour la navigation
if 'page' not in st.session_state:
    st.session_state.page = 'main'
if 'movie_id' not in st.session_state:
    st.session_state.movie_id = None
if 'selected_cast_box' not in st.session_state:
    st.session_state.selected_cast_box = {} 



#           MISE EN PAGE GENERALE



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





# Affichage du titre

# Utilisation HTML CSS pour centrer le titre
st.markdown(
    """
        <div style="text-align: center; font-size: 54px;"><strong>Ciné Plus</strong></div>

    """,
    unsafe_allow_html=True)



#           BARRE DE RECHERCHE


st.sidebar.title("Moteurs de recherche")

# Utiliser st.selectbox pour créer une liste déroulante de recherche

cast.primaryName = cast.primaryName.map(lambda x: x.replace("'", ""))

# Ajouter une option de placeholder au début de la liste
options = ["Insérer et sélectionner un réalisateur, une actrice ou un acteur :"] + cast.primaryName.tolist()

cast_box= st.selectbox('', options)

# Vérifier si un acteur/réalisateur a été sélectionné
if cast_box != "Insérer et sélectionner un réalisateur, une actrice ou un acteur :":
    # Trouver le bon ID du casting
    nconst = cast[cast.primaryName == cast_box]['nconst'].iloc[0]
    st.session_state.selected_cast_box= {}






#           APPEL DE L API




    url = f"https://api.themoviedb.org/3/find/{nconst}?external_source=imdb_id"

    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI1YTcxMmNhMDc4ZDMxM2U0N2IyNmEwMWVjN2NiZDc1NiIsInN1YiI6IjY2NDM2ZTgzYWI2MzYwNWZiNDc3ZTk5NSIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.cLkXfzeu9gn5gkGHabpvCidukG34vJc6Kjan3kXRdQ0"
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        if "person_results" in data and data["person_results"]:
            profile_path = data["person_results"][0].get("profile_path")
    




        

#           MISE EN PAGE ACTEUR



    # Définir les variables pour l'affichage
    titre = st.title(cast_box)
    age = cast[cast['nconst'] == nconst]['age'].iloc[0]
    naissance = cast[cast['nconst'] == nconst]['birthYear'].iloc[0]
    deces = cast[cast['nconst'] == nconst]['deathYear'].iloc[0]
    alive = cast[cast['nconst'] == nconst]['alive'].iloc[0]
    primaryProfession = cast[cast['nconst'] == nconst]['primaryProfession'].iloc[0]
    connu = cast[cast['nconst'] == nconst]['knownTitles'].iloc[0]
    note = cast[cast['nconst'] == nconst]['average_rating_actors'].iloc[0]
    note = np.mean([int(x) for x in note])





    #           AFFICHAGE PAGE



    # Utiliser des colonnes pour la mise en page
    col1, col2 = st.columns(2)

    # Afficher l'image dans la première colonne
    with col1:
        # Essayer de définir le chemin de l'image du profil
        try:
            if profile_path is None:
                sans_photo = "image/inconnu.png"  # Utiliser l'image par défaut si profile_path dans api est nulle
                st.image(sans_photo)
            else :
                st.image("https://image.tmdb.org/t/p/original/" + profile_path)
        except (NameError):
            sans_photo = "image/inconnu.png"  # Utiliser l'image par défaut en cas d'erreur
            st.image(sans_photo)

    # Afficher le texte dans la deuxième colonne
    with col2:
        # Afficher les toutes les informations du cast en ce qui concerne l'âge
        if cast[cast['nconst'] == nconst]['alive'].iloc[0] == False :
            # Afficher la naissance
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>🍼 Née en {naissance}</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher le décès
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>👴 Décédé en {deces} ({age} ans)</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher le métier
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>🧑‍💻 Métier(s) connu(s) : {clean_list_to_string(primaryProfession)}</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher la note moyenne
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>⭐ {note}/10</h2>"
                f"</div>", 
                unsafe_allow_html=True)
        else:
            # Afficher l'âge
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>🎂 Age : {age} ans</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher la naissance
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: left; align-items: left;'>"
                f"<h2 style='text-align: center;'>🐣 Née en {naissance}</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher le métier
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>👩‍🏫 Métier(s) connu(s) : {clean_list_to_string(primaryProfession)}</h2>"
                f"</div>", 
                unsafe_allow_html=True)
            # Afficher la note moyenne
            st.markdown(
                f"<div style='display: flex; flex-direction: column; justify-content: center; align-items: center;'>"
                f"<h2 style='text-align: center;'>⭐ {note}/10</h2>"
                f"</div>", 
                unsafe_allow_html=True)



    #           AFFICHAGE FILM CONNU DU CAST RECHERCHE



    # Affichage des films connus

    if st.session_state.page == 'main':
        st.header("Film(s) Connu(s)")
        nb_colonnes = len(connu)
        col = st.columns(nb_colonnes)  # Crée les colonnes
        for index, film_id in enumerate(connu):
            with col[index]:  # Utilise chaque colonne pour afficher un film
                movie_row = movie[movie['tconst'] == film_id].iloc[0]
                st.image(movie_row["poster_path"], use_column_width=True)
                if st.button(movie_row["title"], key=film_id):  # Création du bouton servant de titre du film
                    st.session_state.page = film_id
                    

    # Affichage des détails du film sélectionné
    else:
        show_movie_details_cast(movie,st.session_state.page,cast)
        if st.button("Retour"):
            st.session_state.page = 'main'
            st.session_state.movie_id = None
            st.rerun()
            



    #           AFFICHAGE FILMS RECOMMANDES DU CAST RECHERCHE



    # Charger les données

    # Définir les features
    numeric_features = ['averageRating', 'numVotes', 'runtimeMinutes', 'revenue', 'startYear']
    category_features = ['production_countries', 'production_companies_name', 'genres']
    boolean_features = ['isAdult']

    # Définir les étapes de prétraitement pour chaque type de feature
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),  # Imputer les valeurs manquantes avec la médiane
        ('scaler', StandardScaler())  # Standardisation
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Imputer les valeurs manquantes avec 'missing'
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Encodage One-Hot
    ])

    binary_transformer = Pipeline(steps=[
        ('binary', BinaryEncoder())  # Encodage binaire
    ])

    # Combiner les étapes de prétraitement en utilisant ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, category_features),
            ('bin', binary_transformer, boolean_features),
        ])

    # Définir le modèle KNN
    knn_model = NearestNeighbors(n_neighbors=7, algorithm="auto")

    # Combiner le prétraitement et le modèle KNN en utilisant Pipeline
    pipeline_with_knn = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('knn', knn_model)
    ])

    # Prétraiter les données
    X = movie.drop(columns=['title', 'tconst', 'overview_fr','casting'])
    X_preprocessed = preprocessor.fit_transform(X)

    # Entraîner le modèle KNN
    knn_model.fit(X_preprocessed)





    #           MISE EN PAGE DES RECOMMANDATIONS





    # Affichage des films recommandés
    if st.session_state.page == 'main':
        st.header("Films recommandés")

        # Définir nearest_neighboors_df
        nearest_neighbors_df = find_nearest_neighbors_cast(connu[0], pipeline_with_knn, movie)
        nearest_neighbors_df = clean_base_knn_duplicates_list(nearest_neighbors_df,connu)
        st.subheader("Affichage des recommandations")
        nb_colonnes_recomm = len(nearest_neighbors_df)
        col = st.columns(nb_colonnes_recomm)  # Crée les colonnes
        for index, row in enumerate(nearest_neighbors_df.itertuples()):
            with col[index % 6]:  # Utilise chaque colonne pour afficher un film
                st.image(row.poster_path, use_column_width=True)
                if st.button(row.title, key=row.tconst):  # Création du bouton servant de titre du film
                    st.session_state.page = 'details'
                    st.session_state.page = row.tconst


    # Affichage des détails du film sélectionné
    if st.session_state.page == 'details'and st.session_state.movie_id:
        show_movie_details_cast(st.session_state.movie_id,cast)
        if st.button("Retour"):
            st.session_state.page = 'main'
            st.session_state.movie_id = None
    # Réinitialiser selected_search_results à la fin de la boucle
    st.session_state.selected_cast_box = None


        
