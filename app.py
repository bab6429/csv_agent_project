"""
Interface Streamlit pour l'agent CSV
"""
import streamlit as st
import pandas as pd
import os
from csv_agent import CSVAgent
from config import Config

# Configuration de la page
st.set_page_config(
    page_title="Agent CSV - Analyse de données IA",
    page_icon="📊",
    layout="wide"
)

# Titre et description
st.title("📊 Agent CSV - Analyse de données avec IA")
st.markdown("""
Uploadez votre fichier CSV et posez des questions en langage naturel !
L'agent IA analysera vos données et répondra à vos questions.
""")

""" Barre latérale: configuration + uploads (données et PDF) """
with st.sidebar:
    st.header("⚙️ Configuration")

    # Vérification de la clé API
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.error("⚠️ Clé API Google Gemini manquante")
        api_key = st.text_input(
            "Entrez votre clé API Google Gemini:",
            type="password",
            help="Obtenez votre clé sur https://makersuite.google.com/app/apikey"
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("✅ Clé API configurée")
        key_preview = api_key[:10] + "..." + api_key[-4:]
        st.text(f"Clé : {key_preview}")

    st.divider()

    # Upload des données tabulaires (CSV/Excel) pour l'agent
    st.subheader("📁 Données (CSV/Excel)")
    data_file = st.file_uploader(
        "Choisissez un fichier de données",
        type=["csv", "xlsx", "xls"],
        help="Uploadez le fichier que l'agent analysera"
    )

    # Upload PDF côté interface (stocké pour futur usage, non analysé par l'agent ici)
    st.subheader("📄 Document (PDF)")
    pdf_file = st.file_uploader(
        "Optionnel: ajouter un PDF",
        type=["pdf"],
        help="Le PDF est conservé pour référence visuelle; l'agent n'en fait pas l'analyse pour l'instant"
    )
    if pdf_file is not None:
        st.session_state["uploaded_pdf_name"] = pdf_file.name
        st.caption(f"PDF chargé: {pdf_file.name}")

    st.divider()

    # Options de l'agent
    st.subheader("Options de l'agent")
    verbose = st.checkbox("Mode verbeux (afficher le raisonnement)", value=False)

# Initialisation de la session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'csv_uploaded' not in st.session_state:
    st.session_state.csv_uploaded = False

# Si un fichier de données est uploadé via la sidebar
if data_file is not None:
    # Sauvegarder temporairement le fichier
    temp_csv_path = f"temp_{data_file.name}"
    with open(temp_csv_path, "wb") as f:
        f.write(data_file.getbuffer())
    
    # Créer l'agent si ce n'est pas déjà fait ou si c'est un nouveau fichier
    if st.session_state.agent is None or not st.session_state.csv_uploaded:
        try:
            with st.spinner("🔧 Initialisation de l'agent..."):
                st.session_state.agent = CSVAgent(
                    temp_csv_path,
                    api_key=api_key if api_key else None,
                    verbose=verbose
                )
                st.session_state.csv_uploaded = True
                st.session_state.chat_history = []
            st.success("✅ Fichier chargé et agent prêt !")
        except Exception as e:
            st.error(f"❌ Erreur lors de l'initialisation : {str(e)}")
            st.stop()
    
    # Interface CHAT UNIQUEMENT (plus d'aperçu/onglets)
    
    # Interface de chat
    st.header("💬 Posez vos questions")
    
    # Afficher l'historique du chat
    chat_container = st.container()
    with chat_container:
        for i, (question, answer) in enumerate(st.session_state.chat_history):
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)
    
    # Exemples de questions
    with st.expander("💡 Exemples de questions"):
        st.markdown("""
        - Quelle est la moyenne de la colonne X ?
        - Combien de lignes ont une valeur > 100 dans la colonne Y ?
        - Quelle est la corrélation entre les colonnes A et B ?
        - Affiche-moi les 10 premières lignes où la colonne Z est égale à "valeur"
        - Quelles sont les statistiques pour la colonne W ?
        - Y a-t-il des valeurs manquantes ?
        - Quelle est la valeur maximale de la colonne V ?
        - Trace l'histogramme de la colonne Age
        - Fais un scatter entre Salaire et Age avec un titre
        - Affiche la courbe des ventes par mois
        """)
    
    # Input pour la question
    question = st.chat_input("Posez votre question sur les données...")
    
    if question:
        # Afficher la question de l'utilisateur
        with st.chat_message("user"):
            st.write(question)
        
        # Obtenir la réponse de l'agent
        with st.chat_message("assistant"):
            with st.spinner("🤔 L'agent réfléchit..."):
                try:
                    answer = st.session_state.agent.query(question)
                    # Afficher le texte
                    st.write(answer)
                    # Détecter et afficher d'éventuels graphiques
                    if isinstance(answer, str):
                        # Cas base64
                        if "PLOT_B64::" in answer:
                            import base64
                            for line in answer.splitlines():
                                if line.startswith("PLOT_B64::"):
                                    b64_data = line.replace("PLOT_B64::", "").strip()
                                    if b64_data:
                                        st.image(base64.b64decode(b64_data), use_column_width=True)
                        # Cas fichier
                        if "PLOT::" in answer:
                            for line in answer.splitlines():
                                if line.startswith("PLOT::"):
                                    img_path = line.replace("PLOT::", "").strip()
                                    if img_path and os.path.exists(img_path):
                                        st.image(img_path, use_column_width=True)
                    
                    # Ajouter à l'historique
                    st.session_state.chat_history.append((question, answer))
                except Exception as e:
                    error_msg = f"❌ Erreur : {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append((question, error_msg))
    
    # Bouton pour effacer l'historique
    if st.session_state.chat_history:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Nettoyage du fichier temporaire lors de la fermeture (optionnel)
    # Le fichier sera écrasé au prochain upload

else:
    # Instructions si aucun fichier n'est uploadé
    st.info("👈 Uploadez un fichier de données (CSV/Excel) dans la barre latérale pour démarrer le chat d'analyse")
    
    # Afficher un exemple
    with st.expander("📝 Exemple de fichier CSV"):
        example_data = pd.DataFrame({
            'Nom': ['Alice', 'Bob', 'Charlie', 'Diana'],
            'Age': [25, 30, 35, 28],
            'Ville': ['Paris', 'Lyon', 'Marseille', 'Paris'],
            'Salaire': [50000, 60000, 55000, 52000]
        })
        st.dataframe(example_data)
        
        st.markdown("**Exemples de questions que vous pourriez poser :**")
        st.markdown("""
        - Quel est l'âge moyen ?
        - Combien de personnes habitent à Paris ?
        - Quel est le salaire maximum ?
        - Affiche-moi les personnes de plus de 30 ans
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    Développé avec ❤️ en utilisant Streamlit et Google Gemini
</div>
""", unsafe_allow_html=True)

