"""
Interface Streamlit pour l'agent CSV
"""
import streamlit as st
import pandas as pd
import os
import base64
import re
from csv_agent import CSVAgent
from plot_registry import get_plot

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

    # Information sur le LLM utilisé
    st.info("""
    **LLM utilisé :**
    - Ollama (local) si disponible
    - Sinon Gemini (nécessite une clé API)
    """)
    
    # Vérification de la clé API (optionnelle si Ollama est disponible)
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        st.warning("⚠️ Clé API Google Gemini non configurée")
        st.caption("Si Ollama n'est pas disponible, une clé API sera nécessaire")
        api_key = st.text_input(
            "Entrez votre clé API Google Gemini (optionnel si Ollama est installé):",
            type="password",
            help="Obtenez votre clé sur https://makersuite.google.com/app/apikey"
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
    else:
        st.success("✅ Clé API configurée (fallback Gemini)")
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
    verbose = st.checkbox("Mode verbeux (afficher le raisonnement)", value=True)

    # Compteur LLM
    if st.session_state.get("agent") is not None:
        try:
            st.metric("Appels LLM (session)", st.session_state.agent.get_llm_iterations())
        except Exception:
            st.caption("Compteur LLM non disponible.")

# Initialisation de la session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'csv_uploaded' not in st.session_state:
    st.session_state.csv_uploaded = False
if 'current_file_name' not in st.session_state:
    st.session_state.current_file_name = None
if 'current_file_hash' not in st.session_state:
    st.session_state.current_file_hash = None
if 'llm_iterations' not in st.session_state:
    st.session_state.llm_iterations = 0


def render_agent_answer(answer: str):
    """Affiche la réponse et rend l'image si un payload base64 est présent."""
    # Cas 0 : plot_id in-memory (Plotly)
    pattern_plot_id = re.compile(
        r"(?:__)?PLOT_ID_START(?:__)?\s*(.*?)\s*(?:__)?PLOT_ID_END(?:__)?",
        re.IGNORECASE | re.DOTALL,
    )
    plot_matches = list(pattern_plot_id.finditer(answer))
    if plot_matches:
        last_idx = 0
        for m in plot_matches:
            prefix = answer[last_idx:m.start()].strip()
            if prefix:
                st.write(prefix)
            plot_id = m.group(1).strip()
            artifact = get_plot(plot_id)
            if artifact and artifact.figure is not None:
                try:
                    # Extraction des données de la figure Plotly
                    if artifact.figure.data:
                        trace = artifact.figure.data[0]
                        
                        # Préparation du DataFrame
                        data_dict = {"x": trace.x, "y": trace.y}
                        if hasattr(trace, "marker") and trace.marker and "color" in trace.marker:
                             # Si on a des couleurs (ex: scatter avec hue), on pourrait essayer de les gérer
                             # Mais pour l'instant restons simple
                             pass
                             
                        df_native = pd.DataFrame(data_dict)
                        
                        # Gestion selon le type de graphique
                        kind = artifact.kind.lower()
                        
                        # Conversion de l'axe X en datetime si possible pour un meilleur rendu
                        if "x" in df_native.columns:
                            try:
                                # On essaie de convertir en datetime pour que Streamlit gère l'axe temporel
                                df_native["x"] = pd.to_datetime(df_native["x"])
                            except:
                                pass
                        
                        if kind == "line":
                            if "x" in df_native.columns:
                                df_native = df_native.set_index("x")
                            st.line_chart(df_native)
                            
                        elif kind == "bar":
                            if "x" in df_native.columns:
                                df_native = df_native.set_index("x")
                            st.bar_chart(df_native)
                            
                        elif kind == "scatter":
                            if hasattr(st, "scatter_chart"):
                                st.scatter_chart(df_native, x="x", y="y")
                            else:
                                # Fallback sur Altair pour scatter si scatter_chart n'existe pas
                                st.vega_lite_chart(df_native, {
                                    'mark': {'type': 'circle', 'tooltip': True},
                                    'encoding': {
                                        'x': {'field': 'x', 'type': 'quantitative' if not pd.api.types.is_datetime64_any_dtype(df_native['x']) else 'temporal'},
                                        'y': {'field': 'y', 'type': 'quantitative'},
                                    },
                                }, use_container_width=True)
                                
                        elif kind == "hist":
                            if "x" in df_native.columns and "y" in df_native.columns:
                                df_native.columns = ["Plage", "Fréquence"]
                                df_native = df_native.set_index("Plage")
                                st.bar_chart(df_native)
                            else:
                                st.warning("Données d'histogramme mal formatées")
                            
                        elif kind == "corr_heatmap":
                            st.write("**Matrice de corrélation**")
                            if hasattr(trace, 'z'):
                                corr_data = trace.z
                                if hasattr(trace, 'x') and hasattr(trace, 'y'):
                                    corr_df = pd.DataFrame(corr_data, index=trace.y, columns=trace.x)
                                else:
                                    corr_df = pd.DataFrame(corr_data)
                                st.dataframe(corr_df.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1).format("{:.2f}"))
                            else:
                                st.info("Impossible d'extraire la matrice de corrélation.")
                            
                        else:
                            st.info(f"Type de graphique '{kind}' non supporté en mode natif.")
                            
                    else:
                        st.info("Pas de données extractibles pour l'affichage.")
                except Exception as e:
                    st.warning(f"Erreur d'affichage : {e}")
            else:
                st.warning("⚠️ Impossible d'afficher le graphique (plot introuvable en mémoire).")
            last_idx = m.end()
        suffix = answer[last_idx:].strip()
        if suffix:
            # On cache les blocs summary JSON dans l'UI (ils servent à l'agent de commentaire)
            suffix = re.sub(r"(?:__)?PLOT_SUMMARY_START(?:__)?[\s\S]*?(?:__)?PLOT_SUMMARY_END(?:__)?", "", suffix, flags=re.IGNORECASE).strip()
            if suffix:
                st.write(suffix)

        return

    # Cas A : chemin de fichier renvoyé par DataViz (recommandé)
    pattern_file = re.compile(
        r"(?:__)?PLOT_FILE_START(?:__)?\s*(.*?)\s*(?:__)?PLOT_FILE_END(?:__)?",
        re.IGNORECASE | re.DOTALL,
    )
    file_matches = list(pattern_file.finditer(answer))
    if file_matches:
        last_idx = 0
        for m in file_matches:
            prefix = answer[last_idx:m.start()].strip()
            if prefix:
                st.write(prefix)
            path = m.group(1).strip()
            try:
                st.image(path, use_container_width=True)
            except Exception:
                st.warning("⚠️ Impossible d'afficher le graphique (fichier introuvable ou non lisible).")
            last_idx = m.end()
        suffix = answer[last_idx:].strip()
        if suffix:
            st.write(suffix)
        return

    # Cas 1 : bloc complet START...END
    pattern_full = re.compile(
        r"(?:__)?PLOT_BASE64_START(?:__)?\s*(.*?)\s*(?:__)?PLOT_BASE64_END(?:__)?",
        re.IGNORECASE | re.DOTALL,
    )
    matches = list(pattern_full.finditer(answer))

    # Cas 2 : START sans END (on prend jusqu'à la fin)
    if not matches:
        pattern_start_only = re.compile(
            r"(?:__)?PLOT_BASE64_START(?:__)?\s*(.*)",
            re.IGNORECASE | re.DOTALL,
        )
        matches = list(pattern_start_only.finditer(answer))

    if not matches:
        st.write(answer)
        return

    last_idx = 0
    for m in matches:
        # Texte avant le bloc
        prefix = answer[last_idx:m.start()].strip()
        if prefix:
            st.write(prefix)

        payload = m.group(1).strip()
        # Retirer espaces/retours multiples éventuels
        payload_clean = "".join(payload.split())
        try:
            img_bytes = base64.b64decode(payload_clean)
            st.image(img_bytes, use_container_width=True)
        except Exception:
            st.warning("⚠️ Impossible d'afficher le graphique (payload invalide).")
        last_idx = m.end()

    # Texte après le dernier bloc
    suffix = answer[last_idx:].strip()
    if suffix:
        st.write(suffix)

# Si un fichier de données est uploadé via la sidebar
if data_file is not None:
    # Sauvegarder temporairement le fichier
    import hashlib

    file_bytes = data_file.getvalue()
    temp_csv_path = f"temp_{data_file.name}"
    with open(temp_csv_path, "wb") as f:
        f.write(file_bytes)

    file_hash = hashlib.md5(file_bytes).hexdigest()
    file_changed = (
        st.session_state.agent is None
        or st.session_state.current_file_name != data_file.name
        or st.session_state.current_file_hash != file_hash
    )

    # Créer ou recréer l'agent si le fichier change
    if file_changed:
        try:
            with st.spinner("🔧 Initialisation de l'agent..."):
                st.session_state.agent = CSVAgent(
                    temp_csv_path,
                    api_key=api_key if api_key else None,
                    verbose=verbose
                )
                st.session_state.csv_uploaded = True
                st.session_state.chat_history = []
                st.session_state.current_file_name = data_file.name
                st.session_state.current_file_hash = file_hash
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
                render_agent_answer(answer)
    
    # Exemples de questions
    with st.expander("💡 Exemples de questions"):
        st.markdown("""
        - Quelle est la structure du fichier ?
        - Affiche-moi les 10 premières lignes
        - Quelles sont les statistiques pour la colonne X ?
        - Y a-t-il des valeurs manquantes ?
        - Quelle est la corrélation entre les colonnes A et B ?
        - Donne-moi la matrice de corrélation complète
        - Combien de lignes et de colonnes contient le fichier ?
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
                    # Afficher la réponse (texte + éventuel graphique)
                    render_agent_answer(answer)
                    
                    # Ajouter à l'historique
                    st.session_state.chat_history.append((question, answer))
                    # Mettre à jour le compteur LLM
                    try:
                        st.session_state.llm_iterations = st.session_state.agent.get_llm_iterations()
                    except Exception:
                        pass
                except Exception as e:
                    error_msg = f"❌ Erreur : {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append((question, error_msg))
    
    # Bouton pour effacer l'historique
    if st.session_state.chat_history:
        if st.button("🗑️ Effacer l'historique"):
            st.session_state.chat_history = []
            # Réinitialiser l'agent pour effacer sa mémoire interne
            try:
                with st.spinner("🔄 Réinitialisation de l'agent..."):
                    st.session_state.agent = CSVAgent(
                        temp_csv_path,
                        api_key=api_key if api_key else None,
                        verbose=verbose
                    )
                st.success("✅ Historique effacé et agent réinitialisé !")
            except Exception as e:
                st.error(f"❌ Erreur lors de la réinitialisation : {str(e)}")
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
    Développé avec ❤️ en utilisant Streamlit, Ollama et Google Gemini
</div>
""", unsafe_allow_html=True)

