import streamlit as st
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import plotly.graph_objects as go
import plotly.express as px

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Attrition Client",
    page_icon="🏦",
    layout="wide"
)

# Initialiser Spark (avec cache pour éviter de recréer à chaque fois)
@st.cache_resource
def init_spark():
    return SparkSession.builder \
        .appName('Attrition_Prediction_App') \
        .master('local[*]') \
        .getOrCreate()

# Charger le modèle
@st.cache_resource
def load_model():
    spark = init_spark()
    model_path = "model/best_rf_model"
    return PipelineModel.load(model_path)

# Titre principal
st.title("🏦 Prédiction de l'Attrition Client Bancaire")
st.markdown("### Application de prédiction en temps réel")
st.markdown("---")

try:
    # Charger le modèle
    with st.spinner("Chargement du modèle..."):
        model = load_model()
        spark = init_spark()

    st.success("✓ Modèle chargé avec succès!")

    # Sidebar pour les inputs
    st.sidebar.header("📝 Informations Client")

    # Inputs utilisateur
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)
    geography = st.sidebar.selectbox("Géographie", ["France", "Spain", "Germany"])
    gender = st.sidebar.selectbox("Genre", ["Male", "Female"])
    age = st.sidebar.slider("Âge", 18, 100, 40)
    tenure = st.sidebar.slider("Ancienneté (années)", 0, 10, 5)
    balance = st.sidebar.number_input("Solde du compte", 0.0, 300000.0, 50000.0)
    num_products = st.sidebar.selectbox("Nombre de produits", [1, 2, 3, 4])
    has_cr_card = st.sidebar.selectbox("Possède une carte de crédit", ["Oui", "Non"])
    is_active_member = st.sidebar.selectbox("Membre actif", ["Oui", "Non"])
    estimated_salary = st.sidebar.number_input("Salaire estimé", 0.0, 200000.0, 100000.0)

    # Convertir les inputs
    geography_map = {"France": 0.0, "Germany": 1.0, "Spain": 2.0}
    gender_map = {"Female": 1.0, "Male": 0.0}

    # Créer le DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [float(credit_score)],
        'Age': [float(age)],
        'Tenure': [float(tenure)],
        'Balance': [float(balance)],
        'NumOfProducts': [float(num_products)],
        'HasCrCard': [1 if has_cr_card == "Oui" else 0],
        'IsActiveMember': [1 if is_active_member == "Oui" else 0],
        'EstimatedSalary': [float(estimated_salary)],
        'Geography_Index': [geography_map[geography]],
        'Gender_Index': [gender_map[gender]]
    })

    # Layout principal
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Données d'entrée")

        # Afficher les données sous forme de cards
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

        with metrics_col1:
            st.metric("Credit Score", credit_score)
            st.metric("Âge", age)
            st.metric("Géographie", geography)

        with metrics_col2:
            st.metric("Solde", f"{balance:,.0f} €")
            st.metric("Ancienneté", f"{tenure} ans")
            st.metric("Genre", gender)

        with metrics_col3:
            st.metric("Salaire estimé", f"{estimated_salary:,.0f} €")
            st.metric("Nb produits", num_products)
            st.metric("Carte crédit", has_cr_card)

    with col2:
        st.subheader("🎯 Prédiction")

        # Bouton de prédiction
        if st.button("🔮 Prédire", type="primary", use_container_width=True):
            with st.spinner("Calcul en cours..."):
                # Convertir en Spark DataFrame
                spark_df = spark.createDataFrame(input_data)

                # Faire la prédiction
                prediction = model.transform(spark_df)

                # Récupérer les résultats
                result = prediction.select("prediction", "probability").collect()[0]
                pred_class = int(result['prediction'])
                probability = result['probability'].toArray()

                # Afficher le résultat
                if pred_class == 1:
                    st.error("⚠️ RISQUE ÉLEVÉ D'ATTRITION")
                    risk_level = "ÉLEVÉ"
                    color = "red"
                else:
                    st.success("✓ RISQUE FAIBLE D'ATTRITION")
                    risk_level = "FAIBLE"
                    color = "green"

                # Probabilité
                prob_churn = probability[1] * 100
                st.metric("Probabilité d'attrition", f"{prob_churn:.1f}%")

                # Jauge de probabilité
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = prob_churn,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Risque d'Attrition"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 80
                        }
                    }
                ))

                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

                # Recommandations
                st.subheader("💡 Recommandations")
                if pred_class == 1:
                    st.warning("""
                    **Actions recommandées:**
                    - Contacter le client de manière proactive
                    - Proposer des offres personnalisées
                    - Améliorer l'engagement client
                    - Analyser les raisons potentielles d'insatisfaction
                    """)
                else:
                    st.info("""
                    **Actions recommandées:**
                    - Maintenir la qualité de service
                    - Continuer l'engagement régulier
                    - Proposer des produits complémentaires
                    """)

    # Section statistiques
    st.markdown("---")
    st.subheader("📈 Statistiques du Modèle")

    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        st.metric("AUC-ROC", "0.85", help="Area Under ROC Curve")
    with stat_col2:
        st.metric("Accuracy", "86%", help="Précision globale")
    with stat_col3:
        st.metric("Precision", "82%", help="Précision des prédictions positives")
    with stat_col4:
        st.metric("Recall", "78%", help="Taux de rappel")

except Exception as e:
    st.error(f"❌ Erreur: {str(e)}")
    st.info("""
    **Assurez-vous que:**
    1. Le modèle a été entraîné et sauvegardé dans `model/best_rf_model`
    2. PySpark est correctement installé
    3. Le notebook d'entraînement a été exécuté
    """)

# Footer
st.markdown("---")
st.markdown("*Application développée avec Streamlit et PySpark MLlib*")
