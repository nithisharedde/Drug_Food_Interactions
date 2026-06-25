import streamlit as st
import pandas as pd
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ---------------- LOGIN FUNCTION ----------------
def login():
    st.title("🔐 Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state.logged_in = True
            st.success("Login Successful ✅")
            st.rerun()
        else:
            st.error("Invalid Username or Password ❌")

# ---------------- LOAD APP ----------------
def main_app():

    # Load models
    interaction_model = joblib.load("interaction_model.pkl")
    taste_model = joblib.load("taste_model.pkl")

    # Load data
    drug_data = pd.read_csv("drug_smiles.csv")
    food_data = pd.read_csv("food_smiles.csv")

    # Function
    def smiles_to_fp(smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    # Reason function
    def get_interaction_reason(drug, food):
        drug = drug.lower()
        food = food.lower()

        if "caffeine" in drug and ("coffee" in food or "tea" in food):
            return "Both contain stimulants → may increase heart rate"

        if "warfarin" in drug and ("green" in food):
            return "Vitamin K reduces effectiveness"

        if "grapefruit" in food:
            return "Alters drug metabolism"

        return "Possible chemical interaction"

    # ---------------- SIDEBAR ----------------
    st.sidebar.title("Navigation")

    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()

    page = st.sidebar.radio("Go to", ["🏠 Home", "💊 Predict", "ℹ️ About"])

    # ---------------- HOME ----------------
    if page == "🏠 Home":
        st.image("logo.png", width=150)
        st.title("Drug-Food Interaction System")
        st.write("Predict interactions and taste using ML.")

    # ---------------- PREDICT ----------------
    elif page == "💊 Predict":

        st.title("Prediction")

        drug_name = st.selectbox("Select Drug", drug_data["Drug"])
        food_name = st.selectbox("Select Food", food_data["Food"])

        drug_smiles = drug_data[drug_data["Drug"] == drug_name]["SMILES"].values[0]
        food_smiles = food_data[food_data["Food"] == food_name]["SMILES"].values[0]

        if st.button("Predict"):

            drug_fp = smiles_to_fp(drug_smiles)
            food_fp = smiles_to_fp(food_smiles)

            if drug_fp is None or food_fp is None:
                st.error("Invalid SMILES")
            else:
                combined_fp = np.concatenate((drug_fp, food_fp)).reshape(1, -1)

                interaction_pred = interaction_model.predict(combined_fp)[0]

                # Taste
                st.subheader("Taste")

                drug_taste = taste_model.predict([drug_fp])[0]
                food_taste = taste_model.predict([food_fp])[0]

                st.write(f"Drug Taste: {'Sweet' if drug_taste==1 else 'Bitter'}")
                st.write(f"Food Taste: {'Sweet' if food_taste==1 else 'Bitter'}")

                # Interaction
                st.subheader("Interaction")

                if interaction_pred == 1:
                    st.error("⚠️ Interaction: YES")
                    reason = get_interaction_reason(drug_name, food_name)
                    st.info(reason)
                else:
                    st.success("✅ No Interaction")

    # ---------------- ABOUT ----------------
    elif page == "ℹ️ About":
        st.title("About")
        st.write("ML-based Drug-Food Interaction Predictor.")

# ---------------- APP FLOW ----------------
if not st.session_state.logged_in:
    login()
else:
    main_app()