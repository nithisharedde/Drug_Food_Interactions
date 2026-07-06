import streamlit as st
import pandas as pd
import joblib
import numpy as np
from datetime import datetime

from rdkit import Chem
from rdkit.Chem import AllChem

import auth

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Drug-Food Interaction",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── SESSION STATE ─────────────────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "_nav_idx" not in st.session_state:
    st.session_state._nav_idx = 0
if "theme" not in st.session_state:
    st.session_state.theme = "light"

NAV_PAGES = ["🏠 Home", "💊 Predict", "🕘 History", "ℹ️ About"]

def go(page_name):
    st.session_state._nav_idx = NAV_PAGES.index(page_name)
    st.rerun()

# ── THEME CSS VARIABLES ────────────────────────────────────────────────────────
LIGHT_VARS = """
:root {
    --bg:           #F0F4FA;
    --card:         #FFFFFF;
    --text:         #14213D;
    --text-muted:   #5C6B82;
    --border:       #D8E0EE;
    --input-bg:     #FFFFFF;
    --sidebar-bg:   #14213D;
    --sidebar-text: #FFFFFF;
    --accent-blue:  #1A5EA8;
    --btn-bg:       linear-gradient(135deg, #14213D, #1A5EA8);
    --tip-bg:       #E8F1FB;
    --success-bg:   #E7F7ED; --success-text: #1A5C35;
    --danger-bg:    #FDEBEC; --danger-text:  #8B1A24;
    --warn-bg:      #FFF5E0; --warn-text:    #7A5000;
    --info-bg:      #E8F1FB; --info-text:    #144A66;
}
"""

DARK_VARS = """
:root {
    --bg:           #0D1117;
    --card:         #161B22;
    --text:         #E6EDF3;
    --text-muted:   #8B949E;
    --border:       #30363D;
    --input-bg:     #21262D;
    --sidebar-bg:   #010409;
    --sidebar-text: #E6EDF3;
    --accent-blue:  #58A6FF;
    --btn-bg:       linear-gradient(135deg, #1F6FEB, #E63946);
    --tip-bg:       #1C2D3D;
    --success-bg:   #0D2E1C; --success-text: #56D18A;
    --danger-bg:    #2D0D10; --danger-text:  #FF8A93;
    --warn-bg:      #2E2100; --warn-text:    #FFD166;
    --info-bg:      #0C1E2D; --info-text:    #79C0FF;
}
"""

def load_css():
    theme_vars = DARK_VARS if st.session_state.theme == "dark" else LIGHT_VARS
    try:
        with open("styles.css") as f:
            sheet = f.read()
    except FileNotFoundError:
        sheet = ""
    st.markdown(f"<style>{theme_vars}\n{sheet}</style>", unsafe_allow_html=True)

load_css()
auth.init_db()

# ── SAFETY TIPS ────────────────────────────────────────────────────────────────
TIPS = [
    "Grapefruit can increase how much medicine enters your blood — always check with your doctor.",
    "Leafy greens like spinach are high in Vitamin K, which can reduce how well blood thinners (e.g. warfarin) work.",
    "Dairy products can reduce the absorption of some antibiotics like tetracycline by up to 60%.",
    "Alcohol mixed with painkillers or sleeping pills can be very dangerous — it increases their effect.",
    "Aged cheese and cured meats contain tyramine, which can cause a dangerous blood pressure spike with certain antidepressants.",
    "Taking iron tablets with vitamin C helps your body absorb the iron much better.",
    "St. John's Wort (a herbal supplement) can make birth control pills less effective.",
    "High-fibre meals slow digestion, which can delay how quickly some medicines are absorbed.",
    "Pomelo and Seville orange have the same effect as grapefruit on drug metabolism.",
    "Licorice candy in large amounts can reduce the effect of blood pressure medications.",
]

def get_tip():
    return TIPS[datetime.utcnow().timetuple().tm_yday % len(TIPS)]

# ── ML HELPERS ─────────────────────────────────────────────────────────────────
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

def get_confidence(model, fp):
    try:
        return float(np.max(model.predict_proba(fp))) * 100
    except Exception:
        return None

def get_reason(drug, food):
    d, f = drug.lower(), food.lower()
    if "caffeine" in d and ("coffee" in f or "tea" in f):
        return "Both have stimulants — can raise heart rate."
    if "warfarin" in d and any(x in f for x in ["green", "spinach", "kale"]):
        return "Vitamin K in this food lowers warfarin's effectiveness."
    if "grapefruit" in f or "pomelo" in f:
        return "Can increase the drug level in your blood."
    if "maoi" in d or "tyramine" in d:
        return "Can cause a dangerous rise in blood pressure."
    if "tetracycline" in d:
        return "Dairy or minerals in this food reduce antibiotic absorption."
    return "Possible interaction — consult a pharmacist."

def get_severity(interaction_yes, confidence):
    if not interaction_yes:
        return "safe", "✅ Safe"
    if confidence and confidence >= 80:
        return "danger", "🔴 High Risk"
    return "caution", "⚠️ Caution"

def make_report(username, drug, food, interaction_yes, reason,
                drug_taste, food_taste, confidence, sev):
    lines = [
        "DRUG-FOOD INTERACTION REPORT",
        "=" * 38,
        f"User       : {username}",
        f"Date (UTC) : {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Drug       : {drug}",
        f"Food       : {food}",
        f"Result     : {'INTERACTION' if interaction_yes else 'NO INTERACTION'}",
        f"Risk level : {sev}",
    ]
    if confidence:
        lines.append(f"Confidence : {confidence:.1f}%")
    if interaction_yes:
        lines.append(f"Reason     : {reason}")
    lines += [
        "",
        f"Drug taste : {drug_taste}",
        f"Food taste : {food_taste}",
        "",
        "-" * 38,
        "NOTE: This is for educational purposes only.",
        "Always consult a licensed healthcare professional.",
    ]
    return "\n".join(lines)


# ── LOGIN / SIGNUP PAGE ───────────────────────────────────────────────────────
def auth_screen():

    # Theme toggle — top centre
    _, tc, _ = st.columns([2, 1, 2])
    with tc:
        choice = st.radio(
            "Theme",
            ["☀️ Light", "🌙 Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        new_theme = "dark" if choice == "🌙 Dark" else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

    # Centre the login card
    _, col, _ = st.columns([1, 1.2, 1])
    with col:

        # Logo + title
        st.write("")
        try:
            st.image("logo.png", width=100)
        except Exception:
            pass
        st.markdown(
            "<h2 style='text-align:center;margin:0.4rem 0 0.1rem'>Drug-Food Interaction</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:var(--text-muted);margin-bottom:1.2rem'>"
            "Sign in to check drug-food interactions</p>",
            unsafe_allow_html=True,
        )

        # Tabs
        tab_login, tab_signup = st.tabs(["  Log in  ", "  Create account  "])

        with tab_login:
            with st.form("login_form"):
                st.text_input("Username", key="li_user", placeholder="your_username")
                st.text_input("Password", type="password", key="li_pass", placeholder="••••••••")
                if st.form_submit_button("Log in", use_container_width=True):
                    u = st.session_state.li_user.strip()
                    p = st.session_state.li_pass
                    if not u or not p:
                        st.error("Please fill in both fields.")
                    else:
                        ok, user = auth.verify_user(u, p)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_id   = user["id"]
                            st.session_state.username  = user["username"]
                            st.session_state._nav_idx  = 0
                            st.rerun()
                        else:
                            st.error("Wrong username or password.")

        with tab_signup:
            with st.form("signup_form"):
                st.text_input("Username", key="su_user", placeholder="3–20 letters, numbers, _")
                st.text_input("Email",    key="su_email", placeholder="you@example.com")
                st.text_input("Password", type="password", key="su_pass", placeholder="min 6 characters")
                st.text_input("Confirm password", type="password", key="su_conf", placeholder="same password again")
                if st.form_submit_button("Create account", use_container_width=True):
                    if st.session_state.su_pass != st.session_state.su_conf:
                        st.error("Passwords do not match.")
                    else:
                        ok, msg = auth.create_user(
                            st.session_state.su_user,
                            st.session_state.su_email,
                            st.session_state.su_pass,
                        )
                        (st.success if ok else st.error)(msg)
                        if ok:
                            st.info("Account created — go to Log in tab to continue.")

        # Did you know tip
        st.markdown(
            f"<div class='tip-box' style='margin-top:1rem'>"
            f"<div class='tip-label'>💡 Did you know?</div>"
            f"<div class='tip-text'>{get_tip()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── MAIN APP ──────────────────────────────────────────────────────────────────
def main_app():

    # Load models and data
    try:
        interaction_model = joblib.load("interaction_model.pkl")
        taste_model       = joblib.load("taste_model.pkl")
        drug_data         = pd.read_csv("drug_smiles.csv")
        food_data         = pd.read_csv("food_smiles.csv")
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Make sure all model and CSV files are in the same folder.")
        st.stop()

    history = auth.get_history(st.session_state.user_id)

    # ── SIDEBAR ───────────────────────────────────────────────────────────────
    with st.sidebar:
        try:
            st.image("logo.png", width=65)
        except Exception:
            pass

        # User card
        last = f"Last: {history[0]['drug']} + {history[0]['food']}" if history else "No checks yet"
        st.markdown(
            f"<div class='sidebar-user'>"
            f"<div class='s-name'>👤 {st.session_state.username}</div>"
            f"<div class='s-role'>{len(history)} prediction{'s' if len(history)!=1 else ''} made</div>"
            f"<div class='s-role' style='font-size:11px;margin-top:3px'>{last}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        page = st.radio(
            "Navigation",
            NAV_PAGES,
            index=st.session_state._nav_idx,
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**Appearance**")
        tc = st.radio(
            "theme",
            ["☀️ Light", "🌙 Dark"],
            index=0 if st.session_state.theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        new_theme = "dark" if tc == "🌙 Dark" else "light"
        if new_theme != st.session_state.theme:
            st.session_state.theme = new_theme
            st.rerun()

        st.markdown("---")
        if st.button("🚪 Log out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_id   = None
            st.session_state.username  = None
            st.rerun()

    # ── HOME ──────────────────────────────────────────────────────────────────
    if page == "🏠 Home":

        # Hero
        try:
            st.image("logo.png", width=120)
        except Exception:
            pass
        st.markdown("<h1 style='text-align:center'>Drug-Food Interaction System</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align:center;color:var(--text-muted);font-size:16px'>"
            "Check if a drug and food are safe to take together — powered by machine learning.</p>",
            unsafe_allow_html=True,
        )

        # Tip of the day
        st.markdown(
            f"<div class='tip-box'>"
            f"<div class='tip-label'>💡 Today's tip</div>"
            f"<div class='tip-text'>{get_tip()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # CTA button
        _, mc, _ = st.columns([1, 1, 1])
        with mc:
            if st.button("🔬 Run a Prediction", use_container_width=True):
                go("💊 Predict")

        # Personal stats (only if they have history)
        if history:
            yes_count = sum(1 for r in history if r["interaction"] == "YES")
            conf_vals = [r["confidence"] for r in history if r["confidence"]]
            avg_conf  = f"{sum(conf_vals)/len(conf_vals):.0f}%" if conf_vals else "—"

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Checks",       len(history))
            c2.metric("Interactions Found", yes_count)
            c3.metric("Avg. Confidence",    avg_conf)

        st.markdown("---")
        st.subheader("What this app does")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("<div class='info-card'><h4>💊 Interaction Check</h4><p>Predicts whether a drug and food compound interact using ML models trained on molecular fingerprints.</p></div>", unsafe_allow_html=True)
            st.markdown("<div class='info-card'><h4>👅 Taste Prediction</h4><p>A separate model estimates whether each compound is sweet or bitter.</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='info-card'><h4>🎯 Risk Level</h4><p>Results are graded Safe, Caution, or High Risk based on the prediction and model confidence.</p></div>", unsafe_allow_html=True)
            st.markdown("<div class='info-card'><h4>📋 History & Reports</h4><p>All checks are saved to your account and can be downloaded as a text report.</p></div>", unsafe_allow_html=True)

    # ── PREDICT ───────────────────────────────────────────────────────────────
    elif page == "💊 Predict":
        st.title("💊 Predict Interaction")
        st.write("Select a drug and a food, then click **Predict**.")

        col1, col2 = st.columns(2)
        with col1:
            drug_name = st.selectbox("Select Drug", drug_data["Drug"])
        with col2:
            food_name = st.selectbox("Select Food", food_data["Food"])

        drug_smiles = drug_data[drug_data["Drug"] == drug_name]["SMILES"].values[0]
        food_smiles = food_data[food_data["Food"] == food_name]["SMILES"].values[0]

        st.write("")
        if st.button("🔬 Predict", use_container_width=True):
            drug_fp = smiles_to_fp(drug_smiles)
            food_fp = smiles_to_fp(food_smiles)

            if drug_fp is None or food_fp is None:
                st.error("Invalid SMILES in your data file — please check the CSV.")
            else:
                combined = np.concatenate((drug_fp, food_fp)).reshape(1, -1)
                pred      = interaction_model.predict(combined)[0]
                conf      = get_confidence(interaction_model, combined)
                drug_taste = "Sweet" if taste_model.predict(drug_fp.reshape(1,-1))[0] == 1 else "Bitter"
                food_taste = "Sweet" if taste_model.predict(food_fp.reshape(1,-1))[0] == 1 else "Bitter"

                interaction_yes     = (pred == 1)
                reason              = get_reason(drug_name, food_name) if interaction_yes else "—"
                sev_key, sev_label  = get_severity(interaction_yes, conf)

                # Capsule visual
                st.markdown(
                    f"<div class='capsule-wrap'>"
                    f"<div class='cap-red'></div><div class='cap-blue'></div>"
                    f"<div class='cap-ekg'>"
                    f"<svg width='120' height='24' viewBox='0 0 120 24'>"
                    f"<polyline points='0,12 30,12 38,2 46,22 54,12 120,12' "
                    f"stroke='white' stroke-width='2.5' fill='none' stroke-linecap='round'/>"
                    f"</svg></div></div>"
                    f"<div class='capsule-label'>{'⚠️ Interaction Detected' if interaction_yes else '✅ No Interaction'}</div>",
                    unsafe_allow_html=True,
                )

                # Badge
                st.markdown(
                    f"<div style='text-align:center;margin:8px 0'>"
                    f"<span class='badge badge-{sev_key}'>{sev_label}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Drug Taste",  drug_taste)
                m2.metric("Food Taste",  food_taste)
                m3.metric("Confidence",  f"{conf:.0f}%" if conf else "—")

                # Result message
                if interaction_yes:
                    st.error(f"⚠️ **Interaction likely.** {reason}")
                else:
                    st.success("✅ No significant interaction predicted.")

                # Save to history
                auth.save_history(
                    st.session_state.user_id, drug_name, food_name,
                    "YES" if interaction_yes else "NO",
                    reason, drug_taste, food_taste, conf,
                )

                # Download report
                report = make_report(
                    st.session_state.username, drug_name, food_name,
                    interaction_yes, reason, drug_taste, food_taste, conf, sev_label,
                )
                st.download_button(
                    "📄 Download Report",
                    data=report,
                    file_name=f"DFI_{drug_name}_{food_name}.txt",
                    mime="text/plain",
                )

                st.markdown(
                    "<p class='disclaimer'>For educational purposes only. "
                    "Always consult a healthcare professional before changing your medication or diet.</p>",
                    unsafe_allow_html=True,
                )

    # ── HISTORY ───────────────────────────────────────────────────────────────
    elif page == "🕘 History":
        st.title("🕘 Your History")

        history = auth.get_history(st.session_state.user_id)

        if not history:
            st.info("You haven't made any predictions yet. Go to the Predict page to get started.")
        else:
            # Build a simple table
            rows = []
            for r in history:
                date = r["created_at"][:10]
                icon = "⚠️" if r["interaction"] == "YES" else "✅"
                rows.append({
                    "Date": date,
                    "Drug": r["drug"],
                    "Food": r["food"],
                    "Result": f"{icon} {r['interaction']}",
                    "Confidence": f"{r['confidence']:.0f}%" if r["confidence"] else "—",
                    "Drug Taste": r["drug_taste"],
                    "Food Taste": r["food_taste"],
                })

            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    "⬇️ Download as CSV",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="my_history.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True):
                    auth.clear_history(st.session_state.user_id)
                    st.success("History cleared.")
                    st.rerun()

    # ── ABOUT ─────────────────────────────────────────────────────────────────
    elif page == "ℹ️ About":
        st.title("ℹ️ About This Project")

        st.write(
            "This is a machine learning project that predicts whether a drug "
            "and a food can have a harmful interaction when taken together."
        )
        st.write("**How it works:**")
        st.write("1. Each drug and food is represented by its SMILES string (a text code for its chemical structure).")
        st.write("2. We convert SMILES into Morgan fingerprints — a list of numbers that represent the molecule's shape.")
        st.write("3. A trained ML model predicts if those two fingerprints together signal an interaction.")
        st.write("4. A second model predicts whether each compound tastes sweet or bitter.")

        st.markdown("---")
        st.write("**Security:** Passwords are never saved as plain text. They are hashed using PBKDF2-SHA256 before being stored.")

        st.markdown(
            f"<div class='tip-box'>"
            f"<div class='tip-label'>💡 Today's tip</div>"
            f"<div class='tip-text'>{get_tip()}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<p class='disclaimer'>This project is for educational purposes only and is not medical advice.</p>",
            unsafe_allow_html=True,
        )


# ── RUN ───────────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    auth_screen()
else:
    main_app()