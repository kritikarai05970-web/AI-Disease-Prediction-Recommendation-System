import streamlit as st
import pandas as pd
import pickle
import numpy as np
import ast


# Loading the datasets 

@st.cache_resource
def load_resources():
    model = pickle.load(open("model.pkl", "rb"))

    desc = pd.read_csv("description.csv")
    precaution = pd.read_csv("precautions_df.csv")
    meds = pd.read_csv("medications.csv")
    diet = pd.read_csv("diets.csv")
    workout = pd.read_csv("workout_df.csv")
    training = pd.read_csv("Training.csv")

    features = training.columns[:-1].tolist()

    # Standardize disease names 
    desc['Disease'] = desc['Disease'].str.strip().str.lower()
    precaution['Disease'] = precaution['Disease'].str.strip().str.lower()
    meds['Disease'] = meds['Disease'].str.strip().str.lower()
    diet['Disease'] = diet['Disease'].str.strip().str.lower()
    workout['disease'] = workout['disease'].str.strip().str.lower()

    return model, desc, precaution, meds, diet, workout, features


model, desc, precaution, meds, diet, workout, features = load_resources()

# about page title 
st.set_page_config(page_title="AI Medical System", layout="wide")
st.title("🩺 AI Disease Prediction & Recommendation System")

if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "top3" not in st.session_state:
    st.session_state.top3 = None

selected_symptoms = st.multiselect(
    "Select Symptoms:",
    options=features
)

if st.button(" Predict Disease"):

    if not selected_symptoms:
        st.error("Please select at least one symptom.")
    else:
        input_vector = np.zeros(len(features))

        for symptom in selected_symptoms:
            input_vector[features.index(symptom)] = 1

        # Prediction
        prediction = model.predict([input_vector])[0]

        # Probability
        probabilities = model.predict_proba([input_vector])[0]
        confidence = round(np.max(probabilities) * 100, 2)

        # Top 3
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        top3_diseases = model.classes_[top3_indices]
        top3_probs = probabilities[top3_indices] * 100

        st.session_state.prediction = prediction
        st.session_state.confidence = confidence
        st.session_state.top3 = list(zip(top3_diseases, top3_probs))


#  results we predication 

if st.session_state.prediction:

    disease = st.session_state.prediction.strip().lower()

    st.success(f"### 🦠 Predicted Disease: {disease.title()}")
    #st.info(f"🎯 Confidence: {st.session_state.confidence}%")

   
    st.subheader(" Top 3 Predictions")
    for d, p in st.session_state.top3:
        st.write(f"{d} — {round(p,2)}%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Description", "Precautions", "Medications", "Diet", "Workout"]
    )

    # description 
    with tab1:
        d = desc[desc['Disease'] == disease]['Description']
        st.write(d.values[0] if len(d) > 0 else "No description available.")

    # precaution
    with tab2:
        p_df = precaution[precaution['Disease'] == disease]
        if not p_df.empty:
            for item in p_df.iloc[0, 1:]:
                if pd.notna(item):
                    st.write(f" {item}")
        else:
            st.write("No precautions available.")

    # medication
    with tab3:
        m = meds[meds['Disease'] == disease]['Medication']
        if len(m) > 0:
            try:
                med_list = ast.literal_eval(m.values[0])
                for med in med_list:
                    st.write(f" {med}")
            except:
                st.write(m.values[0])
        else:
            st.write("No medications available.")

    # diet
    with tab4:
        d = diet[diet['Disease'] == disease]['Diet']
        if len(d) > 0:
            try:
                diet_list = ast.literal_eval(d.values[0])
                for item in diet_list:
                    st.write(f" {item}")
            except:
                st.write(d.values[0])
        else:
            st.write("No diet plan available.")

    # workout
    with tab5:
        w = workout[workout['disease'] == disease]['workout']
        if len(w) > 0:
            for item in w:
                st.write(f" {item}")
        else:
            st.write("No workout advice available.")
