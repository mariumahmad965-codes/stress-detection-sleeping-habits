import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    df = pd.read_csv('SaYoPillow.csv')
    df.rename(columns={
        'sr':'snoring_rate','rr':'respiration_rate','t':'body_temperature',
        'lm':'limb_movement','bo':'blood_oxygen','rem':'rem_sleep',
        'sr.1':'sleep_hours','hr':'heart_rate','sl':'stress_level'
    }, inplace=True)
    df.drop_duplicates(inplace=True)

    df['oxygen_heart_ratio']  = df['blood_oxygen'] / df['heart_rate']
    df['sleep_quality_score'] = df['rem_sleep'] * df['sleep_hours']
    df['stress_indicator']    = df['snoring_rate'] + df['limb_movement'] + df['respiration_rate']

    X = df.drop(columns=['stress_level'])
    y = df['stress_level'].values

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    model = VotingClassifier(estimators=[
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ], voting='hard')
    model.fit(X_train_s, y_train)

    return model, scaler, list(X.columns)

st.set_page_config(page_title="Stress Detector", page_icon="😴", layout="centered")
st.title("😴 Human Stress Detection")
st.subheader("Based on Sleeping Habits — SaYoPillow Dataset (IEEE)")
st.info("⏳ First load trains the model (~10 seconds). After that it's instant!")

model, scaler, columns = train_model()

with st.form("predict"):
    col1, col2 = st.columns(2)
    with col1:
        snoring     = st.slider("Snoring Rate (%)",        0.0, 100.0, 50.0)
        respiration = st.slider("Respiration Rate",       15.0,  30.0, 20.0)
        temp        = st.slider("Body Temperature (°F)",  85.0, 100.0, 96.0)
        limb        = st.slider("Limb Movement",           0.0,  20.0,  8.0)
    with col2:
        oxygen      = st.slider("Blood Oxygen (%)",       85.0, 100.0, 95.0)
        rem         = st.slider("REM Sleep (%)",          80.0, 100.0, 85.0)
        sleep_hrs   = st.slider("Sleep Hours",             0.0,   9.0,  7.0)
        hr          = st.slider("Heart Rate (BPM)",       50.0,  90.0, 65.0)
    submitted = st.form_submit_button("🔍 Predict Stress Level")

if submitted:
    d = pd.DataFrame([[snoring, respiration, temp, limb, oxygen, rem, sleep_hrs, hr]],
                     columns=["snoring_rate","respiration_rate","body_temperature",
                              "limb_movement","blood_oxygen","rem_sleep","sleep_hours","heart_rate"])
    d["oxygen_heart_ratio"]  = d["blood_oxygen"] / d["heart_rate"]
    d["sleep_quality_score"] = d["rem_sleep"] * d["sleep_hours"]
    d["stress_indicator"]    = d["snoring_rate"] + d["limb_movement"] + d["respiration_rate"]

    pred = model.predict(scaler.transform(d[columns]))[0]
    labels = {0:"🟢 Low", 1:"🟡 Low-Medium", 2:"🟠 Medium", 3:"🔴 Medium-High", 4:"🚨 High"}

    st.success(f"### Predicted Stress Level: {pred} — {labels[pred]}")

    tips = {
        0: "Great! Your sleep looks healthy 😊",
        1: "Mild stress. Try to get a bit more sleep.",
        2: "Moderate stress. Consider relaxation before bed.",
        3: "High stress detected. Prioritize sleep hygiene.",
        4: "Very high stress! Please rest and consult a doctor if needed."
    }
    st.warning(tips[pred])
