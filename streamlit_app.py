import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import roc_curve, auc

model = joblib.load("svc_model.pkl")

data = pd.read_csv("cardio_train.csv", sep=";")
data.drop("id",axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data["bmi"] = data["weight"] / (data["height"]/100)**2
out_filter = ((data["ap_hi"]>250) | (data["ap_lo"]>200))
data = data[~out_filter]

out_filter2 = ((data["ap_hi"] < 0) | (data["ap_lo"] < 0))
data = data[~out_filter2]

# SIDEBAR:
st.sidebar.header("Введите свои данные")

age_years = st.sidebar.slider("Возраст (лет)", 18, 100, 50)
age = age_years * 365  # переводим в дни

gender = st.sidebar.radio("Пол", options=[1, 2], format_func=lambda x: "Женщина" if x == 1 else "Мужчина")
height = st.sidebar.number_input("Рост (см)", 100, 220, 170)
weight = st.sidebar.number_input("Вес (кг)", 30, 200, 70)
ap_hi = st.sidebar.number_input("Систолическое давление (ap_hi)", 80, 200, 120)
ap_lo = st.sidebar.number_input("Диастолическое давление (ap_lo)", 50, 140, 80)

cholesterol = st.sidebar.selectbox("Холестерин", options=[1, 2, 3],
                                   format_func=lambda x: {1: "Норма", 2: "Выше нормы", 3: "Значительно выше"}[x])

gluc = st.sidebar.selectbox("Глюкоза", options=[1, 2, 3],
                            format_func=lambda x: {1: "Норма", 2: "Выше нормы", 3: "Значительно выше"}[x])

smoke = st.sidebar.radio("Курение", options=[0, 1], format_func=lambda x: "Нет" if x == 0 else "Да")
alco = st.sidebar.radio("Алкоголь", options=[0, 1], format_func=lambda x: "Нет" if x == 0 else "Да")
active = st.sidebar.radio("Физическая активность", options=[0, 1], format_func=lambda x: "Нет" if x == 0 else "Да")

# Собираем данные
user_data = pd.DataFrame({
    "age": [age],
    "gender": [gender],
    "height": [height],
    "weight": [weight],
    "ap_hi": [ap_hi],
    "ap_lo": [ap_lo],
    "cholesterol": [cholesterol],
    "gluc": [gluc],
    "smoke": [smoke],
    "alco": [alco],
    "active": [active],
    "bmi": weight / ((height / 100) ** 2)
})

st.title("🫀 Cardiovascular Disease Prediction App")

plot_choice = st.selectbox(
    "Выберите график:",
    ["Нет", "Распределение классов", "Корреляция признаков", "Распределение возраста", "ROC-кривая"]
)

if plot_choice == "Распределение классов":
    plt.figure(figsize=(6,4))
    sns.countplot(x="cardio", data=data)
    st.pyplot(plt)

elif plot_choice == "Корреляция признаков":
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(), annot=False, cmap="coolwarm")
    st.pyplot(plt)

elif plot_choice == "Распределение возраста":
    plt.figure(figsize=(6,4))
    sns.histplot(data["age"] / 365, bins=30, kde=True)
    plt.xlabel("Возраст (лет)")
    st.pyplot(plt)

elif plot_choice == "ROC-кривая":
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, auc

    X = data.drop(columns=["cardio"])
    y = data["cardio"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_score = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0,1],[0,1], color="red", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривая")
    plt.legend(loc="lower right")
    st.pyplot(plt)

if st.sidebar.button("Сделать предсказание"):
    prediction = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0][1]

    st.subheader("Результат предсказания")
    st.write(f"Вероятность наличия сердечно-сосудистого заболевания: **{proba:.2f}**")
    st.write(f"Класс: **{'Есть заболевание' if prediction == 1 else 'Здоров'}**")
