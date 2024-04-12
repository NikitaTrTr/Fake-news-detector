import streamlit as st
import pandas as pd
import pickle
from ml_pipeline.utils import clean_text
from ml_pipeline.utils import vectorize
from ml_pipeline.utils import rubert_predict_proba
from streamlit_option_menu import option_menu
from transformers import AutoModelForSequenceClassification

import warnings
warnings.filterwarnings('ignore')


st.subheader("Fake News Detector")

st.markdown("""
<style>
.big-font {
    font-size:20px !important;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu("Fake News Detector", ["Проверка новостей", 'О проекте'], default_index=0)

if selected == "Проверка новостей":
    st.markdown('<p class="big-font">Наш сервис позволяет проверить на правдивость любую новость</p>', unsafe_allow_html=True)

    input = st.text_area("Вставьте текст новостной статьи в поле ниже", key="text")


    def clear_text():
        st.session_state["text"] = ""


    def check_text(input, option, model):
        if option == "ruBERT":
            pred = rubert_predict_proba(clean_text(str(input)), model)
            return pred[0][1]['score']
        else:
            text_news = vectorize(pd.Series([clean_text(str(input))], name='articleBody2'))
            pred = model.predict_proba(text_news)
            return pred[0][0]


    option = st.selectbox(
        'Выберите модель',
        ('Logistic Regression', 'Decision Tree Classifier', 'Random Forest Classifier',
        'Support Vector Classifier', 'CatBoost', 'ruBERT'))

   

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Проверить"):
            try:
                if option == "Logistic Regression":
                    model = pickle.load(open('../Models/logreg.pkl', 'rb'))
                if option == "Decision Tree Classifier":
                    model = pickle.load(open('../Models/tree.pkl', 'rb'))
                if option == "Random Forest Classifier":
                    model = pickle.load(open('../Models/random_forest.pkl', 'rb'))
                if option == "Support Vector Classifier":
                    model = pickle.load(open('../Models/svc.pkl', 'rb'))
                if option == "CatBoost":
                    model = pickle.load(open('../Models/catboost.pkl', 'rb'))
                if option == "ruBERT":
                    model = AutoModelForSequenceClassification.from_pretrained('./Models/rubert/')

                ans = check_text(input, option, model)
                strin = "Вероятность того, что новость правдивая: " + str(round(ans * 100)) + "%"
            except:
                strin = "К сожалению - это не новость"
            st.write(strin)

    with col2:
        st.button("Очистить поле", on_click=clear_text)

else:
    st.write("Над проектом работали: Никита, Кирилл, Алексей, Иван")
