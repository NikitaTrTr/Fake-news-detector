import sys
sys.path.insert(1, 'ml_pipeline/')
import streamlit as st
import pandas as pd
import pickle
from utils import clean_text
from utils import vectorize
from utils import rubert_predict_proba
from streamlit_option_menu import option_menu
from transformers import AutoModelForSequenceClassification



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
            text_news = vectorize(pd.Series([clean_text(str(input))], name='text'))
            pred = model.predict_proba(text_news)
            return pred[0][0]


    option = st.selectbox(
        'Выберите модель',
        ('Logistic Regression', 'Random Forest Classifier',
        'Support Vector Classifier', 'CatBoost', 'ruBERT'))

   

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Проверить"):
            model = None
            try:
                if option == "Logistic Regression":
                    model = pickle.load(open('./models/logreg.pkl', 'rb'))
                if option == "Random Forest Classifier":
                    model = pickle.load(open('./models/random_forest.pkl', 'rb'))
                if option == "Support Vector Classifier":
                    model = pickle.load(open('./models/svc.pkl', 'rb'))
                if option == "CatBoost":
                    model = pickle.load(open('./models/catboost.pkl', 'rb'))
                if option == "ruBERT":
                    model = AutoModelForSequenceClassification.from_pretrained('./models/rubert/')
            except:
                st.write('Ошибка загрузки модели')
            if model:
                try:
                    ans = check_text(input, option, model)
                    strin = "Новость достоверна с вероятностью" + str(round(ans * 100)) + "%."
                    if ans<0.2:
                        st.image('tinkoff/0.png')
                    elif ans<0.4:
                        st.image('tinkoff/1.png')
                    elif ans<0.6:
                        st.image('tinkoff/2.png')
                    elif ans<0.8:
                        st.image('tinkoff/3.png')
                    else:
                        st.image('tinkoff/4.png')
                except:
                    strin = "К сожалению, это не новость"
                st.write(strin)

    with col2:
        st.button("Очистить поле", on_click=clear_text)

else:
    st.write("Над проектом работали: Никита, Кирилл, Алексей, Иван")
