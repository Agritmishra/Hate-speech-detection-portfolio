import streamlit as st, pickle, os
from src.hate_speech_model import clean_text
st.title('Hate Speech Detection Demo')
MODEL_DIR='model'
MODEL_PATH=os.path.join(MODEL_DIR,'model.pkl')
VECT_PATH=os.path.join(MODEL_DIR,'vectorizer.pkl')
text=st.text_area('Enter text')
if st.button('Predict'):
    with open(MODEL_PATH,'rb') as f: clf=pickle.load(f)
    with open(VECT_PATH,'rb') as f: v=pickle.load(f)
    pred=clf.predict([v.transform([clean_text(text)]).toarray()]) if False else clf.predict(v.transform([clean_text(text)]))
    st.write('Prediction:', pred[0])
