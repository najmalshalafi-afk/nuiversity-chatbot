import json
import random
import joblib
import streamlit as st

st.set_page_config(page_title="المساعد الذكي للجامعة", page_icon="🎓")

st.title("🎓 المساعد الذكي للاستفسارات الجامعية")
st.caption("نموذج ذكاء اصطناعي قائم على الشبكات العصبية العميقة (MLP)")

@st.cache_resource
def load_resources():
    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)
    model = joblib.load('chatbot_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    return intents, model, vectorizer

intents, model, vectorizer = load_resources()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "أهلاً بك! كيف يمكنني مساعدتك؟"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    X_input = vectorizer.transform([user_input])
    probs = model.predict_proba(X_input)
    max_prob = max(probs[0])
    predicted_tag = model.predict(X_input)[0]

    if max_prob > 0.30:
        for intent in intents['intents']:
            if predicted_tag == intent["tag"]:
                bot_response = random.choice(intent['responses'])
    else:
        bot_response = "عذراً، لم أفهم استفسارك بدقة. حاول إعادة صياغة السؤال."

    with st.chat_message("assistant"):
        st.markdown(bot_response)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
