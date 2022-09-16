from ssl import Options
import streamlit as st
import time
import requests


st.set_page_config(page_title="RecommenDate:Find your soulmate", page_icon= ":revolving_hearts:", layout="wide", )
st.subheader("RecommenDate: Find your soulmate :revolving_hearts:")
with st.form(key="form1"):
    with st.container():
        st.title("Short questions about you")
        sel_col, disp_col = st.columns(2)
        gender = sel_col.selectbox("What is your gender?", options=["Male", "Female"])
        age = sel_col.number_input("How old are you?", min_value=18, max_value=70)
        orientation = sel_col.selectbox("What is your sexual orientation?", options=["Straight", "Gay", "Lesiban", "Bisexual"])
    with st.container():
        st.title("Now tell me about yourself")
        st.markdown("Don't Forget: The more you tell, the easier you find your soulmate :wink:")
        sel_col, disp_col = st.columns(2)
        essay0 = sel_col.text_input("My self summary")
        essay1 = sel_col.text_input("What I’m doing with my life")
        essay2 =sel_col.text_input("I’m really good at")
        essay3 =sel_col.text_input("The first thing people usually notice about me")
        essay4 =sel_col.text_input("Favorite books, movies, show, music, and food")
        essay5 =sel_col.text_input("The six things I could never do without")
        essay6 =sel_col.text_input("I spend a lot of time thinking about")
        essay7 =sel_col.text_input("On a typical Friday night I am")
        essay8 =sel_col.text_input("The most private thing I am willing to admit")
        essay9 =sel_col.text_input("You should message me if...")

        submitted = st.form_submit_button(label= "Bring my soulmate", )

if submitted:
    with st.spinner(":mag: We are finding your possible soulmates..."):
        time.sleep(2)
    with st.progress(0):
        for percent_complete in range(100):
            time.sleep(0.025)
            st.progress(0).progress(percent_complete + 1)
    st.subheader("Here are the results :revolving_hearts:")
    st.image("https://cdn.icon-icons.com/icons2/906/PNG/512/user-and-heart_icon-icons.com_69814.png", width=250)
    st.button('Go to Profile', key=1)
    st.success(f"%95 Match ✅")
    st.image("https://cdn.icon-icons.com/icons2/906/PNG/512/user-and-heart_icon-icons.com_69814.png", width=250)
    st.button('Go to Profile', key=2)
    st.success(f"%85 Match")
    st.image("https://cdn.icon-icons.com/icons2/906/PNG/512/user-and-heart_icon-icons.com_69814.png", width=250)
    st.button('Go to Profile', key=3)
    st.success(f"%80 Match")

# enter here the address of your flask api
#url = 'https://.ai/predict'

#params = dict(
#   essay0 = essay0,
#   essay1 = essay1,
#   essay2 = essay2,
#   essay3 = essay3,
#   essay4 = essay4,
#   essay5 = essay5,
#   essay6 = essay6,
#   essay7 = essay7,
#   essay8 = essay8,
#   essay9 = essay9)


#response = requests.get(url, params=params)

#prediction = response.json()

#pred = prediction['']

#pred
