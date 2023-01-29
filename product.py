import streamlit as st
import time
import requests
import pandas as pd
st.set_page_config(
    page_title="RecommenDate:Find your soulmate",
    page_icon= ":revolving_hearts:",
    layout="wide",
    initial_sidebar_state="expanded")

with open("style.css") as a:
    st.markdown(f'<style>{a.read()}</style>',unsafe_allow_html=True)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 500px;
        margin-left: -500px;
    }

    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>RecommenDate: Find your soulmate 💞</h1>", unsafe_allow_html=True)
    with st.form(key="form1"):
        with st.container():
            st.title("Short questions about you")
            sel_col, disp_col = st.columns(2)
            gender = sel_col.selectbox("What is your gender?", options=["m", "f"])
            age = sel_col.number_input("How old are you?", min_value=18, max_value=70)
            orientation = disp_col.selectbox("What is your sexual orientation?", options=["straight", "gay","bisexual"])
        with st.container():
            st.title("Now tell me about yourself")
            st.caption("Don't Forget: The more you tell, the easier you find your soulmate :wink:")
            sel_col, disp_col = st.columns(2)
            col1,col2 = st.columns(2)
            with col1:
                essay0 = col1.text_area("My self summary")
                essay1 = col1.text_area("What I’m doing with my life")
                essay2 =col1.text_area("I’m really good at")
                essay3 =col1.text_area("The first thing people usually notice about me")
                essay4 =col1.text_area("Favorite books, movies, show, music, and food")
            with col2:
                essay5 =col2.text_area("The six things I could never do without")
                essay6 =col2.text_area("I spend a lot of time thinking about")
                essay7 =col2.text_area("On a typical Friday night I am")
                essay8 =col2.text_area("The most private thing I am willing to admit")
                essay9 =col2.text_area("You should message me if...")
            submitted = st.form_submit_button(label= "Bring my soulmate", )
url = 'https://recommedate-ygdtvu6iaa-ew.a.run.app/predict'

params ={
    'sex':gender,
    'orientation':orientation,
    'essay0' : essay0,
    'essay1' : essay1,
    'essay2' : essay2,
    'essay3' : essay3,
    'essay4' : essay4,
    'essay5' : essay5,
    'essay6' : essay6,
    'essay7' : essay7,
    'essay8' : essay8,
    'essay9' : essay9}
if submitted:
    with st.spinner(":mag: We are finding your possible soulmates..."):
        time.sleep(2)
    with st.progress(0):
        for percent_complete in range(100):
            time.sleep(0.025)
            st.progress(0).progress(percent_complete + 1).empty()

    req = requests.get(url, params=params)
    res = req.json()
    df=pd.DataFrame(res)
    # df=df.drop(columns=['last_online','status'])
    # df=df.fillna('')
    # df=df.apply(pd.to_numeric, errors='ignore')
    df.drop(columns=[
        "status",
        "body_type",
        "diet",
        "drinks",
        "drugs",
        "education",
        "ethnicity",
        "height","income","job","last_online","location","offspring","pets","religion","sign","smokes","speaks"],inplace=True)
    tab1, tab2, tab3 = st.tabs(["1","2","3"])
    with tab1:
        st.header(df.index[0])
        col1,col2,col3 = st.columns(3)
        with st.container():
            col1.subheader("Age:")
            col1.markdown(df.iloc[0]["age"])
        with st.container():
            col2.subheader("Sex")
            col2.markdown(df.iloc[0]["sex"])
        with st.container():
            col3.subheader("Orientation")
            col3.markdown(df.iloc[0]["orientation"])
        with st.container():
            st.subheader("My self summary:")
            st.text(df.iloc[0]["essay0"])
        with st.container():
            st.subheader("What I'm doing with my life:")
            st.markdown(df.iloc[0]["essay1"])
        with st.container():
            st.subheader("I'm really good at:")
            st.markdown(df.iloc[0]["essay2"])
        with st.container():
            st.subheader("The first thing people usually notice about me:")
            st.markdown(df.iloc[0]["essay3"])
        with st.container():
            st.subheader("Favorite books, movies, show, music, and food:")
            st.markdown(df.iloc[0]["essay4"])
        with st.container():
            st.subheader("The six things I could never do without:")
            st.markdown(df.iloc[0]["essay5"])
        with st.container():
            st.subheader("I spend a lot of time thinking about:")
            st.markdown(df.iloc[0]["essay6"])
        with st.container():
            st.subheader("On a typical Friday night I am:")
            st.markdown(df.iloc[0]["essay7"])
        with st.container():
            st.subheader("The most private thing I am willing to admit:")
            st.markdown(df.iloc[0]["essay8"])
        with st.container():
            st.subheader("You should message me if...:")
            st.markdown(df.iloc[0]["essay9"])
    with tab2:
        st.header(df.index[1])
        col1,col2,col3 = st.columns(3)
        with st.container():
            col1.subheader("Age:")
            col1.markdown(df.iloc[1]["age"])
        with st.container():
            col2.subheader("Sex")
            col2.markdown(df.iloc[1]["sex"])
        with st.container():
            col3.subheader("Orientation")
            col3.markdown(df.iloc[1]["orientation"])
        with st.container():
            st.subheader("My self summary")
            st.markdown(df.iloc[1]["essay0"])
        with st.container():
            st.subheader("What I'm doing with my life")
            st.markdown(df.iloc[1]["essay1"])
        with st.container():
            st.subheader("I'm really good at")
            st.markdown(df.iloc[1]["essay2"])
        with st.container():
            st.subheader("The first thing people usually notice about me")
            st.markdown(df.iloc[1]["essay3"])
        with st.container():
            st.subheader("Favorite books, movies, show, music, and food")
            st.markdown(df.iloc[1]["essay4"])
        with st.container():
            st.subheader("The six things I could never do without")
            st.markdown(df.iloc[1]["essay5"])
        with st.container():
            st.subheader("I spend a lot of time thinking about")
            st.markdown(df.iloc[1]["essay6"])
        with st.container():
            st.subheader("On a typical Friday night I am")
            st.markdown(df.iloc[1]["essay7"])
        with st.container():
            st.subheader("The most private thing I am willing to admit")
            st.markdown(df.iloc[1]["essay8"])
        with st.container():
            st.subheader("You should message me if...")
            st.markdown(df.iloc[1]["essay9"])
    with tab3:
        st.header(df.index[2])
        col1,col2,col3 = st.columns(3)
        with st.container():
            col1.subheader("Age:")
            col1.markdown(df.iloc[2]["age"])
        with st.container():
            col2.subheader("Sex")
            col2.markdown(df.iloc[2]["sex"])
        with st.container():
            col3.subheader("Orientation")
            col3.text(df.iloc[2]["orientation"])
        with st.container():
            st.subheader("My self summary:")
            st.markdown(df.iloc[2]["essay0"])
        with st.container():
            st.subheader("What I'm doing with my life:")
            st.markdown(df.iloc[2]["essay1"])
        with st.container():
            st.subheader("I'm really good at:")
            st.markdown(df.iloc[2]["essay2"])
        with st.container():
            st.subheader("The first thing people usually notice about me:")
            st.markdown(df.iloc[2]["essay3"])
        with st.container():
            st.subheader("Favorite books, movies, show, music, and food:")
            st.markdown(df.iloc[2]["essay4"])
        with st.container():
            st.subheader("The six things I could never do without:")
            st.markdown(df.iloc[2]["essay5"])
        with st.container():
            st.subheader("I spend a lot of time thinking about:")
            st.markdown(df.iloc[2]["essay6"])
        with st.container():
            st.subheader("On a typical Friday night I am:")
            st.markdown(df.iloc[2]["essay7"])
        with st.container():
            st.subheader("The most private thing I am willing to admit:")
            st.markdown(df.iloc[2]["essay8"])
        with st.container():
            st.subheader("You should message me if...:")
            st.markdown(df.iloc[2]["essay9"])
