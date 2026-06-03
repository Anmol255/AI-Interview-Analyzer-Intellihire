import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt

from analyser import sentiment_score
from analyser import similarity_score
from analyser import keyword_score
from analyser import overall_score
from analyser import generate_feedback

from questions import questions


st.set_page_config(
    page_title="IntelliHire",
    page_icon="🤖",
    layout="wide"
)


def save_results(question,
                 sentiment,
                 similarity,
                 keywords,
                 overall):

    data = {
        "Question": [question],
        "Sentiment": [sentiment],
        "Similarity": [similarity],
        "Keywords": [keywords],
        "Overall": [overall]
    }

    df = pd.DataFrame(data)

    if os.path.exists("results.csv"):
        df.to_csv(
            "results.csv",
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            "results.csv",
            index=False
        )


st.title("🤖 IntelliHire")
st.subheader(
    "AI-Powered Interview Response Evaluation System"
)

st.markdown("---")

question = st.selectbox(
    "Select Interview Question",
    list(questions.keys())
)

answer = st.text_area(
    "Enter Your Answer"
)

if st.button("Analyze Answer"):

    ideal_answer = questions[question]

    sentiment = sentiment_score(answer)

    similarity = similarity_score(
        answer,
        ideal_answer
    )

    keywords = keyword_score(answer)

    overall = overall_score(
        sentiment,
        similarity,
        keywords
    )

    feedback = generate_feedback(answer)

    save_results(
        question,
        sentiment,
        similarity,
        keywords,
        overall
    )

    st.subheader("📊 Performance Scores")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Sentiment Score",
            f"{sentiment:.2f}"
        )

        st.metric(
            "Keyword Score",
            f"{keywords:.2f}"
        )

    with col2:
        st.metric(
            "Similarity Score",
            f"{similarity:.2f}"
        )

        st.metric(
            "Overall Score",
            f"{overall:.2f}"
        )

    st.write("Overall Performance")

    st.progress(int(overall))

    scores = {
        "Sentiment": sentiment,
        "Similarity": similarity,
        "Keywords": keywords,
        "Overall": overall
    }

    fig, ax = plt.subplots()

    ax.bar(
        scores.keys(),
        scores.values()
    )

    ax.set_ylim(0, 100)

    st.pyplot(fig)

    st.subheader("💡 Suggestions")

    for item in feedback:
        st.write("•", item)

if os.path.exists("results.csv"):

    st.markdown("---")

    st.subheader("📁 Previous Interview Attempts")

    df = pd.read_csv("results.csv")

    st.dataframe(df)