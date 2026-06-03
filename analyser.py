from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def sentiment_score(text):

    blob = TextBlob(text)

    polarity = blob.sentiment.polarity

    score = (polarity + 1) * 50

    return round(score, 2)


def similarity_score(user_answer, ideal_answer):

    embeddings = model.encode(
        [user_answer, ideal_answer]
    )

    similarity = cosine_similarity(
        [embeddings[0]],
        [embeddings[1]]
    )[0][0]

    return round(similarity * 100, 2)


def keyword_score(text):

    keywords = [
        "project",
        "teamwork",
        "leadership",
        "machine learning",
        "ai",
        "skills",
        "experience",
        "problem solving"
    ]

    text = text.lower()

    found = 0

    for keyword in keywords:
        if keyword in text:
            found += 1

    score = (found / len(keywords)) * 100

    return round(score, 2)


def overall_score(sentiment, similarity, keywords):

    score = (
        0.3 * sentiment +
        0.4 * similarity +
        0.3 * keywords
    )

    return round(score, 2)


def generate_feedback(text):

    feedback = []

    text = text.lower()

    if "project" not in text:
        feedback.append(
            "Mention technical projects you have worked on."
        )

    if "teamwork" not in text and "team" not in text:
        feedback.append(
            "Include examples of teamwork."
        )

    if "leadership" not in text:
        feedback.append(
            "Add leadership experiences if applicable."
        )

    if "experience" not in text:
        feedback.append(
            "Talk about internships or practical experience."
        )

    if len(feedback) == 0:
        feedback.append(
            "Excellent answer. Covers most important areas."
        )

    return feedback