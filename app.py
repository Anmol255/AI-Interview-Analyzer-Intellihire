import pandas as pd
import os

from analyser import sentiment_score
from analyser import similarity_score
from analyser import keyword_score
from analyser import overall_score
from analyser import generate_feedback

from questions import questions


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

    file_name = "results.csv"

    if os.path.exists(file_name):
        df.to_csv(
            file_name,
            mode="a",
            header=False,
            index=False
        )
    else:
        df.to_csv(
            file_name,
            index=False
        )


print("\n===== AI INTERVIEW ANALYZER =====\n")

question_list = list(questions.keys())

for i, q in enumerate(question_list, start=1):
    print(f"{i}. {q}")

choice = int(input("\nSelect Question Number: "))

selected_question = question_list[choice - 1]

answer = input(
    f"\n{selected_question}\n\nYour Answer:\n"
)

ideal_answer = questions[selected_question]

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
    selected_question,
    sentiment,
    similarity,
    keywords,
    overall
)

print("\n==========================")
print(" AI INTERVIEW ANALYSIS")
print("==========================")

print(f"Sentiment Score : {sentiment}")
print(f"Similarity Score: {similarity}")
print(f"Keyword Score   : {keywords}")
print(f"Overall Score   : {overall}")

print("\nSuggestions:")

for item in feedback:
    print("-", item)

print("\nResults saved to results.csv")
print("==========================")