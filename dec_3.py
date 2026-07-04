import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import difflib

# ==========================================================
# LOAD DATASET
# ==========================================================
DATA_PATH = r"D:\projects\decode labs\anime.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Loaded Successfully")
print("Shape :", df.shape)
print("=" * 60)

# ==========================================================
# DATA CLEANING
# ==========================================================
df["synopsis"] = df["synopsis"].fillna("")
df["start_date"] = df["start_date"].fillna("Unknown")
df["end_date"] = df["end_date"].fillna("Unknown")
df["episodes"] = df["episodes"].fillna(0)
df["type"] = df["type"].fillna("Unknown")

# Remove duplicate titles if present
df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)

# ==========================================================
# NORMALIZE NUMERICAL FEATURES
# ==========================================================
scaler = MinMaxScaler()

df[["score_scaled",
    "members_scaled",
    "popularity_scaled"]] = scaler.fit_transform(
    df[["score", "members", "popularity"]]
)

# Lower popularity rank means better anime
df["popularity_scaled"] = 1 - df["popularity_scaled"]

# ==========================================================
# CREATE CONTENT FEATURES
# ==========================================================
df["content"] = (
    df["title"].astype(str) + " " +
    df["synopsis"].astype(str) + " " +
    df["type"].astype(str) + " " +
    df["episodes"].astype(str)
)

# ==========================================================
# TF-IDF VECTORIZATION
# ==========================================================
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=15000,
    ngram_range=(1, 2)
)

tfidf_matrix = vectorizer.fit_transform(df["content"])

print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

# ==========================================================
# COSINE SIMILARITY
# ==========================================================
similarity_matrix = cosine_similarity(tfidf_matrix)

# ==========================================================
# RECOMMENDATION FUNCTION
# ==========================================================
def recommend_anime(anime_name, top_n=10):

    anime_name_lower = anime_name.lower()

    exact_match = df[
        df["title"].str.lower() == anime_name_lower
    ]

    # Fuzzy matching if exact title not found
    if len(exact_match) == 0:

        possible_matches = difflib.get_close_matches(
            anime_name,
            df["title"].tolist(),
            n=5,
            cutoff=0.5
        )

        if len(possible_matches) == 0:
            print("\nAnime not found in dataset.")
            return

        print("\nDid you mean:")
        for i, match in enumerate(possible_matches):
            print(f"{i+1}. {match}")

        choice = int(input("\nChoose option number: "))
        anime_name = possible_matches[choice - 1]

    idx = df[df["title"] == anime_name].index[0]

    similarity_scores = list(
        enumerate(similarity_matrix[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    recommendations = []

    for anime_idx, similarity_score in similarity_scores[1:]:

        weighted_score = (
            similarity_score * 0.60 +
            df.iloc[anime_idx]["score_scaled"] * 0.20 +
            df.iloc[anime_idx]["members_scaled"] * 0.10 +
            df.iloc[anime_idx]["popularity_scaled"] * 0.10
        )

        recommendations.append(
            (
                anime_idx,
                weighted_score,
                similarity_score
            )
        )

    recommendations = sorted(
        recommendations,
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    print("\n")
    print("=" * 90)
    print(f"TOP {top_n} RECOMMENDATIONS FOR: {anime_name}")
    print("=" * 90)

    for rank_num, rec in enumerate(recommendations, start=1):

        anime_idx, final_score, similarity_score = rec
        row = df.iloc[anime_idx]

        print(f"\n#{rank_num}")
        print("-" * 90)
        print("Title            :", row["title"])
        print("Score            :", row["score"])
        print("Rank             :", row["rank"])
        print("Popularity Rank  :", row["popularity"])
        print("Members          :", format(int(row["members"]), ","))
        print("Episodes         :", int(row["episodes"]))
        print("Type             :", row["type"])
        print("Similarity Score :", round(similarity_score, 4))
        print("Final Score      :", round(final_score, 4))
        print("Start Date       :", row["start_date"])
        print("End Date         :", row["end_date"])
        print("Image URL        :", row["image_url"])

        synopsis = str(row["synopsis"])

        if len(synopsis) > 350:
            synopsis = synopsis[:350] + "..."

        print("\nSynopsis:")
        print(synopsis)

    print("\n" + "=" * 90)


# ==========================================================
# MAIN PROGRAM
# ==========================================================
while True:

    print("\n1. Recommend by Anime Name")
    print("2. Show Top Rated Anime")
    print("3. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":

        anime_name = input(
            "\nEnter Anime Name: "
        )

        top_n = input(
            "How many recommendations? (default 10): "
        )

        if top_n == "":
            top_n = 10
        else:
            top_n = int(top_n)

        recommend_anime(
            anime_name,
            top_n
        )

    elif choice == "2":

        top_anime = df.sort_values(
            by="score",
            ascending=False
        ).head(10)

        print("\nTop Rated Anime\n")

        for i, (_, row) in enumerate(
            top_anime.iterrows(),
            start=1
        ):
            print(
                f"{i}. {row['title']} "
                f"(Score: {row['score']})"
            )

    elif choice == "3":
        print("\nThank you for using the Anime Recommendation System.")
        break

    else:
        print("\nInvalid Choice. Please try again.")