import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("CAPSTONE PROJECT 4 : AI BASED RECOMMENDATION SYSTEM")
print("=" * 70)

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("\nMovies Shape :", movies.shape)
print("Ratings Shape :", ratings.shape)

print("\nFirst Five Movies")

print(movies.head())

print("\nFirst Five Ratings")

print(ratings.head())

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

vectorizer = CountVectorizer(stop_words="english")

genre_matrix = vectorizer.fit_transform(movies["genres"])

similarity = cosine_similarity(genre_matrix)

print("\nSimilarity Matrix Shape :", similarity.shape)

movie_index = pd.Series(
    movies.index,
    index=movies["title"]
).drop_duplicates()

def recommend_movies(title, n=5):

    if title not in movie_index:

        print("Movie not found.")

        return

    index = movie_index[title]

    scores = list(enumerate(similarity[index]))

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:n+1]

    recommendations = []

    for i, score in scores:

        recommendations.append({

            "Movie": movies.iloc[i]["title"],

            "Genres": movies.iloc[i]["genres"],

            "Similarity": round(score,3)

        })

    return pd.DataFrame(recommendations)

movie_name = "Toy Story (1995)"

print("\nSelected Movie")

print(movie_name)

print("\nRecommended Movies")

result = recommend_movies(movie_name)

print(result)

ratings_summary = ratings.groupby("movieId")["rating"].agg(
    ["count","mean"]
).reset_index()

ratings_summary.rename(columns={
    "count":"Total Ratings",
    "mean":"Average Rating"
}, inplace=True)

movies = movies.merge(
    ratings_summary,
    on="movieId",
    how="left"
)

print("\nTop Rated Movies")

print(
    movies.sort_values(
        by="Average Rating",
        ascending=False
    )[[
        "title",
        "Average Rating",
        "Total Ratings"
    ]].head(10)
)

print("\nProject Completed Successfully.")