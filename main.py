import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


credits_frame = pd.read_csv("tmdb_5000_credits.csv")
movies_frame = pd.read_csv("tmdb_5000_movies.csv")
# print(movies_frame.head())
# print(credits_frame.head())

#merging the two datasets into one dataframe 
credits_frame.columns = ["id", "title", "cast", "crew"]
movies_frame = movies_frame.merge(credits_frame, on="id")


#Calculating a weighted rating based on the total number of votes and the average rating of each movie

av_vote_average= movies_frame["vote_average"].mean()
vote_count_min = movies_frame["vote_count"].quantile(0.9)
# print("Average vote count is : ", av_vote_average)
# print("90th percentile for number of votes is : ", vote_count_min)
#Removing movies with too few votes
new_movies_frame = movies_frame.copy().loc[movies_frame["vote_count"] >= vote_count_min]
print(new_movies_frame.shape)

#Calculating the weighted rating, accounting for the fact that the higher the numbe of votes, the more likely it is that the movie is popular and is actuallly a good one
#Rev a lil bit
def weighted_rating(movie, C=av_vote_average, m=vote_count_min):
    votes = movie["vote_count"]
    av_rating = movie["vote_average"]
    return (votes/(votes + m) * av_rating) + (m/(votes + m) * C)

new_movies_frame["score"] = new_movies_frame.apply(weighted_rating, axis=1)
new_movies_frame = new_movies_frame.sort_values('score', ascending=False)
#print(new_movies_frame[["title_y", "vote_count", "vote_average", "score"]].head(10))

#Content based filtering : using the data in the overview of the movie to predict similar movies 
# print(movies_frame["overview"].head(5))
#removing all stop words in the english language, for example the, then blah blah blah
tfidf = TfidfVectorizer(stop_words="english")
movies_frame["overview"] = movies_frame["overview"].fillna("")
tfidf_matrix = tfidf.fit_transform(movies_frame["overview"])
# print(tfidf_matrix.shape)

#Computing the similarity of movies within the database
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)#  making the similarity vector
# print(cosine_sim.shape)
indices = pd.Series(movies_frame.index, index=movies_frame["title_y"]).drop_duplicates()
# print(indices.head())



def get_recommendation(title, cosine_sim = cosine_sim ):
    index = indices[title]
    similarity_scores = list(enumerate(cosine_sim[index]))
    scores_ranked = sorted(similarity_scores, key= lambda x: x[1], reverse = True)
    top_ten = scores_ranked[1:11]
    movie_indices = [indice for indice,score in top_ten]
    movies = movies_frame["title_y"].iloc[movie_indices]


    return movies



#try and modify this so th\


