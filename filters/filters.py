## Parametry jsou jednotlive filtery

loader = MLDataLoader(ratings_path, movies_path, tags_path, links_path,
            [RatingLowFilter(4.0), MovieFilterByYear(1990), RatingFilterOld(2010), RatingsPerYearFilter(50.0), RatingUserFilter(100), RatedMovieFilter(), LinkFilter()],
            img_dir_path=img_dir_path, descriptions_path=descriptions_path
        )


# Fitlery se pak aplikuji ve stejnem poradi jako jsou v tom poli
for f in self.filters:
    f(self)


# Samotne filtery jsou nize



class RatingUserFilter:
    def __init__(self, min_ratings_per_user):
        self.min_ratings_per_user = min_ratings_per_user

    def __call__(self, loader):
        # First filter out users who gave <= 1 ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df['userId'].map(loader.ratings_df['userId'].value_counts()) >= self.min_ratings_per_user]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)
        print(f"Ratings shape after user filtering: {loader.ratings_df.shape}, n_users = {loader.ratings_df.userId.unique().size}, n_items = {loader.ratings_df.movieId.unique().size}")
        
# Filters out all low ratings
class RatingLowFilter:
    def __init__(self, min_rating):
        self.min_rating = min_rating
    def __call__(self, loader):
        loader.ratings_df = loader.ratings_df[loader.ratings_df.rating >= self.min_rating]
        loader.ratings_df = loader.ratings_df.reset_index(drop=True)


class RatedMovieFilter:
    def __call__(self, loader):
        # We are only interested in movies for which we hav
        loader.movies_df = loader.movies_df[loader.movies_df.movieId.isin(loader.ratings_df.movieId.unique())]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

# Filters out all ratings of movies that do not have enough ratings per year
class RatingsPerYearFilter:
    def __init__(self, min_ratings_per_year):
        self.min_ratings_per_year = min_ratings_per_year

    def __call__(self, loader):
        movies_df_indexed = loader.movies_df.set_index("movieId")

        # Add column with age of each movie
        movies_df_indexed.loc[:, "age"] = movies_df_indexed.year.max() - movies_df_indexed.year
        
        # Calculate number of ratings per year for each of the movies
        loader.ratings_df.loc[:, "ratings_per_year"] = loader.ratings_df['movieId'].map(loader.ratings_df['movieId'].value_counts()) / loader.ratings_df['movieId'].map(movies_df_indexed["age"])
        
        # Filter out movies that do not have enough yearly ratings
        loader.ratings_df = loader.ratings_df[loader.ratings_df.ratings_per_year >= self.min_ratings_per_year]

class MovieFilterByYear:
    def __init__(self, min_year):
        self.min_year = min_year
        
    def _parse_year(self, x):
        x = x.split("(")
        if len(x) <= 1:
            return 0
        try:
            return int(x[-1].split(")")[0])
        except:
            return 0

    def __call__(self, loader):
        # Filter out unrated movies and old movies
        # Add year column      
        loader.movies_df.loc[:, "year"] = loader.movies_df.title.apply(self._parse_year)
        loader.movies_df = loader.movies_df[loader.movies_df.year >= self.min_year]
        loader.movies_df = loader.movies_df.reset_index(drop=True)

class RatingFilterOld:
    def __init__(self, oldest_rating_year):
        self.oldest_rating_year = oldest_rating_year
    def __call__(self, loader):
        # Marker for oldest rating
        oldest_rating = datetime.datetime(year=self.oldest_rating_year, month=1, day=1, tzinfo=datetime.timezone.utc).timestamp()
        # Filter ratings that are too old
        loader.ratings_df = loader.ratings_df[loader.ratings_df.timestamp > oldest_rating]
        #loader.ratings_df = loader.ratings_df.reset_index(drop=True)


class LinkFilter:
    def __call__(self, loader):
        loader.links_df = loader.links_df[loader.links_df.index.isin((loader.movies_df.movieId))]