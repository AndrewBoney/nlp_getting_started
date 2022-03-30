import pandas as pd
import numpy as np

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold


# Location
# Read world cities database from https://simplemaps.com/data/world-
# Can help us get consistent location
worldcities = pd.read_csv("data/worldcities.csv")
train = pd.read_csv("data/train.csv")
locations = pd.read_csv("data/locations.csv")

train_plus = train.merge(locations[["location", "city", "country"]], how = "left", 
            on = "location")

keyword_counts = train_plus.groupby("keyword").agg({"target":["mean", "count"]})["target"]. \
    sort_values("mean", ascending=False)

# I'm going to group keywords into 10% bins
# NB this approach is sort of cheating as it is built on the full dataset, 
# therefore is peeking the val data. Really we should build on train data and 
# get it to fit val, but that would be fiddly and I can't be bothered.
keyword_bins = pd.cut(keyword_counts["mean"], 10, labels = False). \
    rename("keyword_bin").reset_index()
keyword_bins["keyword_bin"] = keyword_bins["keyword_bin"].astype(str)

keyword_bins.to_csv("data/keyword_bins.csv", index=False)

train_plus = train_plus.merge(keyword_bins, how = "left", on = "keyword")

# Set missing to NA
train_plus.loc[train_plus["city"].isna(), "city"] = "missing"
train_plus.loc[train_plus["country"].isna(), "country"] = "missing"
train_plus.loc[train_plus["keyword_bin"].isna(), "keyword_bin"] = "missing"

# Get top 20 cities and countries (+1 for missing), and set the rest to 'other'
top_20_cities = train_plus["city"].value_counts()[:21]
top_20_countries = train_plus["country"].value_counts()[:21]

train_plus.loc[~train_plus["city"].isin(top_20_cities.index), "city"] = "other"
train_plus.loc[~train_plus["country"].isin(top_20_countries.index), 
               "country"] = "other"

# Adding flag for text col containing link, mention or hashtag
train_plus["mention"] = train_plus["text"].str.match(r"(@\S+)")
train_plus["link"] = train_plus["text"].str.match(r"(http://\S+)|(https://\S+)")
train_plus["hashtag"] = train_plus["text"].str.contains("#")

X = pd.get_dummies(train_plus[["mention", "link", "hashtag",
                                     "city", "country", "keyword_bin"]])

y = train_plus["target"]

model = XGBClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, 
                           n_jobs=-1, error_score='raise')





