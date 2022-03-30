import pandas as pd
import numpy as np

# Location
# Read world cities database from https://simplemaps.com/data/world-
# Can help us get consistent location
worldcities = pd.read_csv("data/worldcities.csv")
train = pd.read_csv("data/train.csv")

"""
Filtering to make matches easier to find, and to reduce duplications:
    - Population > 500000
    - Not in China
"""

filt_cities = worldcities.loc[(worldcities["population"] > 500000) &
                              (worldcities["country"] != "China"),
                       ["city_ascii", "country", "capital", "population"]]

# Convert to lower case
lower = filt_cities["city_ascii"].str.lower()
filt_cities["city_lower"] = lower
filt_cities["country_lower"] = filt_cities["country"].str.lower()

all_words = lower.str.split(" ", expand=True)

initials = pd.Series(data = "", index=all_words.index)
for c in all_words.columns:
    s = all_words[c].str[0]
    s = np.where(s.isna(), "", s)
    
    initials += s

filt_cities["initial"] = np.where(initials.str.len() > 1, 
                                  initials, "")

# Remove any cities with the same name in different countries. 
filt_cities.loc[:, "count"] = filt_cities.groupby('city_lower')['city_lower']. \
    transform('count')

filt_cities = filt_cities.loc[filt_cities["count"] == 1, :]

def get_unq_location(location):
    locations = pd.DataFrame({"location":location.unique()})
    locations["location_lower"] = locations["location"].str.lower()
    
    return locations

locations = get_unq_location(train["location"])

# Pandas really doesn't have cross join???
def cross_join(x, y):
    x = x.copy().drop_duplicates()
    y = y.copy().drop_duplicates()
    
    x.loc[:, "join"] = 1
    y.loc[:, "join"] = 1

    return x.merge(y, on = "join").drop("join", axis=1)


def get_matches(locations, place_df, place_col):
    cj = cross_join(locations, place_df[[place_col]]).dropna()
    
    cj["match"] = cj.apply(lambda x: x[place_col] in x["location_lower"], 
                                   axis=1)
    
    # filter matches
    matches = cj.loc[cj["match"], ["location_lower", place_col]].drop_duplicates()
    
    # remove any cases where more than 1 place is matched
    matches.loc[:, 'count'] = matches.groupby('location_lower') \
        ['location_lower'].transform('count')
    
    return matches.loc[matches["count"] == 1, ["location_lower", place_col]]

city_matches = get_matches(locations, filt_cities, "city_lower")
country_matches = get_matches(locations, filt_cities, "country_lower")

# check we don't have duplicates
city_matches["location_lower"].value_counts()
country_matches["location_lower"].value_counts()

cities_both = city_matches.merge(filt_cities[["city_lower", "country_lower"]], how="left",
                   on = "city_lower")

# Check duplicates
cities_both[["location_lower", "city_lower"]].value_counts()
cities_both[["location_lower", "country_lower"]].value_counts()

# TODO: Try and make it so that it extracts full place name only. at the moment
# e.g. instagram returns agra

locations_out = cities_both.merge(country_matches, how="outer", 
                  on = "location_lower", suffixes = [".city", ".country"])

locations_out["both"] = (~locations_out["country_lower.city"].isna()) & \
                        (~locations_out["country_lower.country"].isna())

df = locations_out[locations_out["both"]]

from IPython.core.display import HTML
display(HTML(df.to_html()))
