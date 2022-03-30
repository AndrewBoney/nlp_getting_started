import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocess_text(text):
    # Make lower case
    text = text.str.lower()

    # Replace mentions with "@mention"
    text = text.str.replace(r"(@\S+)", " @mention", regex=True)

    # Replace link with "?link"
    text = text.str.replace(r"(http://\S+)|(https://\S+)", " ?link", regex=True)

    # Replace \n with space
    text = text.str.replace("\n", " ", regex=False)
    
    return text


def do_padding(sentences, tokenizer, maxlen, padding, truncating):
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(sequences, maxlen=maxlen, padding=padding, truncating=truncating)
    
    return padded, sequences

# Convert keywords to array of shape (, 3)
def process_keyword(keywords):
    keywords = keywords.str.lower()
    keywords = keywords.str.replace("%20", " ")
    keywords = np.where(keywords.isna(), "", keywords)
    
    return keywords

# Convert top 30 locations to sparse array of 1/0 flags.
# This isn't a great way to do this. It would be much better to preprocess
# this into e.g. [country, city] per row
def get_location_cols(location):
    location_counts = location.value_counts()
    return location_counts.index[:30]

def location_dummies(location, cols):
    locations = np.select([location.isin(cols), location.isna()],
                          [location, "missing"],
                          "other")
    
    location_dummies = pd.get_dummies(locations)
    
    df_cols = location_dummies.columns
    
    for c in cols:
        col_exist = c in (df_cols)
        
        if ~col_exist:
            location_dummies.loc[:, c] = 0
        
    return location_dummies[list(cols) + ["missing", "other"]]
          
def convert_cities(df, cities = ['missing', 'new york', 'london', 'cali', 
       'washington', 'los angeles', 'chicago', 'atlanta', 'san francisco', 
       'mumbai', 'toronto', 'manchester', 'calgary', 'seattle', 'sydney', 
       'denver', 'melbourne','houston', 'dallas', 'lagos', 'vancouver']):
    
    df = df.copy()
    
    df.loc[df["city"].isna(), "city"] = "missing"
    df.loc[~df["city"].isin(cities), "city"] = "other"
    
    return df

def convert_countries(df, countries = ['missing', 'united states', 
       'united kingdom', 'canada', 'india', 'australia', 'colombia', 
       'nigeria', 'kenya', 'indonesia', 'ireland',
       'south africa', 'japan', 'pakistan', 'philippines', 'brazil',
       'netherlands', 'germany', 'italy', 'spain', 'france']):
    
    df = df.copy()
    
    df.loc[df["country"].isna(), "country"] = "missing"
    df.loc[~df["country"].isin(countries), "country"] = "other"
    
    return df

def get_extra(df):
    df = df.copy()
    
    df["mention"] = df["text"].str.match(r"(@\S+)").astype(int)
    df["link"] = df["text"].str.match(r"(http://\S+)|(https://\S+)").astype(int)
    df["hashtag"] = df["text"].str.contains("#").astype(int)

    return df

