import pandas as pd
from fuzzywuzzy import process
import re
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter


geolocator = Nominatim(user_agent="beer_app")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)

def clean_text(text):
    
    text_before_comma = text.split(',', 1)[0]
    
    return re.sub(r'\W+', '', text_before_comma.lower())

beers_path = "/Users/michaelrodden/Georgia Tech/ISYE 7406/Project/final_beers.csv"
breweries_path = "/Users/michaelrodden/Georgia Tech/ISYE 7406/Project/breweries_new.csv"
beers_df = pd.read_csv(beers_path)
breweries_df = pd.read_csv(breweries_path)

beers_df['brewery_cleaned'] = beers_df['brewery'].apply(clean_text)
breweries_df['name_cleaned'] = breweries_df['name'].apply(clean_text)

breweries_grouped = breweries_df.groupby('name_cleaned').first().reset_index()

def add_lat_long(row):
    brewery_cleaned = row['brewery_cleaned']
    if brewery_cleaned in breweries_grouped['name_cleaned'].values:
        info = breweries_grouped[breweries_grouped['name_cleaned'] == brewery_cleaned].iloc[0]
        row['brewery_latitude'] = info['latitude']
        row['brewery_longitude'] = info['longitude']
    else:
        query = f"{row['city']}, {row['state']}"  
        location = geocode(query)
        if location:
            row['brewery_latitude'] = location.latitude
            row['brewery_longitude'] = location.longitude
        else:
            row['brewery_latitude'] = None
            row['brewery_longitude'] = None
    return row

beers_df = beers_df.apply(add_lat_long, axis=1)

output_path = "/Users/michaelrodden/Georgia Tech/ISYE 7406/Project/beers_new1.csv"
beers_df.to_csv(output_path, index=False)

print(f"Output saved to {output_path}")

