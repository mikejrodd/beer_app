import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances
import numpy as np
import streamlit as st
import pydeck as pdk
from math import radians, cos, sin, asin, sqrt

def clean_data(filepath):
    df = pd.read_csv(filepath)
    df.loc[df['abv'] < 0.01, 'cat_name'] = "Non Alcoholic"
    df.sort_values(by='brewery_id', inplace=True)
    df['style'] = df['style'].str.title()
    return df

def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 3956  
    return c * r

def cluster_or_locations(df, category):
    category_df = df[df['style'] == category].dropna(subset=['brewery_latitude', 'brewery_longitude'])
    if len(category_df) >= 7:
        kmeans = KMeans(n_clusters=min(7, len(category_df)//7), random_state=0).fit(category_df[['brewery_latitude', 'brewery_longitude']])
        category_df['cluster'] = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        avg_dists = []
        for center in centers:
            center_lat, center_lon = center

            cluster_index = np.where(np.all(centers == center, axis=1))[0][0]
            cluster_breweries = category_df[category_df['cluster'] == cluster_index]
            
            distances = [haversine(center_lat, center_lon, lat, lon) for lat, lon in zip(cluster_breweries['brewery_latitude'], cluster_breweries['brewery_longitude'])]
            avg_dists.append(np.mean(distances))
        
        try:
            score = silhouette_score(category_df[['brewery_latitude', 'brewery_longitude']], kmeans.labels_)
            return category_df, centers, avg_dists, True, score
        except ValueError:
            return category_df, centers, avg_dists, True, None
    return category_df, None, None, False, None

def display_cluster_info_table(centers, avg_dists):
    if centers is not None:
        cluster_info = pd.DataFrame(centers, columns=['Latitude', 'Longitude'])
        cluster_info['Cluster ID'] = ['Cluster ' + str(i+1) for i in range(len(centers))]
        cluster_info['Average Distance'] = avg_dists
        cluster_info = cluster_info[['Cluster ID', 'Latitude', 'Longitude', 'Average Distance']]
        st.table(cluster_info)

def visualize_on_map(df, centers, is_clustered, brewery_location=None):
    cluster_colors = [
        [255, 0, 0, 160],
        [0, 0, 255, 160],
        [204, 204, 0, 160],
        [100, 100, 100, 160],
        [255, 0, 255, 160],
    ]
    
    if is_clustered:
        df['color'] = df['cluster'].apply(lambda x: cluster_colors[x % len(cluster_colors)])
    else:
        df['color'] = [0, 0, 255, 160]
    
    initial_view_state = pdk.ViewState(
        latitude=brewery_location['brewery_latitude'] if brewery_location is not None else 35.257160,
        longitude=brewery_location['brewery_longitude'] if brewery_location is not None else -95.995102,
        zoom=7 if brewery_location is not None else 2.5,
        pitch=0,
    )
    
    layers = [
        pdk.Layer(
            'ScatterplotLayer',
            data=df,
            get_position=['brewery_longitude', 'brewery_latitude'],
            get_color='color',
            get_radius=20000,
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=pd.DataFrame(centers, columns=['latitude', 'longitude']) if centers is not None else pd.DataFrame(),
            get_position=['longitude', 'latitude'],
            get_color=[30, 200, 0, 160],
            get_radius=80000,
        ),
    ]
    
    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=initial_view_state,
        layers=layers,
    ))

def calculate_silhouette_scores(df):
    styles = df['style'].unique()
    scores = {}
    for style in styles:
        category_df = df[df['style'] == style].dropna(subset=['brewery_latitude', 'brewery_longitude'])
        if len(category_df) >= 7:
            kmeans = KMeans(n_clusters=min(7, len(category_df) // 7), random_state=0).fit(
                category_df[['brewery_latitude', 'brewery_longitude']])
            try:
                score = silhouette_score(category_df[['brewery_latitude', 'brewery_longitude']], kmeans.labels_)
                scores[style] = score
            except ValueError:
                continue
    return scores

def display_scores_table(scores):
    scores_df = pd.DataFrame(list(scores.items()), columns=['Beer Style Clusters', 'Silhouette Score'])
    scores_df.sort_values('Silhouette Score', ascending=False, inplace=True)
    st.table(scores_df)

def main():
    st.title('Beer Style Clusters in the US')
    
    df = clean_data('/Users/michaelrodden/Georgia Tech/ISYE 7406/Project/beers_new1.csv')  
    
    categories = df['style'].unique()
    default_index = list(categories).index('American Ipa') if 'American Ipa' in categories else 0
    
    category = st.sidebar.selectbox('Select a Beer Category', categories, index=default_index)
    
    category_df, centers, avg_dists, is_clustered, score = cluster_or_locations(df, category)
    
    brewery_list = category_df.sort_values('brewery')['brewery'].unique().tolist()
    brewery_name = st.sidebar.selectbox('Select a Brewery to View', [''] + brewery_list)
    
    if brewery_name:
        brewery_location = category_df[category_df['brewery'] == brewery_name][['brewery_latitude', 'brewery_longitude']].iloc[0].to_dict()
        brewery_beers = df[(df['brewery'] == brewery_name) & (df['style'] == category)]
        for _, beer in brewery_beers.iterrows():
            st.sidebar.write(f"{beer['label'].split('(')[0]} - {beer['style']} - ABV: {round(beer['abv']*100, 1)}%")
        
        
        visualize_on_map(category_df, centers, is_clustered, brewery_location)
    else:
        visualize_on_map(category_df, centers, is_clustered)
    
    display_cluster_info_table(centers, avg_dists)
    
    all_scores = calculate_silhouette_scores(df)
    display_scores_table(all_scores)

if __name__ == '__main__':
    main()
