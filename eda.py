import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import geopandas as gpd

# Load the data
@st.cache_data
def load_data():
    data = pd.read_csv('/Users/michaelrodden/Georgia Tech/ISYE 7406/Project/beers_new1.csv')
    return data

data = load_data()

# Determine the 15 most common beer styles
top_styles = data['style'].value_counts().nlargest(15).index

# Filter data to include only the top 15 styles
filtered_data = data[data['style'].isin(top_styles)]

# Set the aesthetics for seaborn plots
sns.set(style="whitegrid")

# Streamlit page configuration
st.title('Exploratory Data Analysis of Beer Dataset')

# Plot 1: Histogram of ABV (Alcohol by Volume)
fig1, ax1 = plt.subplots()
sns.histplot(data['abv'].dropna(), kde=True, color='blue', ax=ax1)
ax1.set_title('Distribution of Alcohol by Volume (ABV)')
ax1.set_xlabel('ABV (%)')
ax1.set_ylabel('Frequency')
st.pyplot(fig1)

# Plot 2: Box Plot of IBU for the top 15 most common beer styles
fig2, ax2 = plt.subplots()
sns.boxplot(x='ibu', y='style', data=filtered_data.dropna(subset=['ibu']), ax=ax2)
ax2.set_title('International Bitterness Units (IBU) by Beer Style for Top 15 Styles')
ax2.set_xlabel('IBU')
ax2.set_ylabel('Beer Style')
st.pyplot(fig2)

# Plot 3: Count Plot for Beer Styles
fig3, ax3 = plt.subplots()
sns.countplot(y='style', data=filtered_data, order=filtered_data['style'].value_counts().index, ax=ax3)
ax3.set_title('Count of Beers by Style')
ax3.set_xlabel('Count')
ax3.set_ylabel('Beer Style')
st.pyplot(fig3)

# Plot 4: Scatter Plot of Brewery Locations
fig4, ax4 = plt.subplots()
sns.scatterplot(x='brewery_longitude', y='brewery_latitude', data=data, s=30, ax=ax4)
ax4.set_title('Geographical Distribution of Breweries')
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
handles, labels = ax4.get_legend_handles_labels()
ax4.legend(handles=handles[0:], labels=labels[0:], bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=3)
st.pyplot(fig4)

# Plot 5: Bar Plot of Average ABV by State with improved layout
fig5, ax5 = plt.subplots(figsize=(14, 8))  # Increased width for better label display
state_abv = data.groupby('state')['abv'].mean().sort_values(ascending=True).reset_index()
sns.barplot(x='state', y='abv', data=state_abv, ax=ax5)
ax5.set_title('Average Alcohol by Volume (ABV) by State')
ax5.set_xlabel('State')
ax5.set_ylabel('Average ABV (%)')
plt.xticks(rotation=90)  # Rotate state labels for better readability
st.pyplot(fig5)

# Calculate style dominance
style_counts = data.groupby(['state', 'style']).size().reset_index(name='count')
total_counts = data.groupby(['state']).size().reset_index(name='total')
dominance = style_counts.merge(total_counts, on='state')
dominance['percentage'] = dominance['count'] / dominance['total'] * 100

# Find states where a style is over 30% dominant
dominance_over_30 = dominance[dominance['percentage'] > 25]

# Preparing data for plotting
max_dominance = dominance_over_30.groupby('state')['percentage'].max().reset_index()
max_dominance.sort_values(by='percentage', ascending=False, inplace=True)

# Plotting
fig, ax = plt.subplots(figsize=(12, 8))
# Sort data to have the longest bar at the top
sorted_data = max_dominance.sort_values(by='percentage', ascending=False)
bars = ax.barh(sorted_data['state'], sorted_data['percentage'], color='lightgrey')  # Use barh for horizontal bars

# Highlight bars over 30% in a different color
for bar, pct in zip(bars, sorted_data['percentage']):
    if pct > 25:
        bar.set_color('gold')  # Changed color to dark gold for emphasis

# Annotate with beer style
for index, row in sorted_data.iterrows():
    dominant_style = dominance_over_30[(dominance_over_30['state'] == row['state']) & 
                                       (dominance_over_30['percentage'] == row['percentage'])]['style'].values[0]
    ax.text(4, index, f'{dominant_style}', va='center', color='black', ha='center')  # Align text just to the right of the y-axis

ax.set_ylabel('States')
ax.set_xlabel('Maximum Style Dominance (%)')
ax.set_title('States with a Dominant Beer Style over 25% of Market')
plt.tight_layout()

# Show in Streamlit
st.pyplot(fig)
