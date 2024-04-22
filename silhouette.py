import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data for silhouette scores and beer styles
data = {
    "Beer Style Clusters": [
        "Cider", "American Adjunct Lager", "Scotch Ale / Wee Heavy", "Oatmeal Stout",
        "Winter Warmer", "Vienna Lager", "American Pale Lager", "American Double / Imperial IPA",
        "American Black Ale", "Belgian Pale Ale", "Scottish Ale", "American Pale Wheat Ale",
        "American Amber / Red Lager", "Märzen / Oktoberfest", "Pumpkin Ale", "American Pilsner",
        "German Pilsener", "American Strong Ale", "Munich Helles Lager", "Czech Pilsener",
        "Fruit / Vegetable Beer", "English Brown Ale", "Saison / Farmhouse Ale", "American IPA",
        "American Pale Ale (APA)", "American Amber / Red Ale", "Kölsch", "American Stout",
        "American Porter", "American Blonde Ale", "Hefeweizen", "Extra Special / Strong Bitter (ESB)",
        "Cream Ale", "Witbier", "Rye Beer", "American Brown Ale"
    ],
    "Silhouette Score": [
        0.8078, 0.7659, 0.7621, 0.7389, 0.6785, 0.6704, 0.6626, 0.6620, 0.6609,
        0.6585, 0.6520, 0.6296, 0.6226, 0.6215, 0.6193, 0.6002, 0.5907, 0.5786,
        0.5776, 0.5708, 0.5594, 0.5564, 0.5427, 0.5411, 0.5357, 0.5355, 0.5338,
        0.5317, 0.5236, 0.5102, 0.4956, 0.4844, 0.4809, 0.4738, 0.4587, 0.4469
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Sort DataFrame by Silhouette Score in descending order
df_sorted = df.sort_values(by='Silhouette Score', ascending=False)

# Plotting
plt.figure(figsize=(16, 8))
sns.scatterplot(x='Silhouette Score', y='Beer Style Clusters', data=df_sorted, color='blue')
plt.title('Silhouette Scores by Beer Style Cluster', fontsize=16)
plt.xlabel('Silhouette Score', fontsize=14)
plt.ylabel('Cluster', fontsize=14)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.tight_layout() 
plt.grid(False)

plt.show()
