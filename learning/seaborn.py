import pandas as pd
from matplotlib import pyplot as plt
#%matplotlib inline
import seaborn as sns

df = pd.read_csv('Pokemon.csv', index_col=0)
df.head()

# Recommended way
sns.lmplot(x='Attack', y='Defense', data=df)

# Alternative way
#sns.lmplot(x=df.Attack, y=df.Defense)
# Tweak using Matplotlib
plt.ylim(0, None)
plt.xlim(0, None)

# Scatterplot arguments
sns.lmplot(x='Attack', y='Defense', data=df,
           fit_reg=False, # No regression line
           hue='Stage')   # Color by evolution stage


# Boxplot
sns.boxplot(data=df)

# Pre-format DataFrame
stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)
 
# New boxplot using stats_df
sns.boxplot(data=stats_df)

# Set theme
sns.set_style('whitegrid')
 
# Violin plot -alternative to Box
sns.violinplot(x='Type 1', y='Attack', data=df)

pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

# Swarm plot with Pokemon color palette
sns.swarmplot(x='Type 1', y='Attack', data=df, 
              palette=pkmn_type_colors)

Plot overlay

# Set figure size with matplotlib
plt.figure(figsize=(10,6))
 
# Create plot
sns.violinplot(x='Type 1',
               y='Attack', 
               data=df, 
               inner=None # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='Type 1', 
              y='Attack', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Attack by Type')


Data Wrangling

df.head()
print(df.shape)

# Melt DataFrame
melted_df = pd.melt(stats_df, 
                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep
                    var_name="Stat") # Name of melted variable
melted_df.head()
print(melted_df.shape)

# Swarmplot with melted_df
sns.swarmplot(x='Stat', y='value', data=melted_df, 
              hue='Type 1')


# 1. Enlarge the plot
plt.figure(figsize=(10,6))
 
sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type 1', 
              split=True # 2. Separate points by hue
              palette=pkmn_type_colors) # 3. Use Pokemon palette
 
# 4. Adjust the y-axis
plt.ylim(0, 260)
 
# 5. Place legend to the right
plt.legend(bbox_to_anchor=(1, 1), loc=2)

Heatmaps help you visualize matrix-like data.
# Calculate correlations
corr = stats_df.corr()
# Heatmap
sns.heatmap(corr)

Histograms allow you to plot the distributions of numeric variables.
# Distribution Plot (a.k.a. Histogram)
sns.distplot(df.Attack)

Bar plots help you visualize the distributions of categorical variables.
# Count Plot (a.k.a. Bar Plot)
sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)
# Rotate x-labels
plt.xticks(rotation=-45)

Factor plots make it easy to separate plots by categorical classes.
# Factor Plot
g = sns.factorplot(x='Type 1', 
                   y='Attack', 
                   data=df, 
                   hue='Stage',  # Color by stage
                   col='Stage',  # Separate by stage
                   kind='swarm') # Swarmplot
# Rotate x-axis labels
g.set_xticklabels(rotation=-45)
# Doesn't work because only rotates last plot
# plt.xticks(rotation=-45)

Density plots display the distribution between two variables.
# Density Plot
sns.kdeplot(df.Attack, df.Defense)

Joint distribution plots combine information from scatter plots and histograms to give you detailed information for bi-variate distributions.
# Joint Distribution Plot
sns.jointplot(x='Attack', y='Defense', data=df)