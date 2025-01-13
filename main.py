import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import squarify
# ------------------------------------------------------------
# Cleaning IMDb Top 1000 Dataset
# ------------------------------------------------------------

# Reading the CSV files for IMDb dataset
imdb_df = pd.read_csv(r'C:\Users\Farhan Butt\OneDrive\Desktop\IMDB Individual Project\archive (5).zip')

# Standardize column names (to remove any spaces)
imdb_df.columns = imdb_df.columns.str.strip()

# Drop the 'Poster_Link' and 'Overview' columns
if 'Poster_Link' in imdb_df.columns:
    imdb_df.drop(columns=['Poster_Link'], inplace=True)
if 'Overview' in imdb_df.columns:
    imdb_df.drop(columns=['Overview'], inplace=True)

# Check for missing values in the dataset
missing_values = imdb_df.isnull().sum()

# Display the columns with missing values and their counts
print("Missing values per column before cleaning:")
print(missing_values)

# Drop rows where 'Gross' has missing values
imdb_df.dropna(subset=['Gross'], inplace=True)

# Check the first few rows of the dataset
print(imdb_df.head(5))

# Check the updated 'Gross' values after dropping missing rows
print("Updated 'Gross' values after dropping missing rows:")
print(imdb_df['Gross'].head(5))  # Display the first 10 rows of the 'Gross' column

# Check for missing values after dropping rows with missing 'Gross' values
updated_missing_values = imdb_df.isnull().sum()

# Check on how to fill meta score values (Graph first to check skewness)
# Plot histogram for Meta Score to check skewness of the graph
plt.figure(figsize=(10, 8))
plt.hist(imdb_df['Meta_score'].dropna(), bins=20, edgecolor='black')
plt.title('Distribution of Meta Score')
plt.xlabel('Meta Score')
plt.ylabel('Frequency')
plt.show()

# Calculate the median of 'Meta_score'
meta_median = imdb_df['Meta_score'].median()

# Fill missing 'Meta_score' values with the median
imdb_df['Meta_score'] = imdb_df['Meta_score'].fillna(meta_median)

# Display the updated missing values per column
updated_missing_values = imdb_df.isnull().sum()
print("Updated missing values per column after filling 'Meta_score' with median:")
print(updated_missing_values)

# Updating the missing values for the certificate column with unknown
imdb_df['Certificate'] = imdb_df['Certificate'].fillna('Unknown')

# Check the updated missing values per column
updated_missing_values = imdb_df.isnull().sum()
print("Updated missing values per column after filling 'Certificate':")
print(updated_missing_values)

# --------------------------------------------------------------------------
# filling Apollo19 release year 
# Checking to see if Apollo13 exists
apollo_19_index = imdb_df[imdb_df['Series_Title'] == 'Apollo 13'].index

if not apollo_19_index.empty:
    # Update the 'Released_Year' column for the row
    imdb_df.loc[apollo_19_index, 'Released_Year'] = 1969
    print("Year updated for Apollo 13.")
else:
    print("Apollo 13 not found in the dataset.")

# Verify the change
print(imdb_df[imdb_df['Series_Title'] == 'Apollo 13'])
# ---------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# Fixing special characters in stars' names
# defining a function to clean the actors names 
def clean_name(name):
    if pd.isna(name):  # Checking if value is NAN
        return name
    # Using regex to remove special characters and numbers
    return re.sub(r'[^a-zA-Z\s]', '', name)

# Applied the cleaning function to each of the actor columns
actor_columns = ['Star1', 'Star2', 'Star3', 'Star4']
for col in actor_columns:
    imdb_df[col] = imdb_df[col].apply(clean_name)

# Verifying the cleaned actor names
print("Cleaned actor names:")
print(imdb_df[actor_columns].head(5))
# ----------------------------------------------------------------------------

# Removing duplicate rows
imdb_df.drop_duplicates(inplace=True)
print("duplicates removed if any")
# Check the first few rows of the dataset after changes
print(imdb_df.head(5))

# Check the columns of the dataset after handling missing values
print("Columns after cleaning:")
print(imdb_df.columns)

# ----------------------------------------------------------------------------
# Cleaning special characters in the movie titles (Series_Title column)
# defining a function to clean movie titles
def clean_title(title):
    if pd.isna(title):  # Checking if value is NAN
        return title
    # Using regex to remove special characters and numbers, keeping only alphabets and spaces
    return re.sub(r'[^a-zA-Z\s]', '', title)

# Applyinggd the cleaning function to the 'Series_Title' column to remove special characters
imdb_df['Series_Title'] = imdb_df['Series_Title'].apply(clean_title)

# Checking if there are any titles with special characters left
special_characters_in_titles = imdb_df[imdb_df['Series_Title'].str.contains(r'[^a-zA-Z\s]', regex=True)]

# If there are no special characters, printing a confirmation
if special_characters_in_titles.empty:
    print("All movie titles have been cleaned successfully. No special characters remaining.")
else:
    print(f"There are still {special_characters_in_titles.shape[0]} movie titles with special characters.")
#---------------------------------------------------------------------------------

# Check the first few rows of the cleaned 'Series_Title' column
print("Cleaned 'Series_Title' column:")
print(imdb_df['Series_Title'].head(5))


# ------------------------------------------------------------
# Cleaning Streaming Platform Dataset
# ------------------------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re

# Reading the CSV files for IMDb dataset
streaming_df = pd.read_csv(r'C:\Users\Farhan Butt\OneDrive\Desktop\IMDB data-set2\archive (6).zip')

# Check for missing values
missing_values = streaming_df.isnull().sum()
print("Missing values per column:")
print(missing_values)

# Drop the 'Age' column because it already exists in dataset1
streaming_df.drop(columns=['Age'], inplace=True)

# Verifying the column has been dropped
print("Columns after dropping 'Age':")
print(streaming_df.columns)

# dropping type column because its unnecessary 
streaming_df.drop(columns=['Type'], inplace=True)
print("column type has been dropped")

# updated missing values
updated_missing_values = streaming_df.isnull().sum()
print("Missing values per column after dropping 'Age':")
print(updated_missing_values)

# dropping rows of movies which have missing rotten tomato ratings 
streaming_df.dropna(subset=['Rotten Tomatoes'], inplace=True)
# verifying that the values have dropped 
print("missing rotten tomatoes have been dropped")
print("Updated missing values per column:")
print(streaming_df.isnull().sum())

# renaming the column series_title to title for help during merge
streaming_df.rename(columns={'Title': 'Series_Title'}, inplace=True)
print("column name changed")
# ----------------------------------------------------------------------------- 
# dropping any duplicates if any 
streaming_df.drop_duplicates(subset=['Series_Title'], inplace=True)

# Verifying that duplicates have been dropped
print("Duplicates based on 'Series_Title' have been dropped. Updated dataset:")
# ------------------------------------------------------------------------------ 
# Cleaning special characters in the movie titles (Series_Title column)
# Defining a function to clean movie titles
def clean_title(title):
    if pd.isna(title):  # Checking if value is NAN
        return title
    # Using regex to remove special characters and numbers, keeping only alphabets and spaces
    return re.sub(r'[^a-zA-Z\s]', '', title)

# Apply the cleaning function to the 'Series_Title' column to remove special characters
streaming_df['Series_Title'] = streaming_df['Series_Title'].apply(clean_title)

# Check if there are any titles with special characters left
special_characters_in_titles = streaming_df[streaming_df['Series_Title'].str.contains(r'[^a-zA-Z\s]', regex=True)]

# If there are no special characters, print a confirmation
if special_characters_in_titles.empty:
    print("All movie titles have been cleaned successfully. No special characters remaining.")
else:
    print(f"There are still {special_characters_in_titles.shape[0]} movie titles with special characters.")

# Display the first 10 rows of the cleaned 'Series_Title' column
print("First 10 rows of the cleaned 'Series_Title' column:")
print(streaming_df['Series_Title'].head(10))

# ---------------------------------------------------------------------------- 
# Display the first 10 rows of the updated dataset
print("First 10 rows of the updated dataset:")
print(streaming_df.head(10))

# ------------------------------------------------------------
#  merging the datasets
# ------------------------------------------------------------

# Merge the two datasets on 'Series_Title' 
merged_df = pd.merge(imdb_df, streaming_df, on='Series_Title', how='inner')  # 'inner' keeps only matching rows

# Counting how many Series_Title are the same in both datasets (after merging)
matching_titles_count = merged_df.shape[0]

# Print the result
print(f"Number of matching 'Series_Title' in both IMDb and Streaming platform datasets: {matching_titles_count}")


# Check the merged dataset
print("First 10 rows of the merged dataset:")
print(merged_df.head(10))

# Visulisations
all_stars = pd.concat([merged_df['Star1'], merged_df['Star2'], merged_df['Star3'], merged_df['Star4']])
top_stars = all_stars.value_counts().head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_stars.values, y=top_stars.index, palette='mako')
plt.title('Top 10 Stars with the Most Movies', fontsize=16)
plt.xlabel('Number of Movies', fontsize=12)
plt.ylabel('Stars', fontsize=12)
plt.show()

#Visulisation 2: which platform has the most imdb movies
# Step 1: Define the platform columns
platform_columns = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

# Step 2: Reshape the platform columns to long format
platform_df = merged_df.melt(id_vars=['Series_Title'],  # Reshape by movie titles
                             value_vars=platform_columns,  # Columns for platforms
                             var_name='Platform', 
                             value_name='Available')

# Step 3: Filter rows where 'Available' is 1 (indicating the movie is available on that platform)
platform_df = platform_df[platform_df['Available'] == 1]

# Step 4: Count the number of movies available on each platform
platform_counts = platform_df['Platform'].value_counts()

# Step 5: Visualize the number of movies on each platform
plt.figure(figsize=(10, 6))  
sns.barplot(x=platform_counts.index, y=platform_counts.values, palette='mako')

# Step 6: Add title and axis labels with better formatting
plt.title('Number of Movies Available on Each Streaming Platform', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Platform', fontsize=12, weight='bold', color='darkblue')
plt.ylabel('Number of Movies', fontsize=12, weight='bold', color='darkblue')

# Step 7: Rotate the X-axis labels to prevent overlap and graph presented
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()

# VISULISATION 3 rotten tomatoes against movie scatter plot
# Step 1: Sort the merged dataframe by Rotten Tomatoes ratings
sorted_df = merged_df.sort_values(by='Rotten Tomatoes', ascending=False)

# Step 2: Select top 10 movies with the highest Rotten Tomatoes ratings
top_10_rt = sorted_df[['Series_Title', 'Rotten Tomatoes']].head(10)

# Step 3: Create a scatter plot to show the relationship between Rotten Tomatoes and Series Title
plt.figure(figsize=(12, 6))
sns.scatterplot(x='Series_Title', y='Rotten Tomatoes', data=top_10_rt, color='purple', s=100, marker='o')

# Step 4: Add title and labels with better formatting
plt.title('Top 10 Movies with the Highest Rotten Tomatoes Ratings', fontsize=16, weight='bold', color='darkblue')
plt.xlabel('Series Title', fontsize=12, weight='bold', color='darkblue')
plt.ylabel('Rotten Tomatoes Rating', fontsize=12, weight='bold', color='darkblue')

# Step 5: Rotate X-axis labels to prevent overlap and adjust for readability
plt.xticks(rotation=45, ha='right', fontsize=10)

# Step 6: Show the plot
plt.tight_layout()  # Ensure everything fits without overlap
plt.show()


# # visulisation 4
# Step 1: Clean data to ensure 'IMDB_Rating' and platform columns are numeric
merged_df['IMDB_Rating'] = pd.to_numeric(merged_df['IMDB_Rating'], errors='coerce')

# Step 2: Remove rows with missing values for IMDB_Rating or platforms
merged_df_clean = merged_df.dropna(subset=['IMDB_Rating'])

# Step 3: Define platform columns (e.g., Netflix, Hulu, Prime Video, Disney+)
platform_columns = ['Netflix', 'Hulu', 'Prime Video', 'Disney+']

# Step 4: Filter for high IMDB ratings (e.g., movies with IMDB rating > 7.5)
high_rated_movies = merged_df_clean[merged_df_clean['IMDB_Rating'] > 7.5]

# Step 5: Reshape the platform columns to long format
platform_df = high_rated_movies.melt(id_vars=['Series_Title', 'IMDB_Rating'], 
                                     value_vars=platform_columns, 
                                     var_name='Platform', 
                                     value_name='Available')

# Step 6: Filter rows where 'Available' is 1 (indicating the movie is available on that platform)
platform_df = platform_df[platform_df['Available'] == 1]

# Step 7: Count the number of high-rated movies available on each platform
platform_counts = platform_df.groupby('Platform').size()

# Step 8: Plot the results using a line plot
plt.figure(figsize=(10, 6))
sns.lineplot(x=platform_counts.index, y=platform_counts.values, marker='o', palette='coolwarm')
plt.title('Distribution of High-Rated Movies Across Streaming Platforms', fontsize=16)
plt.xlabel('Platform', fontsize=12)
plt.ylabel('Number of High-Rated Movies', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# VISULISATION 5
# Treemap for IMDb
# Sort the dataset by IMDb ratings and select the top 10 movies
top_10_imdb = merged_df.sort_values(by='IMDB_Rating', ascending=False).head(10)

# Prepare data for the treemap
labels = top_10_imdb['Series_Title'] + '\n' + top_10_imdb['IMDB_Rating'].astype(str)
sizes = top_10_imdb['IMDB_Rating']

# Create the treemap
plt.figure(figsize=(12, 8))
squarify.plot(
    sizes=sizes,
    label=labels,
    alpha=0.8,
    color=plt.cm.viridis_r(sizes / sizes.max()),
    edgecolor="black"  # Add black borders
)

# Add title and formatting
plt.title('Top 10 Movies by IMDb Ratings', fontsize=16, weight='bold', color='darkblue')
plt.axis('off')  # Turn off axes
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------------
# Adding new column Dominant Genre 

# step 1 make the new column dominant genre 
merged_df['Dominant_Genre'] = merged_df['Genre'].str.split(',').str[0].str.strip()

# step 2 analysing imdb rating based on dominant genre 
genre_ratings = merged_df.groupby('Dominant_Genre')['IMDB_Rating'].mean().sort_values(ascending=False)

# step 3 creating the visulisation 
# Plotting the genres vs IMDb ratings
genre_rating = merged_df.groupby('Dominant_Genre')['IMDB_Rating'].mean().sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.heatmap(genre_rating.to_frame().T, annot=True, cmap='coolwarm', cbar=False, fmt=".2f")
plt.title('Average IMDb Rating by Dominant Genre', fontsize=16, weight='bold', color='purple')
plt.xlabel('Dominant Genre', fontsize=12)
plt.xticks(rotation=45, ha='right') 
plt.yticks([])
plt.tight_layout()
plt.show()
# --------------------------------------------------------------------------------

print(merged_df.head(10))