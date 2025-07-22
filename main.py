import pandas as pd #data load and manipulation
import seaborn as sns #plotting
import matplotlib.pyplot as plt #plotting
from sklearn.linear_model import LinearRegression #predicting song popularity
import numpy as np #data selection 

df = pd.read_csv("dataset.csv") #load dataset to dataframe(pandas)
if "Unnamed: 0" in df.columns: #remove if existing unnamed 0 columns
    df = df.drop(columns=["Unnamed: 0"])

def menu(): #main menu display
    print("\n🎵 Spotify Analysis Tool")
    print("1. Correlation matrix")
    print("2. Custom correlation plot (by genre)")
    print("3. Danceability vs Popularity plot (by genre)")
    print("4. List top 10 most popular songs")
    print("5. Add a new song and predict its popularity")
    print("6. Exit")

def choose_genre():#function that helps genre selection numerically
    all_genres = df['track_genre'].dropna().unique().tolist()
    print("\nAvailable genres:")
    for idx, genre in enumerate(all_genres, 1):
        print(f"{idx}. {genre}")
    while True:
        try:
            genre_index = int(input("Enter genre number: "))
            if 1 <= genre_index <= len(all_genres):
                return all_genres[genre_index - 1]
            else:
                print("Please enter a valid genre number.")
        except ValueError:
            print("Please enter a valid integer.")

def correlation_analysis(): #function that shows full correlation matrix as heatmap
    print("\nCorrelation Matrix:")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def custom_correlation_plot(): #lets user choose a genre and plot custom correlation plot
    print("\nCustom Correlation Plot (by genre)")
    genre_input = choose_genre() #by number ofc
    genre_df = df[df['track_genre'] == genre_input] #so it filters only selected genre
    if genre_df.empty:
        print("No data found for the selected genre.")
        return

    numeric_cols = genre_df.select_dtypes(include=np.number).columns.tolist() #list numeric columns for spesific genre
    print("\nAvailable numeric columns in this genre:")
    for idx, col in enumerate(numeric_cols, 1):
        print(f"{idx}. {col}")

    while True:
        try:
            x_idx = int(input("Choose the first variable (number): "))
            if 1 <= x_idx <= len(numeric_cols):
                x_var = numeric_cols[x_idx - 1]
                break
            else:
                print("Please enter a valid column number.")
        except ValueError:
            print("Please enter a valid integer.")

    while True:
        try:
            y_idx = int(input("Choose the second variable (number): "))
            if 1 <= y_idx <= len(numeric_cols) and y_idx != x_idx:
                y_var = numeric_cols[y_idx - 1]
                break
            else:
                print("Please enter a valid column number, different from the first.")
        except ValueError:
            print("Please enter a valid integer.")

    corr = genre_df[[x_var, y_var]].corr().iloc[0,1] ##correlation calculation
    print(f"\nCorrelation coefficient between {x_var} and {y_var} (in genre '{genre_input}'): {corr:.2f}")

    max_point = genre_df.loc[(genre_df[[x_var, y_var]].apply(tuple, axis=1)).idxmax()]
    min_point = genre_df.loc[(genre_df[[x_var, y_var]].apply(tuple, axis=1)).idxmin()]


#here i actually wanted to highlight the top right and bottom left points(most danceable and popular for example), but i couldn't figure out how
    #so i just used the scatter plot to highlight the top right and bottom left points
    plt.figure(figsize=(8,5)) 
    plt.scatter(genre_df[x_var], genre_df[y_var], alpha=0.7)
    plt.scatter(max_point[x_var], max_point[y_var], color='green', marker='*', s=300) #highlight top right (max) point
    
    plt.text(max_point[x_var], max_point[y_var]+1,
             f"{max_point['track_name']} - {max_point['artists']}", color='green', fontsize=9, ha='center')
    plt.scatter(min_point[x_var], min_point[y_var], color='red', marker='*', s=300) #highlight bottom left (min) point
    plt.text(min_point[x_var], min_point[y_var]-4,
             f"{min_point['track_name']} - {min_point['artists']}", color='red', fontsize=9, ha='center')
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.title(f"{x_var} vs {y_var} for {genre_input} (Correlation: {corr:.2f})")
    plt.tight_layout()
    plt.show()

def dance_vs_popularity(): #example of two variables plot for a genre
    genre_input = choose_genre()
    genre_df = df[df['track_genre'] == genre_input]
    if genre_df.empty:
        print("No data found for the selected genre.")
        return

    max_point = genre_df.loc[(genre_df[['danceability', 'popularity']].apply(tuple, axis=1)).idxmax()]
    min_point = genre_df.loc[(genre_df[['danceability', 'popularity']].apply(tuple, axis=1)).idxmin()]

    plt.figure(figsize=(8, 5))
    plt.scatter(genre_df['danceability'], genre_df['popularity'], alpha=0.7, label='All Songs')
    plt.scatter(max_point['danceability'], max_point['popularity'], color='green', marker='*', s=300)
    plt.text(max_point['danceability'], max_point['popularity']+1,
             f"{max_point['track_name']} - {max_point['artists']}", color='green', fontsize=9, ha='center')
    plt.scatter(min_point['danceability'], min_point['popularity'], color='red', marker='*', s=300)
    plt.text(min_point['danceability'], min_point['popularity']-4,
             f"{min_point['track_name']} - {min_point['artists']}", color='red', fontsize=9, ha='center')
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.title(f"Danceability vs Popularity for {genre_input}")
    plt.tight_layout()
    plt.show()

def top_songs():#lists top 10 popular songs (by popularity in the dataset)
    print("\n🎶 Top 10 Most Popular Songs:")
    unique = df.drop_duplicates(subset=["track_name", "artists"])
    top = unique.sort_values(by="popularity", ascending=False).head(10)
    print("\n🎶 Top 10 Most Popular Songs:\n")
    for i, row in top.iterrows():
        print(f"{row['track_name']} - {row['artists']} (Popularity: {row['popularity']})")

def predict_popularity(): #here i couldnt figure out how to make the model predict the popularity of a new song, so i just used the model to predict the popularity of the most popular song in the dataset
    print("\n🎵 Predicting the Popularity of a New Song")
    print("\nEnter the new song details:")
    track_name = input("Track name: ")
    artists = input("Artist(s): ")
    danceability = float(input("Danceability (0-1): "))
    energy = float(input("Energy (0-1): "))
    valence = float(input("Valence (0-1): "))
    tempo = float(input("Tempo (e.g., 120): "))
    while True:
        loudness = float(input("Loudness (range: -60 to 0 dB): "))
        if -60 <= loudness <= 0:
            break
        print("Please enter loudness between -60 and 0.")
    speechiness = float(input("Speechiness (0-1): "))
    acousticness = float(input("Acousticness (0-1): "))

    features = ["danceability", "energy", "valence", "tempo", "loudness", "speechiness", "acousticness"]
    X = df[features]
    y = df["popularity"]

    model = LinearRegression()
    model.fit(X, y)

    new_data = pd.DataFrame([[danceability, energy, valence, tempo, loudness, speechiness, acousticness]], columns=features)
    prediction = model.predict(new_data)[0]
    print(f"\n🔮 Predicted popularity: {round(prediction, 2)} / 100")

    top5_genres = df['track_genre'].value_counts().nlargest(5).index.tolist()
    df_top_songs = pd.DataFrame()
    for genre in top5_genres:
        genre_songs = df[df['track_genre'] == genre].drop_duplicates(subset=["track_name", "artists"])
        top3 = genre_songs.sort_values(by="popularity", ascending=False).head(3)
        df_top_songs = pd.concat([df_top_songs, top3])

    plt.figure(figsize=(11, 6))
    for genre in top5_genres:
        genre_songs = df_top_songs[df_top_songs['track_genre'] == genre]
        plt.scatter(
            genre_songs['danceability'], genre_songs['popularity'],
            alpha=0.7, label=genre
        )
        for _, row in genre_songs.iterrows():
            plt.text(row['danceability'], row['popularity']+1,
                     f"{row['track_name']} - {row['artists']}",
                     fontsize=7, ha='center')
    plt.scatter([danceability], [prediction], color="black", marker="*", s=350, label=f"{track_name} by {artists}")
    plt.text(danceability, prediction+2, f"{track_name}\n{artists}", fontsize=10, color="black", ha='center', fontweight='bold')
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.title("Top 5 Genres: Top 3 Songs and Your Song")
    plt.legend()
    plt.tight_layout()
    plt.show()

while True: #simple switch case loop
    menu()
    choice = input("Your choice (1-6): ")
    match choice:
        case "1":
            correlation_analysis()
        case "2":
            custom_correlation_plot()
        case "3":
            dance_vs_popularity()
        case "4":
            top_songs()
        case "5":
            predict_popularity()
        case "6":
            print("Exiting...")
            break
        case _:
            print("Invalid choice.")
