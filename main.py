import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("dataset.csv")
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

def menu():
    print("\n🎵 Spotify Analysis Tool")
    print("1. Correlation analysis")
    print("2. Danceability vs Popularity plot")
    print("3. List top 10 most popular songs")
    print("4. Add a new song and predict its popularity")
    print("5. Exit")

def correlation_analysis():
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def dance_vs_popularity():
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x="danceability", y="popularity", hue="track_genre", alpha=0.7)
    plt.title("Danceability vs Popularity")
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def top_songs():
    top = df.sort_values(by="popularity", ascending=False).head(10)
    print("\n🎶 Top 10 Most Popular Songs:\n")
    for i, row in top.iterrows():
        print(f"{row['track_name']} - {row['artists']} (Popularity: {row['popularity']})")

def predict_popularity():
    print("\nEnter the new song details:")
    danceability = float(input("Danceability (0-1): "))
    energy = float(input("Energy (0-1): "))
    valence = float(input("Valence (0-1): "))
    tempo = float(input("Tempo (e.g., 120): "))
    loudness = float(input("Loudness (e.g., -5): "))
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

    # --- Görselleştirme ---
    plt.figure(figsize=(10, 6))
    # Var olan şarkıları çiz
    plt.scatter(df["danceability"], df["popularity"], alpha=0.7, label="Dataset Songs")
    # Yeni şarkıyı farklı bir renkle ekle
    plt.scatter([danceability], [prediction], color="red", marker="*", s=200, label="Your Song (Predicted)")
    plt.xlabel("Danceability")
    plt.ylabel("Popularity")
    plt.title("Danceability vs Popularity (Yeni Şarkınız Eklenmiş)")
    plt.legend()
    plt.tight_layout()
    plt.show()


while True:
    menu()
    choice = input("Your choice (1-5): ")

    match choice:
        case "1":
            correlation_analysis()
        case "2":
            dance_vs_popularity()
        case "3":
            top_songs()
        case "4":
            predict_popularity()
        case "5":
            print("Exiting...")
            break
        case _:
            print("Invalid choice.")
