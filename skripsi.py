import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# Fungsi untuk inisialisasi populasi
def initialize_population(pop_size, k_max, n_features):
    population = []
    for _ in range(pop_size):
        k = random.randint(1, k_max)  # Nilai K antara 1 hingga k_max
        k_bin = format(k, '05b')  # Konversi ke 5 bit biner
        feature_selection = [random.randint(0, 1) for _ in range(n_features)]  # Seleksi fitur (8 bit)
        chromosome = list(k_bin) + feature_selection
        population.append(chromosome)
    return population

# Fungsi decode kromosom menjadi K dan fitur yang dipilih
def decode_chromosome(chromosome):
    k_bin = ''.join(chromosome[:5])  # Ambil 5 bit pertama untuk k
    k = int(k_bin, 2)  # Konversi biner ke integer
    k = max(1, k)  # Pastikan k minimal 1
    feature_selection = [int(bit) for bit in chromosome[5:]]  # Seleksi fitur
    return k, feature_selection

# Fungsi evaluasi fitness
def evaluate_fitness(chromosome, X_train, X_test, y_train, y_test):
    k, feature_selection = decode_chromosome(chromosome)
    selected_features = [i for i, bit in enumerate(feature_selection) if bit == 1]
    if len(selected_features) == 0:  # Jika tidak ada fitur yang dipilih, fitness = 0
        return 0
    # Seleksi fitur pada dataset
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]
    # Training KNN
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)
    # Hitung akurasi
    return accuracy_score(y_test, y_pred)

# Fungsi crossover
def crossover(parent1, parent2):
    point = random.randint(1, len(parent1) - 1)
    offspring1 = parent1[:point] + parent2[point:]
    offspring2 = parent2[:point] + parent1[point:]
    return offspring1, offspring2

# Fungsi mutasi
def mutation(chromosome, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = '1' if chromosome[i] == '0' else '0'
    return chromosome

# Fungsi seleksi turnamen
def tournament_selection(population, fitness_scores, tournament_size=3):
    selected = random.sample(list(zip(population, fitness_scores)), tournament_size)
    selected = sorted(selected, key=lambda x: x[1], reverse=True)
    return selected[0][0]  # Kromosom dengan fitness terbaik

# Fungsi Algoritma Genetika
def genetic_algorithm(X_train, X_test, y_train, y_test, pop_size=10, generations=20, k_max=20, mutation_rate=0.1):
    n_features = X_train.shape[1]
    population = initialize_population(pop_size, k_max, n_features)
    best_chromosome = None
    best_fitness = 0

    for generation in range(generations):
        fitness_scores = [evaluate_fitness(chrom, X_train, X_test, y_train, y_test) for chrom in population]
        print(f"Generation {generation+1} - Best Fitness: {max(fitness_scores):.4f}")

        # Simpan kromosom terbaik
        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_chromosome = population[np.argmax(fitness_scores)]

        # Seleksi, crossover, dan mutasi
        new_population = []
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitness_scores)
            parent2 = tournament_selection(population, fitness_scores)
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.append(mutation(offspring1, mutation_rate))
            new_population.append(mutation(offspring2, mutation_rate))
        population = new_population

    print("\n=== Best Solution ===")
    k, feature_selection = decode_chromosome(best_chromosome)
    selected_features = [i for i, bit in enumerate(feature_selection) if bit == 1]
    print(f"Best K: {k}")
    print(f"Selected Features: {selected_features}")
    print(f"Best Fitness (Accuracy): {best_fitness:.4f}")
    return k,selected_features



# Main Program
def main():
    
    url = "dataset/dataset_diabetes_1_test.csv"
    df = pd.read_csv(url)
    X = df.iloc[:, :-1].values  # 8 fitur
    y = df.iloc[:, -1].values   # Label

    scaler=MinMaxScaler()
    
    # Split dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler.fit(X_train)
    X_train=scaler.transform(X_train)
    X_test=scaler.transform(X_test)
    X_test = np.clip(X_test, 0, 1)

    print("Distribusi kelas pada dataset asli:")
    print(pd.Series(y).value_counts())

    print("\nDistribusi kelas pada dataset pelatihan:")
    print(pd.Series(y_train).value_counts())

    print("\nDistribusi kelas pada dataset pengujian:")
    print(pd.Series(y_test).value_counts())
    # Jalankan Algoritma Genetika untuk optimasi KNN
    k,selected_features=genetic_algorithm(X_train, X_test, y_train, y_test, pop_size=90, generations=100, k_max=31, mutation_rate=0.4)

    joblib.dump(k,"k.pkl")
    joblib.dump(selected_features,"selected_features.pkl")

    return k,selected_features

if __name__ == "__main__":
    main()
