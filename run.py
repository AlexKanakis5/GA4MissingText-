import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os 


def plot_creator(x_axis, y_axis, filename, x_axis_name, y_axis_name):

  # Create the "plots" folder if it doesn't exist
  os.makedirs("plots", exist_ok=True)  

  plt.figure(figsize=(10, 6))
  plt.plot(x_axis, y_axis, marker='o', linestyle='-')
  plt.xlabel(x_axis_name)
  plt.ylabel(y_axis_name)
  plt.title(filename)
  plt.grid(True)
  
  plot_path = os.path.join("plots", filename)
  plt.savefig(plot_path)
  plt.close()
  print(f"plot saved to: {plot_path}")


def load_data(file_path, region_id):
    data = pd.read_csv(file_path, sep='\t')
    filtered_data = data[data['region_main_id'] == region_id]
    return filtered_data

def preprocess_texts(texts):
    texts = texts.str.replace(r'\[|\]', '', regex=True)
    texts = texts.str.replace('-{2,}', '', regex=True)

    return texts



def vectorize_texts(texts):
    global vectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer



def compute_cosine_similarity(vectors, target_vector):
    similarities = cosine_similarity(target_vector, vectors)
    return similarities.flatten()



def fitness(vectors, target_vector):

    # debug
    target_text = vectorizer.inverse_transform(target_vector)
    target_text_str = ' '.join(target_text[0])
    #print(f'Target text: {target_text_str}')


    similarities = compute_cosine_similarity(vectors, target_vector)
    top_n_indices = find_top_n(similarities, 5)
    top_n_similarities = similarities[top_n_indices]
    #print(f'mean{np.mean(top_n_similarities)}')
    return np.mean(top_n_similarities)

def find_top_n(similarities, n):
    sorted_indices = np.argsort(similarities)[::-1]  # Descending order
    top_n_indices = sorted_indices[1:n+1]  # Exclude the first one (itself)

    return top_n_indices



def initilize_population(population_size, number_of_tokens):
    population = []
    for i in range(population_size):
        individual = [random.randint(0, number_of_tokens - 1), random.randint(0, number_of_tokens - 1)]
        population.append(individual)
    return population


def selection(population, fitnesses, num_survivors):

  selected_population = []
  tournament_size = 2  # Adjust this value for larger or smaller tournaments

  while len(selected_population) < num_survivors:
    tournament_participants = random.sample(population, tournament_size)
    tournament_fitness = [fitnesses[population.index(p)] for p in tournament_participants]
    
    # Select the individual with the highest fitness score in the tournament
    winner_index = tournament_fitness.index(max(tournament_fitness))
    selected_population.append(tournament_participants[winner_index])

  return selected_population

def crossover(parent1, parent2):
    crossover_point = random.randint(0, 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2



def mutate(individual, number_of_tokens, mutation_rate, best_individual):

    if individual == best_individual:
        # Don't mutate the elite individual
        pass
        #return individual

    if random.random() < mutation_rate:
        mutate_point = random.randint(0, 1)
        individual[mutate_point] = random.randint(0, number_of_tokens - 1)
    return individual


def genetic_algorithm(population_size, number_of_generations, mutation_rate, vectorizer, target_vector, number_of_tokens, vectors, crossover_rate):
    population = initilize_population(population_size, number_of_tokens)
    best_individual = None
    best_fitness = -999
    num_survivors = population_size // 2  # Keep only the top half

    not_improved_counter = 0 
    best_fitness_progression = []
    for generation in range(number_of_generations):
        fitnesses = [fitness(vectors, vectorizer.transform([f"{index_to_token[ind[0]]} αλεξανδρε ουδις {index_to_token[ind[1]]}"])) for ind in population]

        
        population = selection(population, fitnesses, num_survivors)

        new_population = []
        
        while len(new_population) < population_size:
            if random.random() < crossover_rate:  # Use crossover_rate for probability
                parent1, parent2 = random.sample(population, 2)
                child1, child2 = crossover(parent1, parent2)
                child1 = mutate(child1, number_of_tokens, mutation_rate, best_individual)
                child2 = mutate(child2, number_of_tokens, mutation_rate, best_individual)
                new_population.extend([child1, child2])
            else:
                # If crossover doesn't happen, copy parents 
                new_population.extend(random.sample(population, 2))

        population = new_population[:population_size]
        


        old_best_fitness = best_fitness
        for ind, fit in zip(population, fitnesses):
            if fit >= best_fitness:
                best_fitness = fit
                best_individual = ind

        best_fitness_progression.append(best_fitness)

        if old_best_fitness >= best_fitness:
            not_improved_counter += 1   
        else:
            not_improved_counter = 0

        if not_improved_counter >= 35:
            print(f"stopped improving at {generation} generation")
            break

        

    return best_individual, best_fitness, generation, best_fitness_progression



def main():
    file_path = 'iphi2802.csv'
    region_id = 1683
    target_text = 'αλεξανδρε ουδις'
    data = load_data(file_path, region_id)
    texts = preprocess_texts(data['text'])
    target_text_preprocessed = preprocess_texts(pd.Series([target_text]))
    texts = texts._append(target_text_preprocessed, ignore_index=True)
    vectors, vectorizer = vectorize_texts(texts)
    target_vector = vectors[-1]
    global index_to_token
    index_to_token = {idx: word for idx, word in enumerate(vectorizer.get_feature_names_out())}
    
    
    # init some values
    total_final_generation_numbers = []
    total_best_fitnesses = []
    total_best_fitness_progression = []

    number_of_generations = 200

    population_size = 200
    crossover_rate = 0.1
    mutation_rate = 0.01
    
    
    number_of_tokens = len(index_to_token)
    
    
    for i in range(10):
        best_individual, best_fitness, final_generation_number, best_fitness_progression = genetic_algorithm(population_size, number_of_generations, mutation_rate, vectorizer, target_vector, number_of_tokens, vectors, crossover_rate)
    
        total_final_generation_numbers.append(final_generation_number)
        total_best_fitnesses.append(best_fitness)
        total_best_fitness_progression.append(best_fitness_progression)


    if best_individual is not None:
        best_completion = f"{index_to_token[best_individual[0]]} αλεξανδρε ουδις {index_to_token[best_individual[1]]}"
        print(f"Best completion: {best_completion}")
        print(f"Best fitness: {best_fitness}")

        max_fitness_index = total_best_fitnesses.index(max(total_best_fitnesses))
        max_fitness_run_progression = total_best_fitness_progression[max_fitness_index]  

        plot_creator(range(len(total_best_fitnesses)), total_best_fitnesses, filename=(f'{population_size}_population_size_{crossover_rate}_crossover_rate_{mutation_rate}_mutation_rate_fitness_per_run.png'), x_axis_name="runs", y_axis_name="best fitness")
        plot_creator(range(len(max_fitness_run_progression)), total_best_fitness_progression[max_fitness_index], filename=(f'{population_size}_population_size_{crossover_rate}_crossover_rate_{mutation_rate}_mutation_rate_fitness_per_generation.png'), x_axis_name="generation", y_axis_name="best fitness")

        print(f'mean best fitness:{np.mean(total_best_fitnesses)}')
        print(f'mean_generations:{np.mean(total_final_generation_numbers)}')
    else:
        print("Genetic algorithm could not find a suitable completion.")

    


        

if __name__ == "__main__":
    main()