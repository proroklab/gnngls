import networkx as nx
import random
import numpy as np
import time

# Parameters
POP_SIZE = 100
NUM_GENERATIONS = 200
ALPHA = 1
BETA = 5
PHEROMONE_PERSISTENCE = 0.3
INITIAL_PHEROMONE = 0.0001
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
NUM_CITIES=10

# Define cost function
def cost_function(G, solution):
    cost = 0
    for i in range(len(solution) - 1):
        cost += G[solution[i]][solution[i + 1]]['weight']
    return cost

def cost_function_with_regret(G, solution):
    cost = 0
    for i in range(len(solution) - 1):
        cost += G[solution[i]][solution[i + 1]]['regret_pred']
    return cost

# Initialize pheromone levels
def initialize_pheromone(G):
    pheromone = np.full((NUM_CITIES, NUM_CITIES), INITIAL_PHEROMONE)
    return pheromone

# Update pheromone levels
def update_pheromone(pheromone, population, costs):
    pheromone *= (1 - PHEROMONE_PERSISTENCE)
    for solution, cost in zip(population, costs):
        for i in range(len(solution) - 1):
            pheromone[solution[i], solution[i + 1]] += (1 / cost)
    return pheromone

# Construct new solutions using ACO
def construct_solutions_using_aco(G, pheromone):
    population = []
    for i in range(POP_SIZE):
        current_node = random.randint(0, 9)
        visited = set([current_node])
        solution = [current_node]
        while len(solution) < 10:
            pheromone_values = pheromone[current_node, :] ** ALPHA * (1 / G[current_node, :]['weight']) ** BETA
            pheromone_values[list(visited)] = 0
            if sum(pheromone_values) == 0:
                solution.append(random.choice(list(set(range(10)) - visited)))
                visited.add(solution[-1])
            else:
                pheromone_values /= sum(pheromone_values)
                next_node = np.random.choice(10, 1, p=pheromone_values)[0]
                solution.append(next_node)
                visited.add(next_node)
                current_node = next_node
        population.append(solution)
    return population

# Perform crossover
def perform_crossover(parent1, parent2):
    if random.random() < CROSSOVER_PROBABILITY:
        crossover_point1 = random.randint(0, 8)
        crossover_point2 = random.randint(crossover_point1 + 1, 9)
        child = [-1] * 10
        for i in range(crossover_point1, crossover_point2 + 1):
            child[i] = parent1[i]
        j = 0
        for i in range(10):
            if child[i] == -1:
                while parent2[j] in child:
                    j += 1
                child[i] = parent2[j]
        return child
    else:
        return parent1

# Perform mutation
def perform_mutation(solution):
    if random.random() < MUTATION_PROBABILITY:
        mutation_index1 = random.randint(0, 9)
        mutation_index2 = random.randint(0, 9)
        solution[mutation_index1], solution[mutation_index2] = solution[mutation_index2], solution[mutation_index1]
    return solution

# Update the population using the genetic algorithm
def update_population(G, population):
    costs = [cost_function(G, solution) for solution in population]
    fitness_scores = [1 / cost for cost in costs]
    new_population = [population[np.argmax(fitness_scores)]]
    for i in range(POP_SIZE - 1):
        parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
        child = perform_crossover(parent1, parent2)
        child = perform_mutation(child)
        new_population.append(child)
    return new_population

def update_population_with_regret(G, population):
    costs = [cost_function_with_regret(G, solution) for solution in population]
    fitness_scores = [1 / cost for cost in costs]
    new_population = [population[np.argmax(fitness_scores)]]
    for i in range(POP_SIZE - 1):
        parent1, parent2 = random.choices(population, weights=fitness_scores, k=2)
        child = perform_crossover(parent1, parent2)
        child = perform_mutation(child)
        new_population.append(child)
    return new_population
def aco_search(G):
    pheromone = initialize_pheromone(G)
    for i in range(NUM_GENERATIONS):
        population = construct_solutions_using_aco(G, pheromone)
        population = update_population(population)
        pheromone = update_pheromone(pheromone, population, [cost_function(G, solution) for solution in population])
        costs = [cost_function(G, solution) for solution in population]
        min_cost = min(costs)
        min_cost_index = costs.index(min_cost)
        if min_cost < best_cost:
            best_cost = min_cost
            best_solution = population[min_cost_index]
        print(f'Generation {i}: Best cost = {best_cost}')
    print(f'Best solution: {best_solution}, Cost: {best_cost}')

def aco_search_with_regret(G):
    pheromone = initialize_pheromone(G)
    search_progress = []
    for i in range(NUM_GENERATIONS):
        population = construct_solutions_using_aco(G, pheromone)
        population = update_population_with_regret(population)
        pheromone = update_pheromone(pheromone, population, [cost_function_with_regret(G, solution) for solution in population])
        costs = [cost_function_with_regret(G, solution) for solution in population]
        min_cost = min(costs)
        min_cost_index = costs.index(min_cost)
        if min_cost < best_cost:
            best_cost = min_cost
            best_solution = population[min_cost_index]
        print(f'Generation {i}: Best cost = {best_cost}')
    print(f'Best solution: {best_solution}, Cost: {best_cost}')
    search_progress.append({
        'time': time.time()
    })
    return best_solution, best_cost, search_progress


 # Main function
def main():
    G = load_tsp_instance()
    pheromone = initialize_pheromone(G)
    best_solution = None
    best_cost = float('inf')
    for i in range(NUM_GENERATIONS):
        population = construct_solutions_using_aco(G, pheromone)
        population = update_population_with_regret(population)
        pheromone = update_pheromone(pheromone, population, [cost_function(G, solution) for solution in population])
        costs = [cost_function(G, solution) for solution in population]
        min_cost = min(costs)
        min_cost_index = costs.index(min_cost)
        if min_cost < best_cost:
            best_cost = min_cost
            best_solution = population[min_cost_index]
        print(f'Generation {i}: Best cost = {best_cost}')
    print(f'Best solution: {best_solution}, Cost: {best_cost}')

# Run the main loop
if __name__ == '__main__':
    main()

