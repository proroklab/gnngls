import random

class GeneticAlgorithm:
    def __init__(self, G, population_size, crossover_rate, mutation_rate, generations):
        self.G = G
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.generations = generations
        
        self.population = [self.random_tour() for i in range(self.population_size)]
        
    def random_tour(self):
        tour = list(self.G.nodes)
        random.shuffle(tour)
        return tour
        
    def fitness(self, tour):
        return -gnngls.tour_cost(self.G, tour)
        
    def select_parents(self):
        parents = random.choices(self.population, weights=[self.fitness(t) for t in self.population], k=2)
        return parents
        
    def crossover(self, parent1, parent2):
        if random.random() > self.crossover_rate:
            return parent1
        
        child = [None] * len(parent1)
        start = random.randint(0, len(parent1) - 1)
        end = random.randint(start, len(parent1) - 1)
        
        for i in range(start, end + 1):
            child[i] = parent1[i]
            
        j = 0
        for i in range(len(child)):
            if child[i] is None:
                while parent2[j] in child:
                    j += 1
                child[i] = parent2[j]
                
        return child
        
    def mutate(self, tour):
        if random.random() > self.mutation_rate:
            return tour
        
        i = random.randint(0, len(tour) - 1)
        j = random.randint(0, len(tour) - 1)
        
        tour[i], tour[j] = tour[j], tour[i]
        return tour
        
    def evolve(self):
        for generation in range(self.generations):
            next_population = []
            
            for i in range(self.population_size):
                parent1, parent2 = self.select_parents()
                child = self.crossover(parent1, parent2)
                child = self.mutate(child)
                next_population.append(child)
                
            self.population = next_population
        
        return max(self.population, key=self.fitness)
