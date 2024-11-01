import numpy as np
import random


# تابع هدف
def fitness_function(x):
    return x ** 2


# تولید جمعیت اولیه
def create_population(size):
    return [random.randint(0, 31) for _ in range(size)]


# انتخاب والدین
def select_parents(population):
    weights = [fitness_function(x) for x in population]
    return random.choices(population, weights=weights, k=2)


# عملگر تقاطع (Crossover)
def crossover(parent1, parent2):
    crossover_point = random.randint(0, 4)  # نقطه تقاطع
    mask = (1 << crossover_point) - 1
    child1 = (parent1 & mask) | (parent2 & ~mask)
    child2 = (parent2 & mask) | (parent1 & ~mask)
    return child1, child2


# عملگر جهش (Mutation)
def mutate(individual, mutation_rate=0.1):
    if random.random() < mutation_rate:
        mutation_point = random.randint(0, 4)
        individual ^= (1 << mutation_point)  # تغییر یک بیت
    return individual


# الگوریتم ژنتیک
def genetic_algorithm(population_size, generations):
    population = create_population(population_size)

    for generation in range(generations):
        new_population = []

        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population

    # پیدا کردن بهترین فرد
    best_individual = max(population, key=fitness_function)
    return best_individual, fitness_function(best_individual)


# اجرای الگوریتم
best_solution, best_fitness = genetic_algorithm(population_size=10, generations=20)
print(f"Best solution: {best_solution}, Fitness: {best_fitness}")