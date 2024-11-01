import numpy as np
import random
import matplotlib.pyplot as plt


# تابع هدف
def fitness_function(x):
    return x ** 2


# کلاس ذره
class Particle:
    def __init__(self, bounds):
        self.position = np.random.uniform(bounds[0], bounds[1])
        self.velocity = np.random.uniform(-1, 1)
        self.best_position = self.position
        self.best_fitness = fitness_function(self.position)

    def update(self, global_best_position, inertia_weight, cognitive_weight, social_weight):
        # به‌روزرسانی سرعت
        r1, r2 = np.random.rand(2)
        self.velocity = (inertia_weight * self.velocity +
                         cognitive_weight * r1 * (self.best_position - self.position) +
                         social_weight * r2 * (global_best_position - self.position))

        # به‌روزرسانی موقعیت
        self.position += self.velocity

        # اطمینان از اینکه موقعیت در محدوده باشد
        self.position = np.clip(self.position, -10, 10)

        # به‌روزرسانی بهترین موقعیت
        fitness = fitness_function(self.position)
        if fitness < self.best_fitness:
            self.best_fitness = fitness
            self.best_position = self.position


# الگوریتم PSO
def particle_swarm_optimization(num_particles, num_iterations):
    bounds = [-10, 10]
    particles = [Particle(bounds) for _ in range(num_particles)]

    global_best_position = min(particles, key=lambda p: p.best_fitness).best_position

    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5

    for _ in range(num_iterations):
        for particle in particles:
            particle.update(global_best_position, inertia_weight, cognitive_weight, social_weight)

        global_best_position = min(particles, key=lambda p: p.best_fitness).best_position

    return global_best_position, fitness_function(global_best_position)


# اجرای الگوریتم
best_solution, best_fitness = particle_swarm_optimization(num_particles=30, num_iterations=100)
print(f"Best solution: {best_solution}, Fitness: {best_fitness}")

# ترسیم تابع هدف
x = np.linspace(-10, 10, 400)
y = fitness_function(x)

plt.plot(x, y, label='Objective Function: f(x) = x^2')
plt.scatter(best_solution, best_fitness, color='red', label='Best Solution')
plt.title('Particle Swarm Optimization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.grid()
plt.show()