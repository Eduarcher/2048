from archive.mlp import MLPClassifierOverride
import numpy as np
from random import random


class Evolution:
    def __init__(self, population_size, cross_prob=1.0, cross_range=0.01, score_multiplier=1.0, start_gen=1,
                 mutation_prob=0.1, mutation_range=3, max_mutations=5):
        self.population_size = population_size
        self.gen = start_gen
        self.population = self.random_sample()
        self.fitness = []
        self.score_multiplier = score_multiplier
        self.cross_prob = cross_prob
        self.cross_range = cross_range
        self.mutation_prob = mutation_prob
        self.mutation_range = mutation_range
        self.max_mutations = max_mutations
        self.max_fit_count = 0
        self.max_fit_object = None

    def random_sample(self):
        candidates = []
        for i in range(self.population_size):
            tmp = MLPClassifierOverride(expected_inputs=16,
                                        solver='lbfgs',
                                        alpha=1e-5,
                                        hidden_layer_sizes=(3, 2),
                                        random_start_range=5)
            tmp.fit([[10, 5, 0, 10, 5, 0, 10, 5, 0, 10, 5, 0, 10, 5, 0, 1],
                     [11, 5, 0, 10, 5, 0, 10, 5, 0, 10, 5, 0, 10, 5, 0, 1]],
                    [[0, 1], [1, 0]])
            tmp.set_random_coefs()
            candidates.append(tmp)
        return candidates

    def explode_coefs(self, coefs, ints, layers):
        coefs_exploded = []
        for i in range(1, len(layers)):
            for row in coefs[i - 1]:
                [coefs_exploded.append(x) for x in row]
        for layer in ints:
            for w in layer:
                coefs_exploded.append(w)
        return coefs_exploded

    def implode_coefs(self, coefs, layers):
        c = 0
        coefs_imploded = []
        for i in range(1, len(layers)):
            shape = [layers[i - 1], layers[i]]
            layer = []
            for j in range(shape[0]):
                row = []
                for k in range(shape[1]):
                    row.append(coefs[c])
                    c += 1
                layer.append(row)
            coefs_imploded.append(np.array(layer))
        ints_imploded = []
        for i in range(1, len(layers)):
            row = []
            for j in range(layers[i]):
                row.append(coefs[c])
                c += 1
            ints_imploded.append(np.array(row))
        return coefs_imploded, ints_imploded

    def crossover(self):
        parent1 = self.population[self.roulette()]
        parent2 = self.population[self.roulette()]
        layers = [parent1.expected_inputs]
        for layer in parent1.hidden_layer_sizes:
            layers.append(layer)
        layers.append(1)
        if random() < self.cross_prob:
            parent1_coefs = self.explode_coefs(parent1.coefs_, parent1.intercepts_, layers)
            parent2_coefs = self.explode_coefs(parent2.coefs_, parent2.intercepts_, layers)
            n = len(parent1_coefs)
            offsprings = []
            for i in range(n):
                roll = random()
                if roll > 0.75:
                    offsprings.append(parent1_coefs[i])
                elif roll > 0.5:
                    offsprings.append(parent2_coefs[i])
                else:
                    offsprings.append(np.random.uniform(low=min(parent1_coefs[i], parent2_coefs[i]) - self.cross_range,
                                                        high=max(parent1_coefs[i], parent2_coefs[i]) + self.cross_range))
            offsprings = self.mutate(offsprings)
        else:
            offsprings = self.explode_coefs(parent1.coefs_, parent1.intercepts_, layers)
            offsprings = self.mutate(offsprings)
        return self.implode_coefs(offsprings, layers)

    def roulette(self):
        roll = np.random.random()
        total_fit = sum(self.fitness)
        bottom = 0
        for i in range(self.population_size):
            top = bottom + (self.fitness[i] / total_fit)
            if roll <= top:
                match = i
                break
            else:
                bottom = top
        return match

    def mutate(self, coefs):
        n = len(coefs)
        for i in range(self.max_mutations):
            if random() < self.mutation_prob:
                pos = np.random.permutation(range(n))[0]
                coefs[pos] = np.random.normal(coefs[pos], self.mutation_range)
        return coefs

    def get_best_fit_id(self):
        return int(np.argmax(self.fitness))

    def get_best_fit_player(self):
        return self.population[self.get_best_fit_id()]

    def get_mean_fit(self):
        return np.mean(self.fitness)

    def get_max_fit(self):
        return np.max(self.fitness)

    def update_fit(self, game, comA, comB):
        fit_a = (game.bounces_a * 3 + (game.score_a*(self.score_multiplier ** self.gen))) - game.penalty[0]
        fit_b = (game.bounces_b * 3 + (game.score_b*(self.score_multiplier ** self.gen))) - game.penalty[1]
        if fit_a < 1: fit_a = 0.5
        if fit_b < 1: fit_b = 0.5
        if fit_a > self.max_fit_count:
            self.max_fit_object = comA
            self.max_fit_count = fit_a
        if fit_b > self.max_fit_count:
            self.max_fit_object = comB
            self.max_fit_count = fit_b
        self.fitness.append(fit_a)
        self.fitness.append(fit_b)

