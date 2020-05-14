from archive.evolution import Evolution
from game import Game

# Evolution Params
max_iter = 100
M = 20
score_multiplier = 1.0
cross_prob = 0.7
cross_range = 0.1
mutation_prob = 0.10
mutation_range = 3
max_mutations = 3
penalty_clock_paddle = 20

# Initialize Evolution Manager
manager = Evolution(population_size=M, score_multiplier=score_multiplier, cross_range=cross_range, cross_prob=cross_prob,
                    mutation_prob=mutation_prob, mutation_range=mutation_range, max_mutations=max_mutations)

# -------- Main Program Loop -----------
while manager.gen < max_iter:
    manager.fitness = []
    for i in range(M):
        com = manager.population[i]
        game = Game()
        while not game.finish:
            game.move(com.get_update(game.get_state().reshape(1, -1)))
