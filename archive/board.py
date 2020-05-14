import pandas as pd
import numpy as np
from random import randrange
import random


class Board:
    def __init__(self, random_four_start=6):
        self.lx = 4
        self.ly = 4
        self.random_four_start = random_four_start
        self.tiles = pd.DataFrame(np.zeros((4, 4), dtype=int))
        self.random_start()

    def print_board(self):
        for i in range(self.lx):
            for j in range(self.ly):
                print(self.tiles[i][j], end=' ')
            print("")
        print("------")


    def random_add(self):
        for i in random.sample(range(self.lx), self.lx):
            for j in random.sample(range(self.ly), self.ly):
                if self.tiles[i][j] == 0:
                    self.tiles[i][j] = 4 if randrange(10) >= self.random_four_start else 2
                    return True
        return False


    def random_start(self):
        self.random_add()
        self.random_add()


    def b_up(self):
        i = 0
        merge = 0
        points = 0
        updated = False
        while i < self.lx:
            for j in range(self.ly):
                if i is not 0 and self.tiles[i][j] != 0:
                    if self.tiles[i-1][j] == 0:
                        self.tiles[i-1][j] = self.tiles[i][j]
                        self.tiles[i][j] = 0
                        i = 0
                        updated = True
                    elif self.tiles[i-1][j] == self.tiles[i][j] and merge <= 0:
                        self.tiles[i-1][j] *= 2
                        self.tiles[i][j] = 0
                        points += self.tiles[i-1][j]
                        merge = 2
                        updated = True
            merge -= 1
            i += 1
        return points, updated

    def b_down(self):
        i = self.lx - 1
        merge = 0
        points = 0
        updated = False
        while i >= 0:
            for j in range(self.ly):
                if i is not self.lx - 1 and self.tiles[i][j] != 0:
                    if self.tiles[i+1][j] == 0:
                        self.tiles[i+1][j] = self.tiles[i][j]
                        self.tiles[i][j] = 0
                        i = self.lx-1
                        updated = True
                    elif self.tiles[i+1][j] == self.tiles[i][j] and merge <= 0:
                        self.tiles[i+1][j] *= 2
                        self.tiles[i][j] = 0
                        points += self.tiles[i+1][j]
                        merge = 2
                        updated = True
            merge -= 1
            i -= 1
        return points, updated

    def b_left(self):
        j = 0
        merge = 0
        points = 0
        updated = False
        while j < self.ly:
            for i in range(self.lx):
                if j is not 0 and self.tiles[i][j] != 0:
                    if self.tiles[i][j-1] == 0:
                        self.tiles[i][j-1] = self.tiles[i][j]
                        self.tiles[i][j] = 0
                        j = 0
                        updated = True
                    elif self.tiles[i][j-1] == self.tiles[i][j] and merge <= 0:
                        self.tiles[i][j-1] *= 2
                        self.tiles[i][j] = 0
                        points += self.tiles[i][j-1]
                        merge = 2
                        updated = True
            merge -= 1
            j += 1
        return points, updated

    def b_right(self):
        j = self.ly - 1
        merge = 0
        points = 0
        updated = False
        while j >= 0:
            for i in range(self.lx):
                if j is not self.ly - 1 and self.tiles[i][j] != 0:
                    if self.tiles[i][j+1] == 0:
                        self.tiles[i][j+1] = self.tiles[i][j]
                        self.tiles[i][j] = 0
                        j = self.ly-1
                        updated = True
                    elif self.tiles[i][j+1] == self.tiles[i][j] and merge <= 0:
                        self.tiles[i][j+1] *= 2
                        self.tiles[i][j] = 0
                        points += self.tiles[i][j+1]
                        merge = 2
                        updated = True
            merge -= 1
            j -= 1
        return points, updated
