from board import Board
import numpy as np


class Game:
    def __init__(self):
        self.points = 0
        self.board = Board(random_four_start=9, spare_play=True)
        self.finish = False


    def get_state(self):
        state = []
        for i in range(self.board.lx):
            for j in range(self.board.ly):
                state.append(self.board.tiles[i][j])
        return np.array(state)

    def print_points(self):
        print(f'Points: {self.points}')

    def print_board(self):
        self.board.print_board()

    def move(self, direction):
        points, updated = self.move_direction(direction)
        self.points += points
        if not self.board.random_add():
            if not self.board.check_board():
                self.finish = True
                return np.matrix(self.board.tiles), False, 0, self.finish
        return np.matrix(self.board.tiles), updated, points, self.finish

    def move_direction(self, direction):
        if direction == 0:
            return self.board.b_up()
        elif direction == 1:
            return self.board.b_right()
        elif direction == 2:
            return self.board.b_down()
        elif direction == 3:
            return self.board.b_left()
