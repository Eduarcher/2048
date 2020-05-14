from board import Board
import numpy as np


class Game:
    def __init__(self):
        self.points = 0
        self.board = Board(random_four_start=9)
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
        movement, updated = self.move_direction(direction)
        self.points += movement
        if not self.board.random_add():
            print("Game Over")
            self.finish = True
        self.print_board()

    def move_direction(self, direction):
        if direction == 1:
            return self.board.b_up()
        elif direction == 2:
            return self.board.b_right()
        elif direction == 3:
            return self.board.b_down()
        elif direction == 4:
            return self.board.b_left()
