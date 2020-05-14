from math import copysign


def sign(x): return copysign(1, x)


def norm(x, mean): return (x - mean)/mean


def quit_check(events):
    for event in events:
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                return False
    return True


def reset_shape(shape, x, y):
    shape.rect.x = x
    shape.rect.y = y
    return shape
