import numpy as np

block = np.array([
    [0, 0, 0, 0],
    [0, 255, 255, 0],
    [0, 255, 255, 0],
    [0, 0, 0, 0],
])

behive = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 0, 0],
    [0, 255, 0, 0, 255, 0],
    [0, 0, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

loaf = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 0, 0],
    [0, 255, 0, 0, 255, 0],
    [0, 0, 255, 0, 255, 0],
    [0, 0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

boat = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 255, 0, 0],
    [0, 255, 0, 255, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0]
])

tub = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 255, 0, 0],
    [0, 255, 0, 255, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0]
])

blinker = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0]
])

toad_1 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0, 0],
    [0, 255, 0, 0, 255, 0],
    [0, 255, 0, 0, 255, 0],
    [0, 0, 255, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

toad_2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 255, 255, 255, 0],
    [0, 255, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

beacon_1 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 255, 255, 0, 0, 0],
    [0, 255, 255, 0, 0, 0],
    [0, 0, 0, 255, 255, 0],
    [0, 0, 0, 255, 255, 0],
    [0, 0, 0, 0, 0, 0]
])

beacon_2 = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 255, 255, 0, 0, 0],
    [0, 255, 0, 0, 0, 0],
    [0, 0, 0, 0, 255, 0],
    [0, 0, 0, 255, 255, 0],
    [0, 0, 0, 0, 0, 0]
])

glider_1 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 255, 0],
    [0, 255, 255, 255, 0],
    [0, 0, 0, 0, 0]
])
glider_2 = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 0, 255, 0],
    [0, 0, 255, 255, 0],
    [0, 0, 255, 0, 0],
    [0, 0, 0, 0, 0]
])
glider_3 = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 255, 0],
    [0, 255, 0, 255, 0],
    [0, 0, 255, 255, 0],
    [0, 0, 0, 0, 0]
])
glider_4 = np.array([
    [0, 0, 0, 0, 0],
    [0, 255, 0, 0, 0],
    [0, 0, 255, 255, 0],
    [0, 255, 255, 0, 0],
    [0, 0, 0, 0, 0]
])

spaceship_1 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 0, 0, 0, 0, 255, 0],
        [0, 255, 0, 0, 0, 255, 0],
        [0, 0, 255, 255, 255, 255, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
)
spaceship_2 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 255, 255, 0, 0],
        [0, 255, 255, 0, 255, 255, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 0, 255, 255, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
)
spaceship_3 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 255, 255, 255, 0],
        [0, 255, 0, 0, 0, 255, 0],
        [0, 0, 0, 0, 0, 255, 0],
        [0, 255, 0, 0, 255, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
)
spaceship_4 = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 255, 255, 0, 0, 0],
        [0, 255, 255, 255, 255, 0, 0],
        [0, 255, 255, 0, 255, 255, 0],
        [0, 0, 0, 255, 255, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]
)

