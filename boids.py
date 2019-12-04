import numpy as np
import pygame
from pygame.locals import *
import sys
import itertools

window_size = [1080, 720]
fps = 60
palette = {'bg': (33, 33, 33), 'boid': (76, 175, 80), \
           'separating_vectors': (33, 150, 243), 'highlight': (244, 67, 54)}
boid_radius = 4

num_boids = 128
max_speed = 10
sight_range, separation_range = 100, 25
boids = np.zeros((num_boids, 4))

# Generate positions and velocities.
boids[:, 0:2] = np.random.uniform(0, 1, (num_boids, 2)) * np.array(window_size)
boids[:, 2:4] = np.random.uniform(-1, 1, (num_boids, 2)) * max_speed

def update(): # Semi-implicit Euler.
    separating_vectors = boids[:, 0:2] - boids[:, 0:2][:, np.newaxis]
    distances = np.sum(np.square(separating_vectors), axis=2)
    in_sight = np.logical_and(0 < distances,  distances < sight_range**2)
    in_sepatation_range = np.logical_and(0 < distances,  distances < separation_range**2)

    # TODO: Vectorize this for loop.
    for i in range(num_boids):
        neighbors = boids[in_sight[i]]
        if len(neighbors) != 0:
            # Alignment.
            avg_velocity = np.sum(neighbors[:, 2:4], axis=0) / len(neighbors)
            vel_difference = avg_velocity - boids[i, 2:4]
            boids[i, 2:4] += vel_difference * 0.01

            # Cohesion.
            avg_position = np.sum(neighbors[:, 0:2], axis=0) / len(neighbors)
            pos_difference = avg_position - boids[i, 0:2]
            boids[i, 2:4] += pos_difference * 0.005

        # Separation.
        intruders = boids[in_sepatation_range[i]]
        if len(intruders) != 0:
            separation_vector = boids[i, 0:2] - np.sum(intruders[:, 0:2], axis=0) / len(intruders)
            boids[i, 2:4] += separation_vector * 0.05

    # Limit the speed.
    speeds = np.linalg.norm(boids[:, 2:4], axis=1)
    mask = speeds > max_speed
    off_limit = boids[:, 2:4][mask]
    normalized_velocities = off_limit / speeds[:, np.newaxis][mask]
    boids[:, 2:4][mask] = normalized_velocities * max_speed

    boids[:, 0:2] += boids[:, 2:4]
    boids[:, 0:2] %= window_size


pygame.init()
pygame.display.set_caption('Boids')
screen = pygame.display.set_mode(window_size)
clock = pygame.time.Clock()

def render():
    screen.fill(palette['bg'])
    for i in range(num_boids):
        pygame.draw.circle(screen, palette['boid'], boids[i, 0:2].astype(int), boid_radius)
    pygame.display.flip()

while True:
    clock.tick(fps)
    update()
    render()

    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
