import numpy as np
import pygame
from pygame.locals import *
import sys
import itertools

window_size = [1080, 720]
fps = 60
palette = {'bg': (33, 33, 33), 'boid': (33, 150, 243)}
boid_radius = 7

num_boids = 256
max_speed = 500
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

    for i in range(num_boids):
        neighbors = boids[in_sight[i]]
        if len(neighbors) != 0:
            # Alignment.
            avg_velocity = np.sum(neighbors[:, 2:4], axis=0) / len(neighbors)
            vel_difference = avg_velocity - boids[i, 2:4]
            boids[i, 2:4] += avg_velocity * 0.1

            # Cohesion.
            avg_position = np.sum(neighbors[:, 0:2], axis=0) / len(neighbors)
            pos_difference = avg_position - boids[i, 0:2]
            boids[i, 2:4] += pos_difference * 0.1

        # Separation.
        intruders = boids[in_sepatation_range[i]]
        if len(intruders) != 0:
            separation_vector = boids[i, 0:2] - np.sum(intruders[:, 0:2], axis=0) / len(intruders)
            norm_separation_vec = separation_vector / np.linalg.norm(separation_vector)
            boids[i, 2:4] += 50 * norm_separation_vec

    # Limit the speed.
    speeds = np.linalg.norm(boids[:, 2:4], axis=1)
    mask = speeds > max_speed
    off_limit = boids[:, 2:4][mask]
    normalized_velocities = off_limit / speeds[:, np.newaxis][mask]
    boids[:, 2:4][mask] = normalized_velocities * max_speed

    # Fear mouse clicks.
    mouse_pressed, _, _ = pygame.mouse.get_pressed()
    mouse_range = 200
    if mouse_pressed:
        mouse_pos = np.array(pygame.mouse.get_pos())
        mouse_vectors = boids[:, 0:2] - mouse_pos
        mouse_distances = np.sum(np.square(mouse_vectors), axis=1)
        mouse_in_sight = np.logical_and(0 < mouse_distances,  mouse_distances < mouse_range**2)
        boids_mouse = boids[mouse_in_sight]
        boids_mouse[:, 2:4] += 10000 * mouse_vectors[mouse_in_sight] / mouse_distances[mouse_in_sight][:, np.newaxis]
        boids[mouse_in_sight] = boids_mouse

    # Update position.
    boids[:, 0:2] += boids[:, 2:4] * 1/fps
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
        if event.type == KEYDOWN and event.key == K_ESCAPE:
            pygame.quit()
            sys.exit()
