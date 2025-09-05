import pygame
import numpy as np

# This fuction only let user to control the first agent
#  All other agents conitnue as they were
def get_interactive_action(action):
    action[0] = np.zeros_like(action[0])

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            return
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action[0,0] = 1.
            elif event.key == pygame.K_LEFT:
                action[0, 1] = -1.0
            elif event.key == pygame.K_RIGHT:
                action[0, 1] = 1.0

    keys = pygame.key.get_pressed()

    if keys[pygame.K_SPACE]:
        action[0, 0] = 1.
    if keys[pygame.K_LEFT]:
        action[0, 1] = -1.0
    if keys[pygame.K_RIGHT]:
        action[0, 1] = 1.0

    return action
