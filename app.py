import pygame
import neat
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import datetime

# Constants
WIDTH = 800
HEIGHT = 600
FOOD_RADIUS = 5
AGENT_RADIUS = 10
OBSTACLE_RADIUS = 15
NUM_FOOD = 75
NUM_OBSTACLES = 50
SAMPLE_DISTANCE = 15  # Distance to sample color values from
CENTER_MARGIN = OBSTACLE_RADIUS * 3

# Agent class
class Agent:
    def __init__(self, net, genome):
        self.net = net
        self.genome = genome
        self.position = 0.5 * np.array([WIDTH, HEIGHT])
        self.color = (0, 0, 255)
        self.energy = 100

    def sample_surrounding_colors(self, window):
        surrounding_colors = []
        for angle in range(0, 360, 45):  # Sample colors in 8 directions
            x_offset = int(np.cos(np.radians(angle)) * SAMPLE_DISTANCE)
            y_offset = int(np.sin(np.radians(angle)) * SAMPLE_DISTANCE)
            sample_x = int(np.clip(self.position[0] + x_offset, 0, WIDTH - 1))
            sample_y = int(np.clip(self.position[1] + y_offset, 0, HEIGHT - 1))
            color = window.get_at((sample_x, sample_y))[:3]
            normalized_color = [value / 255.0 for value in color]
            surrounding_colors.extend(normalized_color)
        return np.array(surrounding_colors)

    def update(self, screen):
        input_data = self.sample_surrounding_colors(screen)
        agent_position = self.position / np.array([WIDTH, HEIGHT])
        time_input = [(datetime.datetime.now().second + 1) / 60]
        input_data = np.concatenate((input_data, agent_position, time_input))
        input_data = torch.FloatTensor(input_data)

        output_data = self.net.activate(input_data)

        move_x = (output_data[0] - output_data[1]) * 5
        move_y = (output_data[2] - output_data[3]) * 5
        move_direction = np.array([move_x, move_y])
        
        self.color = (output_data[4] * 255, output_data[5] * 255, output_data[6] * 255)

        self.position += move_direction.astype(float)
        self.position = np.clip(self.position, 0, [WIDTH, HEIGHT])

        self.energy -= 1
        self.genome.fitness += 1

    def draw(self, screen):
        color = tuple(map(int, self.color))
        pygame.draw.circle(screen, color, self.position.astype(int), AGENT_RADIUS)

# Evaluation function
def eval_genomes(genomes, config):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    agents = []
    for genome_id, genome in genomes:
        genome.fitness = 0
        substrate = neat.nn.RecurrentNetwork.create(genome, config)
        agent = Agent(substrate, genome)
        agents.append(agent)

    food_positions = np.random.rand(NUM_FOOD, 2) * np.array([WIDTH, HEIGHT])
    obstacle_positions = np.random.rand(NUM_OBSTACLES, 2) * np.array([WIDTH, HEIGHT])
    
    running = True
    population_count = len(agents)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        for food_pos in food_positions:
            pygame.draw.circle(screen, (0, 255, 0), food_pos.astype(int), FOOD_RADIUS)
    
        new_obstacle_positions = []
        for obstacle_pos in obstacle_positions:
            if (( obstacle_pos[0] < (( WIDTH / 2 ) - CENTER_MARGIN) or obstacle_pos[0] > (( WIDTH / 2 ) + CENTER_MARGIN)) or (obstacle_pos[1] < (( HEIGHT / 2 ) - CENTER_MARGIN) or obstacle_pos[1] > (( HEIGHT / 2 ) + CENTER_MARGIN)) ):
                pygame.draw.circle(screen, (255, 0, 0), obstacle_pos.astype(int), OBSTACLE_RADIUS)
                new_obstacle_positions.append(obstacle_pos)
        obstacle_positions = np.array(new_obstacle_positions)

        for agent in agents:
            agent.update(screen)
            agent.draw(screen)

            food_distances = np.linalg.norm(food_positions - agent.position, axis=1)
            if np.min(food_distances) < FOOD_RADIUS + AGENT_RADIUS:
                agent.energy += 50
                agent.genome.fitness += 50
                food_positions = np.delete(food_positions, np.argmin(food_distances), axis=0)

            obstacle_distances = np.linalg.norm(obstacle_positions - agent.position, axis=1)
            if np.min(obstacle_distances) < OBSTACLE_RADIUS + AGENT_RADIUS:
                agent.energy -= 100
                agent.genome.fitness -= 500

            if agent.energy <= 0:
                agents.remove(agent)
                population_count -= 1

        if len(food_positions) < NUM_FOOD:
            new_food_positions = np.random.rand(NUM_FOOD - len(food_positions), 2) * np.array([WIDTH, HEIGHT])
            food_positions = np.concatenate((food_positions, new_food_positions))

        font = pygame.font.Font(None, 36)
        population_text = font.render(f"Population: {population_count}", True, (255, 255, 255))
        screen.blit(population_text, (WIDTH - 200, 20))

        pygame.display.flip()
        clock.tick(60)

        if population_count == 0:
            break

    pygame.quit()

# Load configuration
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-neat')

# Create population
population = neat.Population(config)

# Add reporter
population.add_reporter(neat.StdOutReporter(True))

# Run NEAT
winner = population.run(eval_genomes)