# Artificial Life Simulation with NEAT-Python

This program creates an artificial environment where neural-network-powered agents (simple artificial lifeforms) learn to compete, cooperate, and interact using the NEAT-python library.

## Installation

1. Install Python 3.x from the official website: https://www.python.org/downloads/

2. Install the required libraries using pip:

   ```
   pip install -r requirements.txt
   ```

## Configuration

The program uses a configuration file named `config-neat` to define the settings for the NEAT algorithm. You can modify this file to adjust the parameters of the neural networks and the evolutionary process.

## Running the Program

   ```
   python app.py
   ```

## Customization

You can customize the program by modifying the constants at the top of the code:

- `WIDTH` and `HEIGHT`: The dimensions of the simulation window.
- `FOOD_RADIUS`, `AGENT_RADIUS`, and `OBSTACLE_RADIUS`: The sizes of the food, agents, and obstacles.
- `NUM_FOOD` and `NUM_OBSTACLES`: The number of food items and obstacles in the environment.

You can also experiment with different neural network architectures and configurations by modifying the `config-neat` file.
