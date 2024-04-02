# Artificial Life Simulation with PyTorch-NEAT

This program creates an artificial environment where neural-network-powered agents (simple artificial lifeforms) learn to compete, cooperate, and interact using the PyTorch-NEAT library.

## Requirements

- Python 3.x
- PyTorch
- PyTorch-NEAT
- Pygame
- NumPy

## Installation

1. Install Python 3.x from the official website: https://www.python.org/downloads/

2. Install the required libraries using pip:

   ```
   pip install torch pytorch-neat pygame numpy
   ```

## Configuration

The program uses a configuration file named `config-hyperneat` to define the settings for the NEAT algorithm. You can modify this file to adjust the parameters of the neural networks and the evolutionary process.

## Running the Program

1. Save the program code in a Python file, for example, `artificial_life.py`.

2. Create a configuration file named `config-feedforward` in the same directory as the program file. You can use the default configuration or modify it according to your needs.

3. Open a terminal or command prompt and navigate to the directory where the program file is located.

4. Run the program using the following command:

   ```
   python artificial_life.py
   ```

5. The program will start the artificial life simulation. You will see a window displaying the agents (colored circles), food (red circles), and obstacles (black circles).

6. The agents will move around the environment, trying to collect food while avoiding obstacles. They will learn and evolve over time using the NEAT algorithm.

7. The simulation will continue until you close the window.

## Customization

You can customize the program by modifying the constants at the top of the code:

- `WIDTH` and `HEIGHT`: The dimensions of the simulation window.
- `FOOD_RADIUS`, `AGENT_RADIUS`, and `OBSTACLE_RADIUS`: The sizes of the food, agents, and obstacles.
- `NUM_FOOD` and `NUM_OBSTACLES`: The number of food items and obstacles in the environment.

You can also experiment with different neural network architectures and configurations by modifying the `config-feedforward` file.

## License

This program is released under the MIT License. Feel free to use, modify, and distribute it as per the terms of the license.