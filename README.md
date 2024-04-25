# Machine Learning Trading Bot

This collection of Python scripts integrates a reinforcement learning agent, specifically a Deep Q-Network (DQN), to automate trading strategies on Centralized Exchanges. Below, you will find detailed descriptions of each component, setup instructions, and usage details.

## Project Structure

The repository is organized into several key scripts, each fulfilling a distinct role within the project:

### `dqn_agent.py`

This script is responsible for defining the architecture of the DQN agent. It includes:

- A neural network model for making trading decisions.
- Methods for action selection (`choose_action`), updating the model (`learn`), and saving/loading the model.
- Implementation of the replay memory to store previous experiences, enabling the agent to learn from past actions.

### `trading_environment.py`

This script defines the environment in which the trading bot operates. Key features include:

- A simulated market environment where the agent can test trading strategies.
- Reward calculations based on the profitability of trades.
- Methods to reset the environment and provide the agent with new market data.

### `trading_bot.py`

This is the main executable script that integrates the DQN agent with the trading environment. Its features include:

- Initialization of the DQN agent and the trading environment.
- A loop to continuously make trading decisions based on the current market state.
- Performance tracking and output to monitor the bot's decisions and successes over time.

### `bot_config.py`

Contains all configuration parameters for the bot, such as:

- Settings for the DQN agent (e.g., learning rate, discount factor).
- Trading environment settings (e.g., initial capital, transaction costs).
- Operational settings such as model save paths and logging configurations.

## Setup Instructions

To set up and run the trading bot, follow these steps:

1. **Install Python**: Ensure Python 3.8 or later is installed on your system.

2. **Clone the Repository**: Download or clone this repository to your local machine.

3. **Install Dependencies**: Navigate to the project directory and install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Settings**: Modify the `bot_config.py` file according to your trading preferences and agent parameters.

5. **Run the Bot**:

    ```bash
    python trading_bot.py
    ```

    This will start the trading simulation. Outputs and logs will be generated based on the configurations set in `bot_config.py`.

## Security Considerations

Ensure your private keys are stored securely and never hard-coded directly into your configuration files. Use environment variables or secure key management solutions.

## Disclaimer

This bot is for educational and development purposes only.

