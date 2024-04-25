import ccxt
import numpy as np
import logging
from config.bot_config import Config
import time
import threading
import optuna
from optuna.pruners import MedianPruner
import json
from dqn_agent import DQNAgent
from scripts.trading_environment import TradingEnvironment
import urllib3

# import tensorflow as tf

import atexit


# Configure logging to a file and console
log_file_path = f"logs/trading_bot_{Config.RUN_MODE}.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Get the root logger
log = logging.getLogger()
log.setLevel(logging.INFO)
log.addHandler(console_handler)
log.info(f"Running in {Config.RUN_MODE} mode")

# CCXT logger setup
ccxt_logger = logging.getLogger("ccxt")
ccxt_logger.setLevel(logging.ERROR)
# Set requests to IPV4
urllib3.util.connection.HAS_IPV6 = False

# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
# physical_devices = tf.config.list_physical_devices('GPU')
# if physical_devices:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)


def retrain_agent_thread(agent, env, log):
    while True:
        try:
            # Retrain the agent
            agent.retrain_agent(agent, env, log)
            log.info("Retraining completed. Waiting for the next retrain interval.")
            time.sleep(Config.RETRAIN_INTERVAL)
        except Exception as e:
            log.error(f"Error during retraining: {e}")


def train_with_optuna(agent, trade_env, log):
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=Config.N_STARTUP_TRIALS),
    )

    def objective_wrapper(trial):
        # Reset total_rewards for each trial
        total_rewards = []

        # Use the hyperparameters for training
        state_size = len(trade_env.get_state())
        agent = DQNAgent(
            log,
            state_size,
            3,
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            gamma=trial.suggest_float(
                "gamma", 0.8, 1
            ),  # Higher gamma for more emphasis on future rewards
            epsilon_decay=trial.suggest_float("epsilon_decay", 0.0, 0.1),
            epsilon=trial.suggest_float("epsilon", 1, 2),
            min_epsilon=trial.suggest_float("min_epsilon", 0.01, 0.99),
            epoch=trial.suggest_int("epoch", 1, 20),
            dropout=trial.suggest_float("dropout", 0.2, 0.5),
            replay_memory_size=trial.suggest_int(
                "replay_memory_size", 3000000, 5000000
            ),
            batch_size=trial.suggest_int("batch_size", 64, 256),
            symbol=Config.SYMBOL,
        )

        state = trade_env.reset(update_data=False)
        total_reward = 0
        while not trade_env.done:
            action = agent.choose_action(state)
            next_state, reward, done = trade_env.step(action)
            agent.replay_memory.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
            total_rewards.append(total_reward)
        agent.save_replay_memory()
        agent.train()

        log.info(f"Trial {trial.number + 1} Total Reward: {total_reward}")
        log.info(f"Initial Portfolio Balance: {trade_env.initial_balance}")
        log.info(f"Initial Portfolio Value: {trade_env.initial_profoilio_value}")
        log.info(f"Final Portfolio Balance: {trade_env.current_balance}")

        # Return the negative sum of total rewards as the objective value
        return -np.sum(total_rewards)

    study.optimize(objective_wrapper, n_trials=Config.STUDY_TRAILS)

    # Log the best hyperparameters
    log.info("Best hyperparameters:")
    log.info(study.best_params)

    # Save the best hyperparameters to a file
    hyperparameters_file_path = "best_hyperparameters.json"
    with open(hyperparameters_file_path, "w") as f:
        json.dump(study.best_params, f)

    log.info(f"Best hyperparameters saved to {hyperparameters_file_path}")


# @tf.function
# @profile
def run():
    exchange = ccxt.binanceus(
        {
            "apiKey": Config.BINANCE_API_KEY,
            "secret": Config.BINANCE_API_SECRET,
            "enableRateLimit": True,
            "fetchMarkets": True,
        }
    )

    trade_env = TradingEnvironment(
        log,
        exchange,
        Config.SYMBOL,
        Config.TIMEFRAME,
    )

    state_size = len(trade_env.get_state())
    num_actions = 3

    # Load hyperparameters from the file
    hyperparameters_file_path = "config/hyperparameters_config.json"
    with open(hyperparameters_file_path, "r") as f:
        hyperparameters = json.load(f)

    # Initialize DQNAgent with hyperparameters from the file
    agent = DQNAgent(
        log,
        state_size,
        num_actions,
        learning_rate=hyperparameters["learning_rate"],
        gamma=hyperparameters["gamma"],
        epsilon_decay=hyperparameters["epsilon_decay"],
        epsilon=hyperparameters["epsilon"],
        min_epsilon=hyperparameters["min_epsilon"],
        epoch=hyperparameters["epoch"],
        dropout=hyperparameters["dropout"],
        replay_memory_size=hyperparameters["replay_memory_size"],
        batch_size=hyperparameters["batch_size"],
        symbol=Config.SYMBOL,
    )

    # atexit.register(agent.save_on_exit)

    if Config.RUN_MODE == "production":
        # Start the retraining thread
        retrain_thread = threading.Thread(
            target=retrain_agent_thread, args=(agent, trade_env, log)
        )
        retrain_thread.daemon = True
        retrain_thread.start()
        retrain_thread.join()
        while True:
            trade_env.live_trade(agent, trade_env, log)
            current_timestamp = int(time.time())
            # Check if it's time to retrain
            if (
                current_timestamp
                >= Config.LAST_MODEL_LOAD_TIME + Config.LOAD_MODEL_INTERVAL
            ):
                agent.load_model()

    elif Config.RUN_MODE == "study":
        train_with_optuna(agent, trade_env, log)

    elif Config.RUN_MODE == "train":
        total_reward = 0
        state = trade_env.reset(update_data=False)
        while not trade_env.done:
            action = agent.choose_action(state)
            next_state, reward, done = trade_env.step(action)
            agent.replay_memory.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state
        agent.save_replay_memory()
        agent.train()

        log.info(f"Initial Portfolio Balance: {trade_env.initial_balance}")
        log.info(f"Initial Portfolio Value: {trade_env.initial_profoilio_value}")
        log.info(f"Final Portfolio Balance: {trade_env.current_balance}")
        log.info(f"Profit: {trade_env.profit}")
        log.info(f"Episode Done: {trade_env.done}")

        if trade_env.done:
            log.info("Episode completed successfully.")
        else:
            log.warning("Episode did not complete.")

    elif Config.RUN_MODE == "backtest":
        agent.backtest_dqn_agent(agent, trade_env)

    elif Config.RUN_MODE == "resume":
        log.info("Resuming training")
        agent.train()

    else:
        log.warning(
            f"Invalid mode: {Config.RUN_MODE}. Please use 'production', 'train', 'study', or 'backtest'."
        )


if __name__ == "__main__":
    run()
