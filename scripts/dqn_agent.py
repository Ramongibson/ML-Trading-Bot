import atexit
import numpy as np
from datetime import datetime
import random
from keras import layers, models, optimizers
from keras.layers import Dropout
from collections import deque
from config.bot_config import Config
import time
import json
import jsonpickle
import joblib
from sklearn.metrics import mean_squared_error
from keras.callbacks import TensorBoard
import tensorflow as tf


class DQNAgent:
    def __init__(
        self,
        log,
        state_size,
        action_size,
        learning_rate,
        gamma,
        epsilon_decay,
        epsilon,
        min_epsilon,
        epoch,
        dropout,
        replay_memory_size,
        batch_size,
        symbol="",
    ):
        self.log = log
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.epoch = epoch
        self.dropout = dropout
        self.replay_memory_maxlen = deque(maxlen=replay_memory_size)
        self.replay_memory = None
        self.batch_size = batch_size
        self.symbol = symbol.replace("/", "_")  # Store the symbol name
        self.model_filename = f"{Config.MODEL_FILENAME}{self.symbol}.h5"
        self.replay_memory_filename = (
            f"{Config.REPLAY_MEMORY_FILENAME}{self.symbol}.json"
        )
        self.learning_rate = learning_rate
        self.model = None
        self.target_model = self.build_model()
        # atexit.register(self.save_on_exit)
        self.reset_agent()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(64, input_dim=self.state_size, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.action_size, activation="linear"))
        model.add(Dropout(self.dropout))
        optimizer = optimizers.Adam(learning_rate=self.learning_rate)  # Updated line
        model.compile(optimizer=optimizer, loss="mse")
        return model

    def update_target_model(self):
        target_weights = [weight.numpy() for weight in self.model.weights]
        self.target_model.set_weights(target_weights)

    def choose_action(self, state):
        if Config.RUN_MODE != "production" or Config.RUN_MODE != "backtest":
            if np.random.rand() <= self.epsilon:
                return random.randrange(self.action_size)
            else:
                q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
                return np.argmax(q_values[0])
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def create_data_pipeline(self, features, labels, batch_size, buffer_size=10000):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(buffer_size)
        return dataset

    def train(self):
        self.log.info("Starting training...")
        for episode in range(Config.NUM_EPISODES):
            states, targets = [], []

            for state, action, reward, next_state, done in self.replay_memory:
                target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)[0]

                if done:
                    target[action] = reward
                else:
                    Q_future = max(
                        self.target_model.predict(
                            np.expand_dims(next_state, axis=0), verbose=0
                        )[0]
                    )
                    self.log.debug(Q_future)
                    target[action] = reward + self.gamma * Q_future
                    self.debug.info(target[action])

                states.append(state)
                targets.append(target)

            # Create a parallel data loading pipeline
            dataset = self.create_data_pipeline(
                np.array(states), np.array(targets), self.batch_size
            )
            self.log.info("data buffered")

            # Define a callback to log information for TensorBoard
            tensorboard_callback = TensorBoard(log_dir="./logs", histogram_freq=1)

            # Perform a single update using the batch
            history = self.model.fit(
                np.array(states),
                np.array(targets),
                epochs=self.epoch,
                callbacks=[tensorboard_callback],
                verbose=2,
            )
            # Calculate mean squared error
            mse = mean_squared_error(
                np.array(targets), self.model.predict(np.array(states), verbose=0)
            )

            # Log MSE and other training-related information
            self.log.info(f"Training MSE: {mse}")

            self.log.info(f"Epochs: {self.epoch}")
            self.log.info(f"Final Loss: {history.history['loss'][-1]}")
            self.log.info(f"Episode: {episode + 1}/{Config.NUM_EPISODES}")

            # Save the model and replay memory after each batch
            self.update_target_model()
            self.save_model()

            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay

            # Update the LAST_RETRAIN_TIME after completing retraining
            Config.LAST_RETRAIN_TIME = int(time.time())

    def save_on_exit(self):
        if Config.RUN_MODE != "study":
            # Save model
            self.save_model()
            self.log.info(f"Model saved as: {self.model_filename}")

            # Save replay memory

            self.save_replay_memory()
            self.log.info(f"Replay memory saved as: {self.replay_memory_filename}")

    def save_replay_memory(self):
        if Config.RUN_MODE != "study":
            with open(self.replay_memory_filename, "w") as file:
                replay_memory_serializable = [
                    jsonpickle.encode(entry) for entry in self.replay_memory
                ]
                file.write(json.dumps(replay_memory_serializable))

            self.log.info(f"Replay memory saved as: {self.replay_memory_filename}")

    def save_model(self):
        if (
            Config.SAVE_MODEL
            and Config.RUN_MODE != "study"
            and Config.RUN_MODE != "backtest"
        ):
            joblib.dump(self.model, self.model_filename)
            self.log.info(f"Model saved as: {self.model_filename}")

    def load_model(self):
        if Config.RUN_MODE != "study":
            try:
                loaded_model = joblib.load(self.model_filename)
                self.log.info(f"Model loaded successfully from {self.model_filename}")
                Config.LAST_MODEL_LOAD_TIME = int(time.time())
                return loaded_model
            except (OSError, IOError):
                self.log.info(
                    f"No model found at {self.model_filename}. Starting fresh."
                )
                return self.build_model()
        return self.build_model()

    def load_replay_memory(self):
        if Config.RUN_MODE == "resume":
            try:
                with open(self.replay_memory_filename, "r") as f:
                    replay_memory_serializable = json.load(f)
                    replay_memory_data = [
                        jsonpickle.decode(entry) for entry in replay_memory_serializable
                    ]
                    self.log.info(
                        f"Replay memory loaded successfully from {self.replay_memory_filename}"
                    )
                    return deque(replay_memory_data)
            except (OSError, IOError, json.JSONDecodeError) as e:
                self.log.warn(
                    f"Error loading replay memory from {self.replay_memory_filename}: {str(e)}"
                )
                return self.replay_memory_maxlen
        return self.replay_memory_maxlen

    def reset_agent(self):
        """
        Reset the internal state of the DQNAgent.
        """
        self.model = self.load_model()
        self.update_target_model()
        self.epsilon = self.init_epsilon  # Reset exploration parameter
        self.replay_memory = self.load_replay_memory()
        if Config.RUN_MODE != "resume":
            self.replay_memory.clear()
            self.log.info("Replay memory cleared")
        self.log.info("Agent reset successfully.")

    def retrain_agent(self, agent, trading_env):
        current_timestamp = int(time.time())

        # Check if it's time to retrain
        if current_timestamp >= Config.LAST_RETRAIN_TIME + Config.RETRAIN_INTERVAL:
            self.log.info("Not yet time to retrain. Skipping retraining.")
            return

        # Fetch the latest historical data for retraining
        historical_data = trading_env.fetch_candlestick_data(
            trading_env.exchange,
            trading_env.symbol,
            trading_env.timeframe,
            start_date=Config.START_DATE,
            end_date=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        # Update the raw data in the trading environment
        trading_env.raw_data = historical_data

        # Train the DQN agent on the updated data
        for _ in range(Config.RETRAIN_NUM_EPISODES):
            state = trading_env.reset(update_data=False)
            total_reward = 0

            while not trading_env.done:
                action = agent.choose_action(state)
                next_state, reward, done = trading_env.step(action, retrain=True)
                if agent.is_good_memory(reward):
                    agent.replay_memory.append(
                        (state, action, reward, next_state, done)
                    )
                state = next_state
                total_reward += reward
            agent.train()
            self.log.info(f"Retraining Episode - Total Reward: {total_reward}")

    def backtest_dqn_agent(self, agent, env):
        state = env.get_state()
        total_reward = 0
        actions_taken = {"buy": 0, "sell": 0, "hold": 0}
        trades = []

        while not env.done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            state = next_state

            # Record actions taken during backtesting
            if action == 0:
                actions_taken["buy"] += 1
                trades.append(
                    {
                        "action": "buy",
                        "timestamp": env.raw_data[env.current_step]["timestamp"],
                    }
                )
            elif action == 1:
                actions_taken["sell"] += 1
                trades.append(
                    {
                        "action": "sell",
                        "timestamp": env.raw_data[env.current_step]["timestamp"],
                    }
                )
            else:
                actions_taken["hold"] += 1

        # Log individual trades
        self.log.info("Individual Trades:")
        for trade in trades:
            self.log.debug(
                f"Action: {trade['action']}, Timestamp: {trade['timestamp']}"
            )

        # Log detailed backtesting results
        self.log.info(f"Backtesting Results - Total Reward: {total_reward}")
        self.log.info(f"Initial Portfolio Balance: {env.initial_balance}")
        self.log.info(f"Initial Portfolio Value: {env.initial_profoilio_value}")
        self.log.info(f"Final Portfolio Balance: {env.current_balance}")
        self.log.info(f"Total Actions Taken: {sum(actions_taken.values())}")
        self.log.info(f"Number of Buy Actions: {actions_taken['buy']}")
        self.log.info(f"Number of Sell Actions: {actions_taken['sell']}")
        self.log.info(f"Number of Hold Actions: {actions_taken['hold']}")
