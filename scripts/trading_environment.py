import ccxt
import numpy as np
from datetime import datetime, timedelta
import logging
from config.bot_config import Config
import time
import json
from sklearn.preprocessing import MinMaxScaler


class TradingEnvironment:
    def __init__(self, log, exchange, symbol, timeframe):
        self.log = log
        self.exchange = exchange
        self.symbol = symbol
        self.timeframe = timeframe
        self.raw_data = []
        self.symbol_base, self.symbol_quote = symbol.split(
            "/"
        )  # Split symbol into base and quote
        self.current_step = 0
        self.initial_balance = 0.0
        self.initial_qoute_balance = 0.0
        self.initial_base_balance = 0.0
        self.initial_profoilio_value = 0.0
        self.current_balance = 0.0
        self.qoute_balance = 0.0
        self.base_balance = 0.0
        self.exchange_rate = 0.0
        self.profit = 0.0
        if Config.RUN_MODE == "production":
            self.update_balance_from_exchange()
            self.market = self.load_market()
        self.done = False
        self.reset()

    def fetch_candlestick_data(self, exchange, symbol, timeframe, start_date, end_date):
        all_data = []

        # Format start and end dates
        start_date = f"{start_date}T00:00:00Z"
        if end_date is None:
            end_date = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            end_date = f"{end_date}T00:00:00Z"

        self.log.info(f"Provided start_date: {start_date}, end_date: {end_date}")

        try:
            since = exchange.parse8601(start_date)
            self.log.info(f"since {type(since)}: {since}")
            end_timestamp = exchange.parse8601(end_date)
            self.log.info(f"end_timestamp {type(end_timestamp)}: {end_timestamp}")

        except Exception as e:
            self.log.error(f"Failed to parse start or end date. Error: {e}")
            return []

        if since is None or end_timestamp is None:
            self.log.error("Parsed date is None. Please check your date formats.")
            return []

        save_file_path = Config.DATA_FILENAME
        existing_data = []
        try:
            with open(save_file_path, "r") as json_file:
                existing_data = json.load(json_file)
                if existing_data:
                    last_timestamp_str = existing_data[-1]["timestamp"]
                    last_timestamp = datetime.strptime(
                        last_timestamp_str, "%Y-%m-%dT%H:%M:%SZ"
                    ).timestamp()
                    last_timestamp_in_milliseconds = int(last_timestamp * 1000)
                    since = max(last_timestamp_in_milliseconds, since)
        except FileNotFoundError:
            existing_data = []
            self.log.warning("No existing data found. Starting fresh.")

        while since < end_timestamp:
            try:
                ohlcv = exchange.fetch_ohlcv(
                    symbol, timeframe, since, limit=Config.DATA_FETCH_LIMIT
                )
            except ccxt.NetworkError as e:
                self.log.error(f"Network error while fetching historical data: {e}")
                raise
            except ccxt.ExchangeError as e:
                self.log.error(f"Exchange error while fetching historical data: {e}")
                raise
            except Exception as e:
                self.log.error(f"An unexpected error occurred: {e}")
                raise

            if not ohlcv:
                self.log.warning(
                    f"No OHLCV data fetched for the specified date range at timestamp {since}."
                )
                break

            all_data.extend(
                {
                    "timestamp": datetime.utcfromtimestamp(candle[0] / 1000).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "open": candle[1],
                    "high": candle[2],
                    "low": candle[3],
                    "close": candle[4],
                    "volume": candle[5],
                    "exchange_rate": candle[4],
                }
                for candle in ohlcv
            )

            since = ohlcv[-1][0] + 1

        if all_data:
            with open(save_file_path, "w") as json_file:
                if all_data and existing_data:
                    self.log.debug("Exisiting and new data found")
                    # Append new data to the existing data
                    existing_data.extend(all_data)

                    # Seek to the beginning of the file
                    json_file.seek(0)

                    # Save the updated data as a JSON array
                    json.dump(existing_data, json_file)
                    json_file.truncate()
                    return existing_data
                elif all_data:
                    self.log.debug("Using new data only")
                    json_file.seek(0)
                    json.dump(all_data, json_file)
                    json_file.truncate()
                    return all_data
        else:
            self.log.debug("No new data found. Using exisitng.")
            return existing_data

    def fetch_live_data(self):
        try:
            live_data = self.exchange.fetch_ticker(self.symbol)
            exchange_rate = live_data.get("ask", 1.0)
            volume = 0.0
            if "volume" in live_data and live_data["volume"] is not None:
                volume = live_data["volume"]
            live_candle = {
                "timestamp": datetime.utcfromtimestamp(live_data["timestamp"] / 1000),
                "open": live_data["open"],
                "high": live_data["high"],
                "low": live_data["low"],
                "close": live_data["close"],
                "volume": volume,
                "exchange_rate": exchange_rate,
            }
            return live_candle
        except Exception as e:
            self.log.error(f"Failed to fetch live data. Error: {e}")
            return None

    def load_market(self):
        self.exchange.load_markets()
        return self.exchange.market(self.symbol)

    def update_live_data(self):
        live_candle = self.fetch_live_data()
        if live_candle:
            self.raw_data.append(live_candle)

    def reset(self, update_data=True):
        if update_data and Config.RUN_MODE != "production":
            self.log.info("Updating data...")

            # Fetch raw data including data from TRAIN_START_DATE
            self.raw_data = self.fetch_candlestick_data(
                self.exchange,
                self.symbol,
                self.timeframe,
                start_date=Config.FETCH_FROM_DATE,
                end_date=Config.FETCH_TO_DATE,
            )
            if Config.TRAIN_START_DATE is not None:
                # Filter raw data based on TRAIN_START_DATE
                self.raw_data = [
                    candle
                    for candle in self.raw_data
                    if self.exchange.parse8601(candle["timestamp"])
                    >= self.exchange.parse8601(f"{Config.TRAIN_START_DATE}T00:00:00Z")
                ]
                self.log.info(f"Training data filter from {Config.TRAIN_START_DATE}")
            self.log.info("Data updated successfully.")
        elif update_data and Config.RUN_MODE == "production":
            self.log.debug("Updating live data...")
            self.update_live_data()
            self.log.debug("Live data updated successfully.")

        self.set_initial_balances()
        if not self.raw_data:
            self.log.warning(
                "Empty data fetched when resetting. Consider adjusting date range."
            )
            self.done = True
            return None

        return self.get_state()

    def set_initial_balances(self):
        if Config.RUN_MODE != "production":
            self.initial_base_balance = Config.INITIAL_BASE_BALANCE
            self.initial_qoute_balance = Config.INITIAL_QOUTE_BALANCE
            self.base_balance = Config.INITIAL_BASE_BALANCE
            self.qoute_balance = Config.INITIAL_QOUTE_BALANCE
            self.exchange_rate = self.raw_data[self.current_step]["close"]
            self.initial_balance = (
                Config.INITIAL_BASE_BALANCE * self.exchange_rate
            ) + Config.INITIAL_QOUTE_BALANCE
            self.current_balance = 0.0
            self.current_step = 0
            self.done = False
            self.initial_profoilio_value = 0.0
            self.profit = 0.0

        else:
            self.initial_base_balance = self.base_balance
            self.initial_qoute_balance = self.qoute_balance
            self.initial_balance = (
                self.base_balance * self.exchange_rate
            ) + self.qoute_balance
            self.current_balance = 0.0
            self.current_step = 0
            self.done = False
            self.initial_profoilio_value = 0.0
            self.profit = 0.0

    def update_balance_from_exchange(self):
        try:
            # Fetch the current account balance from the exchange API
            balance_info = self.exchange.fetch_balance()

            # Get the balance for the base currency of the trading pair
            base_balance = balance_info["free"].get(self.symbol_base, 0.0)
            if base_balance is None:
                base_balance = 0.0

            # Get the balance for the quote currency of the trading pair
            quote_balance = balance_info["free"].get(self.symbol_quote, 0.0)
            if quote_balance is None:
                quote_balance = 0.0

            # Fetch the exchange rate between quote and base currencies
            ticker = self.exchange.fetch_ticker(self.symbol)
            exchange_rate = ticker.get(
                "ask"
            )  # You may use bid, ask, or other rates depending on your needs

            self.exchange_rate = exchange_rate
            self.qoute_balance = quote_balance
            self.base_balance = base_balance
        except Exception as e:
            self.log.error(f"Failed to update balance from exchange. Error: {e}")

    def get_state(self):
        portfolio_state = np.array(
            [
                self.current_balance,
                self.exchange_rate,
                self.initial_profoilio_value,
            ]
        )

        timestamp_str = self.raw_data[self.current_step]["timestamp"]

        # Convert the string timestamp to a datetime object
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")

        # Create the NumPy array
        market_state = np.array(
            [
                timestamp.timestamp(),
                float(self.raw_data[self.current_step]["open"]),
                float(self.raw_data[self.current_step]["high"]),
                float(self.raw_data[self.current_step]["low"]),
                float(self.raw_data[self.current_step]["close"]),
                float(self.raw_data[self.current_step]["volume"]),
            ]
        )

        return np.concatenate((portfolio_state, market_state))

    def normalize_state(self, state):
        # Extract portfolio and market components from the state
        portfolio_state = state[:5]
        market_state = state[5:]

        # Normalize portfolio state
        scaler_portfolio = MinMaxScaler()
        normalized_portfolio_state = scaler_portfolio.fit_transform(
            portfolio_state.reshape(1, -1)
        ).flatten()

        # Normalize market state
        scaler_market = MinMaxScaler()
        normalized_market_state = scaler_market.fit_transform(
            market_state.reshape(1, -1)
        ).flatten()

        # Concatenate normalized portfolio and market states
        normalized_state = np.concatenate(
            (normalized_portfolio_state, normalized_market_state)
        )

        return normalized_state

    def get_normalized_state(self):
        # Get the original state
        state = self.get_state()

        # Normalize the state
        normalized_state = self.normalize_state(state)

        return normalized_state

    def calculate_position_size(self, action):
        if action == 0:
            return Config.INVEST_PERCENTAGE * (self.qoute_balance / self.exchange_rate)
        return self.base_balance

    def take_action(self, action, retrain=False):
        # Only execute live trades in production mode
        if Config.RUN_MODE == "production":
            base_currency, quote_currency = self.symbol.split("/")
            if action == 0:  # Buy
                # Calculate quantity based on available quote currency balance
                if self.qoute_balance > 0:
                    position_size = self.calculate_position_size(action)
                    max_position = min(position_size, Config.MAX_POSITION_SIZE)
                    amount = self.exchange.currency_to_precision(
                        base_currency, max_position
                    )
                    if max_position >= Config.MIN_TRADE_AMOUNT:
                        order = self.exchange.create_market_buy_order(
                            self.symbol, amount
                        )
                        self.log.info(f"Buy Order Executed")
                        time.sleep(Config.BALANCE_UPDATE_INTERVAL)
                        self.update_balance_from_exchange()  # Update balance after the order execution
                else:
                    self.log.warning("Quote balance is 0. Skipping buy order.")

            elif action == 1:  # Sell
                if self.base_balance > 0:
                    position_size = self.calculate_position_size(action)
                    amount = self.exchange.currency_to_precision(
                        base_currency, position_size
                    )
                    if position_size >= Config.MIN_TRADE_AMOUNT:
                        order = self.exchange.create_market_sell_order(
                            self.symbol, amount
                        )
                        self.log.info(f"Sell Order Executed")
                        time.sleep(Config.BALANCE_UPDATE_INTERVAL)
                        self.update_balance_from_exchange()  # Update balance after the order execution
                else:
                    self.log.warning("Base balance is 0. Skipping sell order.")

            elif action == 2:  # Hold
                self.log.info("Hold Executed")

        elif (
            Config.RUN_MODE == "train"
            or Config.RUN_MODE == "backtest"
            or Config.RUN_MODE == "study"
            or retrain == True
        ):
            self.exchange_rate = self.raw_data[self.current_step]["close"]
            # For training and backtest modes, simulate the action without executing a live trade
            if action == 0:  # Buy
                if self.qoute_balance > 0:
                    position_size = self.calculate_position_size(action)
                    max_position = min(position_size, Config.MAX_POSITION_SIZE)
                    if max_position >= Config.MIN_TRADE_AMOUNT:
                        order = {
                            "cost": Config.INVEST_PERCENTAGE
                            * (max_position * self.exchange_rate),
                            "filled": Config.INVEST_PERCENTAGE * max_position,
                        }
                        self.qoute_balance -= order["cost"]
                        self.base_balance += order["filled"]
                else:
                    self.log.debug("Quote balance is 0. Skipping buy order.")

            elif action == 1:  # Sell
                if self.base_balance > 0:
                    if self.base_balance >= Config.MIN_TRADE_AMOUNT:
                        order = {
                            "cost": self.base_balance * self.exchange_rate,
                            "filled": self.base_balance,
                        }
                    self.qoute_balance += order["cost"]
                    self.base_balance -= order["filled"]
                else:
                    self.log.debug("Base balance is 0. Skipping sell order.")

            elif action == 2:  # Hold
                self.log.debug("Hold Executed")

            self.current_balance = (
                self.base_balance * self.exchange_rate
            ) + self.qoute_balance

    def calculate_reward(self, action):
        self.initial_profoilio_value = (
            self.initial_base_balance * self.exchange_rate + self.initial_qoute_balance
        )

        profit = self.current_balance - self.initial_profoilio_value
        self.profit += profit
        reward = 0
        if profit > 0:
            reward = 1
        elif profit < 0:
            reward = -1
        return reward

    def step(self, action, retrain=False):
        self.take_action(action, retrain)
        reward = self.calculate_reward(action)
        logging.debug(f"Step: {self.current_step}, Action: {action}, Reward: {reward}")
        self.done = self.current_step == len(self.raw_data) - 1
        if not self.done:
            self.current_step += 1
            self.exchange_rate = self.raw_data[self.current_step]["close"]
        return self.get_state(), reward, self.done

    def live_trade_step(self, action, retrain=False):
        self.take_action(action, retrain)
        reward = self.calculate_reward(action)
        time.sleep(Config.LIVE_TRADE_INTERVAL)
        return self.reset(), reward, False

    def live_trade(self, agent, env, log):
        try:
            state = env.get_state()
            action = agent.choose_action(state)
            next_state, reward, done = self.live_trade_step(action)
            agent.replay_memory.append((state, action, reward, next_state, done))
            state = next_state
            time.sleep(Config.LIVE_TRADE_INTERVAL)
        except Exception as e:
            log.error(f"Error during live trading: {e}")
