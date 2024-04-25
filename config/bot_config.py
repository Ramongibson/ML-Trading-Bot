import time
import datetime


class Config:
    # Binance API key for authentication
    BINANCE_API_KEY = "T8s9BPuR7EFGrU57wSa71ZggRqQcNk8aXTppu6n7YOvVSxjmuJI7hnsvlGb9lr37"

    # Binance API secret for authentication
    BINANCE_API_SECRET = ""

    # Trading symbol on Binance (e.g., "BTC/USD")
    SYMBOL = "BTC/USDT"

    # Timeframe for candlestick data (e.g., "1m" for 1-minute)
    TIMEFRAME = "1m"

    # Start date for fetching historical data
    FETCH_FROM_DATE = "2019-01-25"

    FETCH_TO_DATE = None

    # Start date for fetching historical data
    TRAIN_START_DATE = None

    # Number of training episodes for the DQN agent
    NUM_EPISODES = 10

    # Run mode ("train", "production", "study", "resume" or "backtest")
    RUN_MODE = "backtest"

    MODEL_FILENAME = f"model/trained_model_{FETCH_FROM_DATE}_"

    REPLAY_MEMORY_FILENAME = f"replay/replay_memory_{FETCH_FROM_DATE}_"

    DATA_FILENAME = f"data/historical_data_{FETCH_FROM_DATE}.json"

    SAVE_MODEL = True

    # Initial balance for the trading environment
    INITIAL_QOUTE_BALANCE = 100

    INITIAL_BASE_BALANCE = 0.0024

    # Interval for retraining the DQN agent (in seconds)
    RETRAIN_INTERVAL = 3600

    # Number of episodes used for retraining the DQN agent
    RETRAIN_NUM_EPISODES = 1

    # Timestamp of the last retraining
    LAST_RETRAIN_TIME = int(time.time())

    LOAD_MODEL_INTERVAL = 1800

    LAST_MODEL_LOAD_TIME = None

    # Interval for live trading actions (in seconds)
    LIVE_TRADE_INTERVAL = 30

    BALANCE_UPDATE_INTERVAL = 1

    INVEST_PERCENTAGE = 0.90

    MAX_POSITION_SIZE = 0.0008

    DATA_FETCH_LIMIT = 1500

    STUDY_TRAILS = 100

    N_STARTUP_TRIALS = 10

    MIN_TRADE_AMOUNT = 0.0001

    MONGODB_URI = (
        "mongodb+srv://cluster0.cnoytkg.mongodb.net/?retryWrites=true&w=majority"
    )

    MONGODB_DATABASE = "TRADING_BOT"

    MONGODB_COLLECTION = "REPLAY_MEMORY"

    MIN_REWARD_THRESHOLD = 1
