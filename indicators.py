import pandas as pd
import pandas_ta as pdta
import numpy as np
from finta import TA
from sklearn.cluster import KMeans
from helpers.class_helpers import Helpers
from scipy.signal import argrelextrema

class Trend:
    Extreme_Bullish = 3
    Bullish = 1
    Bearish = -1
    Sideways = 0
    Neutral = 2

class BotIndicators():
    def __init__(self, bot_obj):
        Helpers.__init__(self, name="BotIndicators", bot_obj=bot_obj)
        self.bot_obj = bot_obj

    def scalping_strategy_result(
        self, data: pd.DataFrame, stop_loss_percent=0.99, take_profit_percent=1.015
    ) -> str:
        data = self.heikin_ashi(data)
        current_price = data["Close"].tail(1).item()
        market_cipher = self.market_cipher(data)
        stop_loss = current_price * stop_loss_percent
        take_profit = current_price * take_profit_percent
        
        return {
            "type": market_cipher,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
        }

    def market_cipher(self, data):
        current_price = data["Close"].tail(1).item()
        _, merged_extrema_prices, __ = self.determine_support_resistance(data)
        support, resistance = self.find_support_resistance_levels(merged_extrema_prices, current_price)
        signal = self.check_price_at_support_resistance(current_price, support, resistance)
        
        bullish_condition = signal == Trend.Bullish
        bearish_condition = signal == Trend.Bearish
            
        if bullish_condition:
            return Trend.Bullish
        elif bearish_condition:
            return Trend.Bearish
        else:
            return Trend.Neutral

    @staticmethod
    def hull_moving_average(df, window=30):
        return TA.HMA(df, window)
    
    def determine_support_resistance(self, ohlcv, volume_threshold=1, num_clusters=6, distance_threshold=1):
        # Ensure 'Volume' column is numeric
        ohlcv['Volume'] = pd.to_numeric(ohlcv['Volume'], errors='coerce')

        # Calculate Hull Moving Average
        close_ma = self.hull_moving_average(ohlcv)

        # Convert the moving average to a numpy array and flatten it
        data = close_ma.to_numpy().flatten()
        original_data = data.copy()

        # Find turning points
        maxima = argrelextrema(data, np.greater)
        minima = argrelextrema(data, np.less)
        extrema = np.concatenate((maxima, minima), axis=1)[0]
        extrema_prices = np.concatenate((data[maxima], data[minima]))

        # Calculate average traded volume
        average_volume = ohlcv['Volume'].mean()

        # Filter out levels based on volume
        filtered_extrema = []
        filtered_extrema_prices = []

        i = 0
        while i < len(extrema) - 1:
            current_volume = ohlcv['Volume'].iloc[extrema[i]]

            if pd.notna(current_volume) and current_volume >= volume_threshold * average_volume and abs(extrema[i] - extrema[i + 1]) > distance_threshold:
                filtered_extrema.append(extrema[i])
                filtered_extrema_prices.append(extrema_prices[i])

            i += 1

        # Add the last extremum if not processed
        if i == len(extrema) - 1:
            filtered_extrema.append(extrema[-1])
            filtered_extrema_prices.append(extrema_prices[-1])

        filtered_extrema = np.array(filtered_extrema)
        filtered_extrema_prices = np.array(filtered_extrema_prices)

        # Create a feature matrix with position and volume
        feature_matrix = np.column_stack((filtered_extrema, filtered_extrema_prices))
        if len(feature_matrix) < num_clusters:
            print("Not enough samples for the specified number of clusters.")
            return None, None, None
        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
        cluster_labels = kmeans.fit_predict(feature_matrix)

        # Merge levels within each cluster
        merged_extrema = []
        merged_extrema_prices = []

        for cluster_id in range(num_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                # Merge levels within the cluster by taking the average position and volume
                merged_index = int(np.mean(filtered_extrema[cluster_indices]))
                merged_price = np.mean(filtered_extrema_prices[cluster_indices])

                merged_extrema.append(merged_index)
                merged_extrema_prices.append(merged_price)

        merged_extrema = np.array(merged_extrema)
        merged_extrema_prices = np.array(merged_extrema_prices)

        return merged_extrema, merged_extrema_prices, original_data
    
    @staticmethod
    def check_price_at_support_resistance(current_price, next_support, next_resistance, tolerance=0.0005):
        """
        Check if the current price is approaching or very close to the next support or resistance levels.

        Parameters:
        - current_price (float): The current market price.
        - next_support (float): The next support level.
        - next_resistance (float): The next resistance level.
        - tolerance (float): Tolerance range to consider the price as approaching the levels.

        Returns:
        - signal (str): 'Buy', 'Sell', or 'No Signal'.
        """

        if next_support is not None and current_price <= next_support * (1 + tolerance):
            return Trend.Bullish  # Signal to buy when approaching or very close to the support level
        elif next_resistance is not None and current_price >= next_resistance * (1 - tolerance):
            return Trend.Bearish  # Signal to sell when approaching or very close to the resistance level
        else:
            return Trend.Neutral  # No significant movement, no signal to issue
    
    
    def find_support_resistance_levels(self, merged_extrema_prices, current_price, min_percentage_difference=1.0):
        if merged_extrema_prices is None or len(merged_extrema_prices) == 0:
            print("No merged extrema prices available.")
            return None, None
        
        # Sort merged_extrema_prices in ascending order
        sorted_extrema_prices = np.sort(merged_extrema_prices)

        # Find the closest value in sorted_extrema_prices above and below the current price
        resistance_candidates = sorted_extrema_prices[sorted_extrema_prices > current_price]
        support_candidates = sorted_extrema_prices[sorted_extrema_prices < current_price]

        # Check if there are valid candidates for support and resistance
        resistance = self.find_valid_level(resistance_candidates, min_percentage_difference)
        support = self.find_valid_level(support_candidates, min_percentage_difference)

        return support, resistance

    @staticmethod
    def find_valid_level(candidates, min_percentage_difference):
        sorted_candidates = np.sort(candidates)
        n = len(sorted_candidates)
        
        if n == 0:
            return None

        for i in range(n - 1):
            if (sorted_candidates[i + 1] - sorted_candidates[i]) >= min_percentage_difference / 100 * sorted_candidates[i]:
                return sorted_candidates[i]

        # If no valid level found, return the last candidate
        return sorted_candidates[-1]

    def has_min_difference(candidates, min_percentage_difference):
        return np.ptp(candidates) >= min_percentage_difference / 100 * candidates.max()

    def heikin_ashi(self, df):
        ha_df = pd.DataFrame()
        ha_df["Open"] = (df["Open"] + df["Close"]) / 2
        ha_df["High"] = df[["High", "Open", "Close"]].max(axis=1)
        ha_df["Low"] = df[["Low", "Open", "Close"]].min(axis=1)
        ha_df["Close"] = (df["Open"] + df["High"] + df["Low"] + df["Close"]) / 4
        ha_df["Volume"] = df["Volume"]
        return ha_df

