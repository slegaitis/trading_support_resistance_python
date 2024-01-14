# Python Get support and resistance levels

## Contributions welcome

This code defines a class method determine_support_resistance and two static methods check_price_at_support_resistance and find_next_support_resistance. The purpose of these methods is to identify support and resistance levels in financial time series data and provide signals based on the current market price.

## Here's a breakdown of each method:

### determine_support_resistance method:

Takes historical OHLCV (Open, High, Low, Close, Volume) data as input.
Calculates the Hull Moving Average of the closing prices.
Identifies turning points (maxima and minima) in the moving average.
Filters these turning points based on volume and distance criteria.
Applies KMeans clustering to group filtered turning points into clusters.
Merges levels within each cluster by taking the average position and volume.
Returns the merged support and resistance levels, along with the original data.

### check_price_at_support_resistance method:

Checks if the current price is approaching or very close to the next support or resistance levels.
Returns a signal ('Buy', 'Sell', or 'No Signal') based on the comparison.


### find_next_support_resistance method:

Takes the merged support and resistance levels along with the current market price.
Sorts the levels in ascending order.
Finds the next support and resistance levels based on the current price.
Handles cases where the next support and resistance are the same, finding the next distinct levels in both directions.
Note: The Trend class is used for representing the signal types ('Bullish', 'Bearish', 'Neutral'), but its definition is not provided in the code snippet.

To use these methods, you would typically create an instance of the class and call determine_support_resistance with historical data. Then, you can use the other static methods to check the current price against the identified support and resistance levels.

![2024-01-13 12_24_28-Figure 2](https://github.com/slegaitis/trading_support_resistance_python/assets/6602657/0dc9822d-eae9-41b5-951b-b09802ef5e65)
![2024-01-13 10_39_11-indicators py - app - Visual Studio Code](https://github.com/slegaitis/trading_support_resistance_python/assets/6602657/08a6e101-8618-4f9f-b3eb-7048ace52ca8)
![2024-01-13 12_33_30-Figure 2](https://github.com/slegaitis/trading_support_resistance_python/assets/6602657/1dd9d314-3a67-45d6-88d3-a97cee2d0fe3)
