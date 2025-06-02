import time
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import joblib  # For saving and loading models

class AdvancedDynamicPricing:
    def __init__(self, historical_data_path="ride_data_advanced.csv", demand_model_path="demand_model.joblib"):
        self.base_fare = 50
        self.price_per_km = 10
        self.price_per_minute = 2
        self.demand_model = self.load_model(demand_model_path)
        self.historical_data = self.load_historical_data(historical_data_path)
        if self.demand_model is None and self.historical_data is not None:
            self.train_demand_model()
            self.save_model(self.demand_model, demand_model_path)

        self.price_elasticity_model = self.train_price_elasticity_model() # Simplified for now

    def load_historical_data(self, path):
        try:
            df = pd.read_csv(path)
            required_cols = ['timestamp', 'latitude', 'longitude', 'drivers_available', 'rider_requests',
                             'base_price', 'distance_km', 'duration_minutes', 'surge_multiplier',
                             'weather_condition', 'event_indicator', 'competitor_price']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Historical data missing required column: {col}")
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['dayofweek'] = df['timestamp'].dt.dayofweek
            df['demand_supply_ratio'] = df['rider_requests'] / (df['drivers_available'] + 1e-6)
            return df
        except FileNotFoundError:
            print(f"Warning: Historical data file not found at {path}. Proceeding without pre-trained model.")
            return None

    def train_demand_model(self, model_type='random_forest'):
        if self.historical_data is not None:
            features = ['latitude', 'longitude', 'drivers_available', 'hour', 'dayofweek',
                        'base_price', 'distance_km', 'duration_minutes', 'weather_condition',
                        'event_indicator', 'competitor_price']
            target = 'rider_requests'
            X = pd.get_dummies(self.historical_data[features], drop_first=True) # Handle categorical features
            y = self.historical_data[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == 'linear':
                model = LinearRegression()
            elif model_type == 'gradient_boosting':
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
            elif model_type == 'neural_network':
                model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=42, max_iter=300)
            else: # Default to Random Forest
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f"Demand model ({model_type}) trained. Test MSE: {mse:.2f}")
            self.demand_model = model
        else:
            print("No historical data available to train the demand model.")

    def predict_demand(self, current_state):
        if self.demand_model:
            features = ['latitude', 'longitude', 'drivers_available', 'hour', 'dayofweek',
                        'base_price', 'distance_km', 'duration_minutes', 'weather_condition',
                        'event_indicator', 'competitor_price']
            input_df = pd.DataFrame([current_state])[features]
            input_df = pd.get_dummies(input_df, drop_first=True, columns=['weather_condition'])
            train_cols = self.get_model_feature_names(self.demand_model)
            for col in train_cols:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[train_cols]

            predicted_demand = self.demand_model.predict(input_df)
            if isinstance(predicted_demand, np.ndarray):
                return max(0, int(round(predicted_demand[0])))
            else:
                return max(0, int(round(predicted_demand)))
        else:
            ratio = current_state['rider_requests'] / (current_state['drivers_available'] + 1e-6)
            return current_state['rider_requests'] + int(ratio * 2)

    def train_price_elasticity_model(self):
        # Simplified: Assume a fixed elasticity for demonstration
        # In a real system, you'd train a model based on historical price changes and demand response
        return -0.5 # Example: A 1% increase in price leads to a 0.5% decrease in demand

    def predict_demand_with_elasticity(self, base_demand, price_change_percentage):
        elasticity = self.price_elasticity_model
        demand_change_percentage = elasticity * price_change_percentage
        new_demand = base_demand * (1 + demand_change_percentage / 100)
        #print("i am new demand",new_demand)
        return max(0, int(round(new_demand[0])))

    def revenue_function(self, surge, current_state):
        hypothetical_state = current_state.copy()
        current_price = self.base_fare * hypothetical_state.get('surge_multiplier', 1.0)
        hypothetical_price = self.base_fare * surge
        price_change_percentage = ((hypothetical_price - current_price) / (current_price + 1e-6)) * 100
        hypothetical_state['base_price'] = hypothetical_price
        predicted_base_demand = self.predict_demand(hypothetical_state)
        predicted_demand = self.predict_demand_with_elasticity(predicted_base_demand, price_change_percentage)
        estimated_fare_per_ride = (hypothetical_price +
                                   current_state['distance_km'] * self.price_per_km +
                                   current_state['duration_minutes'] * self.price_per_minute)
        return - (predicted_demand * estimated_fare_per_ride)

    def calculate_optimal_surge(self, current_state, min_surge=1.0, max_surge=5.0):
        initial_surge = current_state.get('surge_multiplier', 1.0)
        bounds = [(min_surge, max_surge)]
        # Pass current_state as an argument to revenue_function
        result = minimize(self.revenue_function, [initial_surge], args=(current_state,), bounds=bounds, method='L-BFGS-B')
        return max(min_surge, min(max_surge, result.x[0])) if result.success else initial_surge

    def calculate_fare(self, distance_km, duration_minutes, current_state):
        surge = self.calculate_optimal_surge(current_state)
        base_price = self.base_fare * surge
        distance_cost = distance_km * self.price_per_km
        duration_cost = duration_minutes * self.price_per_minute
        total_fare = base_price + distance_cost + duration_minutes
        return max(total_fare, self.base_fare)

    def get_surge_multiplier(self, current_state):
        return self.calculate_optimal_surge(current_state)

    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
        print(f"Demand model saved to {filepath}")

    def load_model(self, filepath):
        try:
            model = joblib.load(filepath)
            print(f"Demand model loaded from {filepath}")
            return model
        except FileNotFoundError:
            print("No saved demand model found.")
            return None

    def get_model_feature_names(self, model):
        if hasattr(model, 'feature_names_in_'):
            return list(model.feature_names_in_)
        elif hasattr(model, 'feature_name_'): # For older scikit-learn versions
            return list(model.feature_name_)
        elif hasattr(model, 'n_features_in_'): # Fallback for basic models
            # We have to infer the order based on how we trained
            return ['latitude', 'longitude', 'drivers_available', 'hour', 'dayofweek',
                    'base_price', 'distance_km', 'duration_minutes', 'weather_condition_Rainy',
                    'weather_condition_Sunny', 'event_indicator_1', 'competitor_price'] # Example order
        return []

# Simulation with more features
if __name__ == "__main__":
    # Create a more comprehensive dummy historical data CSV
    dates = pd.to_datetime(pd.date_range(start='2025-06-01', end='2025-06-05', freq='h'))
    num_samples = len(dates)
    data_advanced = {
        'timestamp': dates,
        'latitude': np.random.uniform(12.90, 13.10, num_samples), # Chennai latitudes
        'longitude': np.random.uniform(80.10, 80.30, num_samples), # Chennai longitudes
        'drivers_available': np.random.randint(5, 30, num_samples),
        'rider_requests': np.random.randint(2, 40, num_samples),
        'base_price': np.random.uniform(40, 70, num_samples),
        'distance_km': np.random.uniform(1, 15, num_samples),
        'duration_minutes': np.random.randint(5, 45, num_samples),
        'surge_multiplier': np.random.uniform(1.0, 2.5, num_samples),
        'weather_condition': np.random.choice(['Sunny', 'Rainy', 'Cloudy'], num_samples),
        'event_indicator': np.random.randint(0, 2, num_samples), # 1 if event, 0 otherwise
        'competitor_price': np.random.uniform(35, 75, num_samples)
    }
    df_advanced = pd.DataFrame(data_advanced)
    df_advanced.to_csv("ride_data_advanced.csv", index=False)

    pricing_model_advanced = AdvancedDynamicPricing("ride_data_advanced.csv", "demand_model_advanced.joblib")

    # Simulate real-time scenario with more features
    current_conditions_area1_adv = {
        'latitude': 12.97,
        'longitude': 80.22,
        'drivers_available': 12,
        'rider_requests': 25,
        'distance_km': 5,
        'duration_minutes': 15,
        'timestamp': pd.Timestamp('2025-06-02 18:30:00'),
        'weather_condition': 'Rainy',
        'event_indicator': 1,
        'competitor_price': 65,
        'hour': pd.Timestamp('2025-06-02 18:30:00').hour,
        'dayofweek': pd.Timestamp('2025-06-02 18:30:00').dayofweek
    }

    current_conditions_area2_adv = {
        'latitude': 13.05,
        'longitude': 80.15,
        'drivers_available': 20,
        'rider_requests': 8,
        'distance_km': 8,
        'duration_minutes': 25,
        'timestamp': pd.Timestamp('2025-06-02 18:45:00'),
        'weather_condition': 'Sunny',
        'event_indicator': 0,
        'competitor_price': 50,
        'hour': pd.Timestamp('2025-06-02 18:45:00').hour,
        'dayofweek': pd.Timestamp('2025-06-02 18:45:00').dayofweek
    }

    surge_area1_adv = pricing_model_advanced.get_surge_multiplier(current_conditions_area1_adv)
    fare_area1_adv = pricing_model_advanced.calculate_fare(current_conditions_area1_adv['distance_km'], current_conditions_area1_adv['duration_minutes'], current_conditions_area1_adv)
    print(f"\nAdvanced Model - Area 1 - Predicted Surge Multiplier: {surge_area1_adv:.2f}, Estimated Fare: {fare_area1_adv:.2f} INR")

    surge_area2_adv = pricing_model_advanced.get_surge_multiplier(current_conditions_area2_adv)
    fare_area2_adv = pricing_model_advanced.calculate_fare(current_conditions_area2_adv['distance_km'], current_conditions_area2_adv['duration_minutes'], current_conditions_area2_adv)
    print(f"Advanced Model - Area 2 - Predicted Surge Multiplier: {surge_area2_adv:.2f}, Estimated Fare: {fare_area2_adv:.2f} INR")