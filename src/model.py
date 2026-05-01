import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def train_model(data):

    # 1. Split features and target
    X = data.drop(['units_sold', 'date', 'demand_forecast'], axis=1)
    y = data['units_sold']

    # 2. Keep only numeric columns
    X = X.select_dtypes(include=['int64', 'float64'])

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)

    # 5. Predictions
    y_pred = model.predict(X_test)

    # 6. Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, X_test, y_test, mae, r2


#  SAVE MODEL
def save_model(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)


# LOAD MODEL
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


#  TEST BLOCK
if __name__ == "__main__":

    from preprocess import load_and_clean_data

    # load data
    data = load_and_clean_data("data/sales.csv")

    # train model
    model, X_test, y_test, mae, r2 = train_model(data)

    print("MAE:", mae)
    print("R2:", r2)

    # save model
    save_model(model)

    print("Model saved successfully!")