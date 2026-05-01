import pandas as pd

def load_and_clean_data(path):
    
    # 1. Load data
    data = pd.read_csv(path)

    # 2. Clean column names
    data.columns = data.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")

    # 3. Convert date column
    data['date'] = pd.to_datetime(data['date'])

    # 4. Handle missing values
    data['units_sold'] = data['units_sold'].fillna(data['units_sold'].mean())
    data['price'] = data['price'].fillna(data['price'].mean())

    # 5. Fix holiday/promotion column
    data['holiday_promotion'] = data['holiday_promotion'].apply(
        lambda x: 1 if str(x).lower() in ['yes', 'holiday', 'promotion'] else 0
    )

    # 6. Feature Engineering (date features)
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data['day_of_week'] = data['date'].dt.dayofweek

    # 7. Remove duplicates (safe step)
    data = data.drop_duplicates()

    # 8. Encoding categorical columns
    data = pd.get_dummies(data, columns=[
        'category',
        'region',
        'weather_condition',
        'seasonality'
    ], drop_first=True)

    return data
if __name__ == "__main__":
    data = load_and_clean_data("data/sales.csv")
    print(data.head())