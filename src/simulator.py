import pandas as pd

def predict_sales(model, input_data, columns):
    
    # convert Series → DataFrame
    input_df = input_data.to_frame().T

    # ensure same column order as training
    input_df = input_df[columns]

    # prediction
    prediction = model.predict(input_df)[0]

    return prediction