from pickle import dump, load
import xgboost as xgb
from sklearn.metrics import f1_score
import pandas as pd

def split_data(df: pd.DataFrame):
    y = df['Churn Value']
    X = df.drop(['Churn Value'], axis = 1)
    return X, y

def open_data(path="data/clean_churn.csv"):
    df = pd.read_csv(path)
    return df


def fit_and_save_model(X_df, y_df, path="model_weights_2.mw"):
    model =xgb.XGBClassifier(random_state = 42, subsample= 1.0, min_child_weight= 5, max_depth= 4, gamma = 1.5,
colsample_bytree= 1.0)
    model.fit(X_df, y_df)

    test_prediction = model.predict(X_df)
    f1 = f1_score(test_prediction, y_df)
    print(f"Model's f1 score is {f1}")

    with open(path, "wb") as file:
        dump(model, file)

    print(f"Model was saved to {path}")

def load_model_and_predict(df, path="model_weights_2.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
    # prediction = np.squeeze(prediction)

    prediction_probas = model.predict_proba(df)[0]
    # prediction_proba = np.squeeze(prediction_proba)

    encode_prediction_proba = {
        0: "Not churn with a probability:",
        1: "Churn with a probability:"
    }

    encode_prediction = {
        0: "The customer does not churn!",
        1: "The customer wants to churn ..."
    }


    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_probas[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    # Если не сработает удали эту строчку внизу:
    prediction_df["Not churn with a probability:"] = prediction_df["Not churn with a probability:"].astype(float).map("{:.1%}".format)
    prediction_df["Churn with a probability:"] = prediction_df["Churn with a probability:"].astype(float).map("{:.1%}".format)

    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_df, y_df = split_data(df)
    fit_and_save_model(X_df, y_df)
