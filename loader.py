import pandas as pd
from tensorflow.keras.models import model_from_json

def load_file(file_path, sep=','):
    file_format = file_path.split(".")[-1]
    if file_format == 'csv':
        dataframe = pd.read_csv(file_path, sep=sep, low_memory=False)
    elif file_format in ('xls', 'xlsx'):
        dataframe = pd.read_excel(file_path, low_memory=False)
    else:
        dataframe = False
    return dataframe


def file_write(dataframe: pd.DataFrame, file_path, append, csv_sep=';', go=False):
    if go:
        dataframe.to_csv(file_path)
        return True
    else:
        old_file_dataframe = append
        headers = list(dataframe)
        for header in headers:
            old_file_dataframe[header] = dataframe[header]
        file_write(old_file_dataframe, file_path, append, go=True)


def drop_tables(dataframe: pd.DataFrame, leave_tables: list):
    headers = list(dataframe)
    for header in headers:
        if header not in leave_tables:
            del dataframe[header]
    return dataframe


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model