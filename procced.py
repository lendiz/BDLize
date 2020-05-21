import loader
import tables_to_leave
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import load, dump



LOAD_PATH = 'C:\\Users\\Tom\\PycharmProjects\\wiseNeuro\\data\\pred.csv'
SAVE_PATH = 'C:\\Users\\Tom\\PycharmProjects\\wiseNeuro\\data\\prediction.csv'

df = loader.load_file(LOAD_PATH, ',')
df = loader.drop_tables(df, tables_to_leave.TABLE_LIST)
df = df.fillna(0)
df = df.to_numpy(dtype=float)
scaler=load('std_scaler.bin')
#scaler = StandardScaler()
#scaler.fit(df)
df = scaler.transform(df)
model = loader.load_model()
prediction = model.predict(df)
newframe = pd.DataFrame()
newframe['result'] = prediction.flatten().astype(float)

def drop_result(df):
    df_head = list(df)
    for head in df_head:
        if head not in ('RAJ2000', 'DEJ2000'):
            del df[head]
    return df

df = loader.load_file(LOAD_PATH, ',')
df = drop_result(df)

loader.file_write(newframe, SAVE_PATH, df, csv_sep=',')
