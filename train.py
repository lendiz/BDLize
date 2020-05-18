from __future__ import absolute_import, division, print_function, unicode_literals
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
import loader
import tables_to_leave
from sklearn.externals.joblib import dump, load



KNOWN_PATCH = 'C:\\Users\\Tom\\PycharmProjects\\wiseNeuro\\data\\known.csv'
SKY_PATCH = 'C:\\Users\\Tom\\PycharmProjects\\wiseNeuro\\data\\sky_train.csv'
TEST_PATH = ''


known_df = loader.load_file(KNOWN_PATCH, sep=';')
known_df['isdwarf'] = 1
known_df = loader.drop_tables(known_df, tables_to_leave.TABLE_LIST)
print(known_df.head())
sky_df = loader.load_file(SKY_PATCH, sep=',')
sky_df['isdwarf'] = 0
sky_df = loader.drop_tables(sky_df, tables_to_leave.TABLE_LIST)
print(sky_df.head())
df = pd.concat([known_df, sky_df], sort=False)
df = df.fillna(0)
df_len = len(df.columns)
print(df_len)
target = df.pop('isdwarf')
scaler = StandardScaler()
scaler.fit(df)
dump(scaler, 'std_scaler.bin', compress=True)
df = scaler.transform(df)
dataset = tf.data.Dataset.from_tensor_slices((df, target.values))
train_dataset = dataset.shuffle(len(df)).batch(1)


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(df_len, activation='sigmoid'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(200, activation='tanh'),
        tf.keras.layers.Dense(400, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    optim = tf.keras.optimizers.SGD(learning_rate=0.01)
    loss_fn = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer=optim,
                  loss=loss_fn,
                  metrics=['accuracy'])
    return model

model = get_compiled_model()
model.fit(train_dataset, epochs=1)

loader.save_model(model)

