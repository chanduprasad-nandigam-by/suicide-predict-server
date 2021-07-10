import pandas as pd
import keras
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn import preprocessing


def replace_text(what, to):
    df.replace(what, to, inplace=True)


def encode_outputcheck(df_copy, name):
    le = preprocessing.LabelEncoder()
    df_copy[name] = le.fit_transform(df_copy[name])
    return le.classes_


def encode_inputs(df_copy, name):
    dummies = pd.get_dummies(df_copy[name])
    for x in dummies.columns:
        dummy_name = "{}-{}".format(name, x)
        df_copy[dummy_name] = dummies[x]
    df_copy.drop(name, axis=1, inplace=True)


df = pd.read_csv("foreveralone.csv")
#print(df.isna().any())
df.dropna(inplace=True)
#print(df.isna().any())

df['job_title'] = df.job_title.str.strip()
replace_text('none', 'None')
replace_text("N/a", 'None')
replace_text('na', 'None')
replace_text('-', 'None')
replace_text('.', 'None')
replace_text('*', 'None')
replace_text('ggg', 'None')

df_copy = df

df_copy = df_copy.drop(columns=['time', 'what_help_from_others'])
df_copy = df_copy.drop(columns=['improve_yourself_how'])
#df_for_sampling = df_copy.copy(deep=True)
inputs = ["gender", "age", "sexuallity", "income", "race", "bodyweight", "virgin", "prostitution_legal",
          "pay_for_sex", "social_fear", "depressed", "employment", "job_title", "edu_level"]
output = ['attempt_suicide']

for i in output:
  encode_outputcheck(df_copy, i)
for j in inputs:
    print(j)
    encode_inputs(df_copy, j)

nnoutputs = df_copy['attempt_suicide'].values
nninputs = df_copy.drop(columns=['attempt_suicide'])
nninputs = nninputs.values
#nnoutputs=nnoutputs[:2]
print(nninputs.shape, nnoutputs.shape)
x_train, x_test, y_train, y_test = train_test_split(
    nninputs, nnoutputs, test_size=0.2, random_state=42)
checkpointer2 = ModelCheckpoint(
    filepath="./checkpoint.h5", verbose=0, save_best_only=True)
num_classes = 2
y_train1 = keras.utils.to_categorical(y_train, num_classes)

y_test1 = keras.utils.to_categorical(y_test, num_classes)
print(x_train.shape)
print(y_test1.shape)

for i in range(5):
    model = Sequential()
    model.add(Dense(25, input_dim=x_train.shape[1], activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(y_train1.shape[1], activation='softmax'))
    monitor = EarlyStopping(
        monitor='val_loss', min_delta=1e-5, patience=6, verbose=2, mode='auto')
    model.compile(loss="categorical_crossentropy", optimizer='adam')
    model.fit(x_train, y_train1, callbacks=[
              monitor, checkpointer2], validation_data=(x_test, y_test1), epochs=10)

model.load_weights('checkpoint.h5')
y_pred_c = model.predict(x_test)
y_pred_c = np.argmax(y_pred_c, axis=1)
y_test_c = np.argmax(y_test1, axis=1)
score = metrics.accuracy_score(y_test_c, y_pred_c)
print("Accuracy score: {}".format(score*100))
