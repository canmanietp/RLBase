import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

data = pd.read_csv("Q_values_<TaxiEnv instance>.csv")

# data = data.astype({'state': 'int32', 'action': 'int32'})
data = data.sort_values(by=['action', 'subopt_act_val', 'opt_act_val'], ascending=False)

# print(data.head())

states = []
groupings = []
pairs = []

prev_row = None

for index, row in data.iterrows():
    if prev_row is None:
        prev_row = row
        groupings.append([int(row['state'])])
        states.append(int(row['state']))
    else:
        if row['action'] == prev_row['action'] and row['opt_act_val'] > prev_row['subopt_act_val']:
            for ps in groupings[-1]:
                pairs.append([ps, int(row['state'])])
            groupings[-1].append(int(row['state']))
        else:
            groupings.append([int(row['state'])])
            states.append(int(row['state']))
        prev_row = row

num_random_samples = len(pairs) * 3
X = []
y = []
for s in range(num_random_samples):
    pair = np.random.choice(states, 2, replace=False)
    _x = list(pair)
    # print(pair, [f for f in groupings if pair[0] in f][0])
    _y = int(pair[1] in [f for f in groupings if pair[0] in f][0])
    if _y == 1:
        print("YES", _x, _y)
    X.append(_x)
    y.append(_y)

for p in pairs:
    X.append(p)
    y.append(1)

X = np.array(X)
y = np.array(y)

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=20, batch_size=50)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


print(model.predict(np.array([[84, 409], [84, 494], [338, 497], [338, 203], [318, 379], [399, 299]])))