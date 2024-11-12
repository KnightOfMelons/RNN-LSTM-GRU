Выполнить исследование и сравнительный анализ возможностей RNN,
LSTM и GRU.

Выполнить работу по варианту, соответствующему номеру с id
авиационного двигателя в наборе данных (у меня 6 вариант)

1. Сравнить полученные нейронные сети по Accuracy, Precision,
Recall, F1, Loss на train и test.
Выполнить несколько запусков программы с разными seed:

```
# Setting seed for reproducability
np.random.seed(1234)
```

Выбрать лучший вариант.

2. Выполнить исследования на примере фрагмента кода с заменой
LSTM на RNN и GRU.

```
# build the network
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]
model = Sequential()
model.add(LSTM(
 input_shape=(sequence_length, nb_features),
 units=100,
 return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(
 units=50,
 return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=nb_out, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
metrics=['accuracy'])
```

Изучить и описать назначение используемых методов и параметров.

3. Исследовать, как определяется число параметров Param в каждом
слое.
4. Изучить и описать назначение используемых методов и параметров.
5. Вывести графические зависимости для Loss и Accuracy на train и
val (на обучающей и валидационной подвыборках).
6. Оценить время разработки классификаторов с CPU.
7. Оценить время разработки классификаторов с GPU (в Google
Colab).