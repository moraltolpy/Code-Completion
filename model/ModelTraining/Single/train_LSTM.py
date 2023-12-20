import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

with open('result_files/prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer = pickle.load(file)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_len - 1))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

model.save('result_files/code_generation_lstm_model.keras')
