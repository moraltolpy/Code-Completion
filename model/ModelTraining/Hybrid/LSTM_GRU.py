import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GRU, Dense

# Load the data
with open('result_files/prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer = pickle.load(file)

# Define the model
model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_len - 1))
model.add(LSTM(128, return_sequences=True))
model.add(GRU(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))  # Output layer

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

# Save the model
model.save('result_files/code_generation_hybrid_lstm_gru_model.keras')
