import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, GRU, Dense

with open('result_files/prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer = pickle.load(file)

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length=max_sequence_len - 1))
model.add(SimpleRNN(128, return_sequences=True))
model.add(GRU(128, return_sequences=False))
model.add(Dense(vocab_size, activation='softmax'))  # Output layer

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.2)

model.save('result_files/code_generation_hybrid_rnn_gru_model.keras')

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('result_files/RNN_GRU_T_and_V_Loss.png')
plt.show()
