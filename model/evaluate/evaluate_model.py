import pickle
from keras.models import load_model

with open('result_files/prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer = pickle.load(file)

model = load_model('result_files/code_generation_lstm_model.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'LSTM test Loss: {loss:.2f}')
print(f'LSTM test Accuracy: {accuracy:.2f}')

model = load_model('result_files/code_generation_rnn_model.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'RNN test Loss: {loss:.2f}')
print(f'RNN test Accuracy: {accuracy:.2f}')

model = load_model('result_files/code_generation_gru_model.keras')

loss, accuracy = model.evaluate(X_test, y_test)
print(f'GRU test Loss: {loss:.2f}')
print(f'GRU test Accuracy: {accuracy:.2f}')
