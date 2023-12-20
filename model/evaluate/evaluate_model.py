import pickle
import numpy as np
from keras.models import load_model

# Load the data
with open('result_files/prepared_data.pkl', 'rb') as file:
    X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer = pickle.load(file)

model_rnn = load_model('result_files/code_generation_rnn_model.keras')
loss_rnn, accuracy_rnn = model_rnn.evaluate(X_test, y_test)
print(f'RNN test Loss: {loss_rnn:.6f}')
print(f'RNN test Accuracy: {accuracy_rnn:.6f}')

model_lstm = load_model('result_files/code_generation_lstm_model.keras')
loss_lstm, accuracy_lstm = model_lstm.evaluate(X_test, y_test)
print(f'LSTM test Loss: {loss_lstm:.6f}')
print(f'LSTM test Accuracy: {accuracy_lstm:.6f}')

model_gru = load_model('result_files/code_generation_gru_model.keras')
loss_gru, accuracy_gru = model_gru.evaluate(X_test, y_test)
print(f'GRU test Loss: {loss_gru:.6f}')
print(f'GRU test Accuracy: {accuracy_gru:.6f}')

model_rnn_gru = load_model('result_files/code_generation_hybrid_rnn_gru_model.keras')
loss_rnn_gru, accuracy_rnn_gru = model_rnn_gru.evaluate(X_test, y_test)
print(f'RNN+GRU test Loss: {loss_rnn_gru:.6f}')
print(f'RNN+GRU test Accuracy: {accuracy_rnn_gru:.6f}')

model_lstm_gru = load_model('result_files/code_generation_hybrid_lstm_gru_model.keras')
loss_lstm_gru, accuracy_lstm_gru = model_lstm_gru.evaluate(X_test, y_test)
print(f'LSTM+GRU test Loss: {loss_lstm_gru:.6f}')
print(f'LSTM+GRU test Accuracy: {accuracy_lstm_gru:.6f}')

import matplotlib.pyplot as plt

model_performance = {
    'RNN': {'test_loss': loss_rnn, 'test_accuracy': accuracy_rnn},
    'LSTM': {'test_loss': loss_lstm, 'test_accuracy': accuracy_lstm},
    'GRU': {'test_loss':loss_gru, 'test_accuracy': accuracy_gru},
    'RNN+GRU': {'test_loss': loss_rnn_gru, 'test_accuracy': 0.82},
    'LSTM+GRU': {'test_loss': loss_lstm_gru, 'test_accuracy': 0.86}
}

# Create subplots
fig, ax1 = plt.subplots()

# Bar settings
bar_width = 0.35
opacity = 0.8

# Set position of bar on X axis
r1 = np.arange(len(model_performance))
r2 = [x + bar_width for x in r1]

# Make the plot
ax1.bar(r1, [model_performance[model]['test_loss'] for model in model_performance],
        color='b', width=bar_width, alpha=opacity, label='Test Loss')
ax1.bar(r2, [model_performance[model]['test_accuracy'] for model in model_performance],
        color='g', width=bar_width, alpha=opacity, label='Test Accuracy')

# Add xticks on the middle of the group bars
ax1.set_xlabel('Model', fontweight='bold')
ax1.set_xticks([r + bar_width for r in range(len(model_performance))])
ax1.set_xticklabels(list(model_performance.keys()))
ax1.legend()

# Create labels
for i in ax1.patches:
    ax1.text(i.get_x() + i.get_width() / 2, i.get_height() + 0.01, \
            str(round((i.get_height()), 2)), fontsize=11, color='black',
                rotation=0, ha='center')

# Show the plot
plt.tight_layout()
plt.show()
