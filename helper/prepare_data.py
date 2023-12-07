from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import pickle

with open('result_files/tokenized_code.txt', 'r', encoding='utf-8') as file:
    tokenized_lines = file.readlines()

filtered_tokens = [line for line in tokenized_lines if not line.startswith('COMMENT') and not line.strip() == 'NL:']

sequences = [line.strip().split(':')[1] for line in filtered_tokens if ':' in line and line.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences_numeric = tokenizer.texts_to_sequences(sequences)

max_sequence_len = max([len(seq) for seq in sequences_numeric])

sequences_padded = pad_sequences(sequences_numeric, maxlen=max_sequence_len, padding='post')

X = np.array(sequences_padded[:, :-1])
y = np.array(sequences_padded[:, -1])

vocab_size = len(tokenizer.word_index) + 1
y = to_categorical(y, num_classes=vocab_size)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

with open('result_files/prepared_data.pkl', 'wb') as file:
    pickle.dump((X_train, X_test, y_train, y_test, max_sequence_len, vocab_size, tokenizer), file)
