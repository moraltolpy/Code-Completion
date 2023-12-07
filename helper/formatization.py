from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

with open('tokenized_code.txt', 'r', encoding='utf-8') as file:
    tokenized_lines = file.readlines()

filtered_tokens = [line for line in tokenized_lines if not line.startswith('COMMENT') and not line.strip() == 'NL:']

sequences = [line.strip().split(':')[1] for line in filtered_tokens if line.strip()]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences_numeric = tokenizer.texts_to_sequences(sequences)

max_sequence_len = max([len(seq) for seq in sequences_numeric])

sequences_padded = pad_sequences(sequences_numeric, maxlen=max_sequence_len, padding='post')
