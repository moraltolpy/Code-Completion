import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

def generate_code(model, tokenizer, seed_text, max_length):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    generated_code = seed_text

    for _ in range(max_length):
        token_list_padded = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted = np.argmax(model.predict(token_list_padded), axis=-1)
        predicted_index = predicted[0]

        if predicted_index in tokenizer.index_word:
            output_word = tokenizer.index_word[predicted_index]
            token_list.append(predicted_index)
            generated_code += " " + output_word

            if output_word == '<END>':
                break
        else:
            print(f"Token index {predicted_index} not found in tokenizer's index_word.")
            break

    return generated_code

# Load the GRU model
model = load_model('result_files/code_generation_lstm_model.keras')

# Load the tokenizer
with open('result_files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Generate code
seed_text = "def my_function(arg):"
max_sequence_len = 47  # This should match the sequence length used during model training
generated_code = generate_code(model, tokenizer, seed_text, max_sequence_len - 1)  # max_length parameter adjusted
print(generated_code)
