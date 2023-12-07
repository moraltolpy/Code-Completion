import pickle

from keras.models import load_model
import numpy as np
from keras.preprocessing.sequence import pad_sequences

def generate_code(model, tokenizer, seed_text, max_length=50):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    generated_code = seed_text

    for _ in range(max_length):
        token_list_padded = pad_sequences([token_list], maxlen=max_length, padding='pre')
        predicted = np.argmax(model.predict(token_list_padded), axis=-1)
        output_word = tokenizer.index_word[predicted[0]]
        token_list.append(predicted[0])
        generated_code += " " + output_word

        if output_word == '<END>':
            break

    return generated_code

model = load_model('result_files/code_generation_rnn_model.keras')

with open('result_files/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

seed_text = "def my_function(arg):"
generated_code = generate_code(model, tokenizer, seed_text)
rint(generated_code)
