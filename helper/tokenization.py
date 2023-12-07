import os
import ast
import pickle
import tokenize

from keras.src.preprocessing.text import Tokenizer


def get_ast_tokens(node, tokens=[]):
    for field_name in node._fields:
        field = getattr(node, field_name)
        if isinstance(field, ast.AST):
            get_ast_tokens(field, tokens)
        elif isinstance(field, list):
            for item in field:
                if isinstance(item, ast.AST):
                    get_ast_tokens(item, tokens)
        else:
            token_type = type(node).__name__
            token_value = getattr(node, field_name, None)
            if token_value is not None and isinstance(token_value, (int, float, str)):
                tokens.append(f'{token_type}:{token_value}')
    return tokens

def tokenize_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    ast_tree = ast.parse(source_code)
    tokens = get_ast_tokens(ast_tree)
    return tokens

def deprecated_tokenize_file(file_path):
    with open(file_path, 'rb') as file:
        tokens = tokenize.tokenize(file.readline)
        return [f'{tokenize.tok_name[token.type]}:{token.string}' for token in tokens if token.type != tokenize.ENCODING]

def process_directory(directory):
    all_tokens = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                tokens = tokenize_file(file_path)
                all_tokens.extend(tokens)
    return all_tokens

def save_tokens(tokens, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for token in tokens:
            file.write(token + '\n')

repository_path = 'C:/AI Project/Git repositories/django'
tokens = process_directory(repository_path)
save_tokens(tokens, 'result_files/tokenized_code.txt')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)

# Serialize the tokenizer to a file
with open('result_files/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)