import subprocess
pythonPath = "C:/AI PRoject/Code Completion/venv/Scripts/python.exe"

subprocess.run([pythonPath, "helper/tokenization.py"], check=True)

subprocess.run([pythonPath, "helper/prepare_data.py"], check=True)

# subprocess.run([pythonPath, "model/train_LSTM.py"], check=True)
# subprocess.run([pythonPath, "model/train_RNN.py"], check=True)
# subprocess.run([pythonPath, "model/train_GRU.py"], check=True)

subprocess.run([pythonPath, "model/evaluate/evaluate_model.py"], check=True)
subprocess.run([pythonPath, "model/evaluate/visualize_element_counts.py"], check=True)

subprocess.run([pythonPath, "generateCode/genLSTM.py"], check=True)
subprocess.run([pythonPath, "generateCode/genRNN.py"], check=True)
subprocess.run([pythonPath, "generateCode/genGru.py"], check=True)
