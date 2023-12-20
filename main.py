import subprocess
pythonPath = "C:/AI PRoject/Code Completion/venv/Scripts/python.exe"

# subprocess.run([pythonPath, "helper/tokenization.py"], check=True)
#
# subprocess.run([pythonPath, "helper/prepare_data.py"], check=True)
#
# subprocess.run([pythonPath, "model/ModelTraining/Single/train_RNN.py"], check=True)
# subprocess.run([pythonPath, "model/ModelTraining/Single/train_GRU.py"], check=True)
# subprocess.run([pythonPath, "model/ModelTraining/Single/train_LSTM.py"], check=True)
subprocess.run([pythonPath, "model/ModelTraining/Hybrid/RNN_GRU.py"], check=True)
subprocess.run([pythonPath, "model/ModelTraining/Hybrid/LSTM_GRU.py"], check=True)

subprocess.run([pythonPath, "model/evaluate/evaluate_model.py"], check=True)
subprocess.run([pythonPath, "model/evaluate/visualize_element_counts.py"], check=True)

# subprocess.run([pythonPath, "generateCode/Single/genLSTM.py"], check=True)
# subprocess.run([pythonPath, "generateCode/Single/genRNN.py"], check=True)
# subprocess.run([pythonPath, "generateCode/Single/genGru.py"], check=True)
# subprocess.run([pythonPath, "generateCode/Hybrid/RNN_GRU.py"], check=True)
# subprocess.run([pythonPath, "generateCode/Hybrid/LSTM_GRU.py"], check=True)
