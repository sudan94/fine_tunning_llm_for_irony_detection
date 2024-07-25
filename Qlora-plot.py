import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data from CSV files for binary classification
file_model1_hp1 = "training-evaluation-results/binary/evaluation_distillbert_set1_qlora.csv"
file_model1_hp2 = "training-evaluation-results/binary/evaluation_distillbert_set2_qlora.csv"
file_model2_hp1 = "training-evaluation-results/binary/evaluation_flant5_set1_qlora.csv"
file_model2_hp2 = "training-evaluation-results/binary/evaluation_flant5_set2_qlora.csv"
file_model3_hp1 = "training-evaluation-results/binary/evaluation_bert_set1_qlora.csv"
file_model3_hp2 = "training-evaluation-results/binary/evaluation_bert_set2_qlora.csv"

# uncomment the following lines for multi-class classification and comment the above lines

# file_model1_hp1 = "training-evaluation-results/multi-class/evaluation_distillbert_set1_qlora.csv"
# file_model1_hp2 = "training-evaluation-results/multi-class/evaluation_distillbert_set2_qlora.csv"
# file_model2_hp1 = "training-evaluation-results/multi-class/evaluation_flant5_set1_qlora.csv"
# file_model2_hp2 = "training-evaluation-results/multi-class/evaluation_flant5_set2_qlora.csv"
# file_model3_hp1 = "training-evaluation-results/multi-class/evaluation_bert_set1_qlora.csv"
# file_model3_hp2 = "training-evaluation-results/multi-class/evaluation_bert_set2_qlora.csv"


df_model1_hp1 = pd.read_csv(file_model1_hp1)
df_model1_hp2 = pd.read_csv(file_model1_hp2)
df_model2_hp1 = pd.read_csv(file_model2_hp1)
df_model2_hp2 = pd.read_csv(file_model2_hp2)
df_model3_hp1 = pd.read_csv(file_model3_hp1)
df_model3_hp2 = pd.read_csv(file_model3_hp2)

model1 = 'DistilBERT'
model2 = 'FlanT5'
model3 = 'BERT'

# Plotting training and validation losses for Hyperparameter Set 1
plt.figure(figsize=(12, 6))

plt.plot(df_model1_hp1['epoch'], df_model1_hp1['training_loss'], label=model1+' - Train Loss', linestyle='-', color='blue', marker='o')
plt.plot(df_model1_hp1['epoch'], df_model1_hp1['validation_loss'], label=model1+' - Validation Loss', linestyle='--', color='blue', marker='o')

plt.plot(df_model2_hp1['epoch'], df_model2_hp1['training_loss'], label=model2+' - Train Loss', linestyle='-', color='green', marker='o')
plt.plot(df_model2_hp1['epoch'], df_model2_hp1['validation_loss'], label=model2+' - Validation Loss', linestyle='--', color='green', marker='o')

plt.plot(df_model3_hp1['epoch'], df_model3_hp1['training_loss'], label=model3+' - Train Loss', linestyle='-', color='orange', marker='o')
plt.plot(df_model3_hp1['epoch'], df_model3_hp1['validation_loss'], label=model3+' - Validation Loss', linestyle='--', color='orange', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Hyperparameter Set 1 - QLoRA')
plt.legend()
plt.grid(True)
plt.show()

# Plotting training and validation losses for Hyperparameter Set 2
plt.figure(figsize=(12, 6))

plt.plot(df_model1_hp2['epoch'], df_model1_hp2['training_loss'], label=model1+' - Train Loss', linestyle='-', color='blue', marker='o')
plt.plot(df_model1_hp2['epoch'], df_model1_hp2['validation_loss'], label=model1+' - Validation Loss', linestyle='--', color='blue', marker='o')

plt.plot(df_model2_hp2['epoch'], df_model2_hp2['training_loss'], label=model2+' - Train Loss', linestyle='-', color='green', marker='o')
plt.plot(df_model2_hp2['epoch'], df_model2_hp2['validation_loss'], label=model2+' - Validation Loss', linestyle='--', color='green', marker='o')

plt.plot(df_model3_hp2['epoch'], df_model3_hp2['training_loss'], label=model3+' - Train Loss', linestyle='-', color='orange', marker='o')
plt.plot(df_model3_hp2['epoch'], df_model3_hp2['validation_loss'], label=model3+' - Validation Loss', linestyle='--', color='orange', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for Hyperparameter Set 2 - QLoRA')
plt.legend()
plt.grid(True)
plt.show()

# Plotting accuracy for Hyperparameter Set 1
plt.figure(figsize=(12, 6))

plt.plot(df_model1_hp1['epoch'], df_model1_hp1['accuracy'], label=model1+' - Accuracy', linestyle='-', color='blue', marker='o')
plt.plot(df_model1_hp1['epoch'], df_model1_hp1['f1_score'], label=model1+' - F1-score', linestyle='--', color='blue', marker='o')

plt.plot(df_model2_hp1['epoch'], df_model2_hp1['accuracy'], label=model2+' - Accuracy', linestyle='-', color='green', marker='o')
plt.plot(df_model2_hp1['epoch'], df_model2_hp1['f1_score'], label=model2+' - F1-score', linestyle='--', color='green', marker='o')

plt.plot(df_model3_hp1['epoch'], df_model3_hp1['accuracy'], label=model3+' - Accuracy', linestyle='-', color='orange', marker='o')
plt.plot(df_model3_hp1['epoch'], df_model3_hp1['f1_score'], label=model3+' - F1-score', linestyle='--', color='orange', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Accuracy / F1-score')
plt.title('Accuracy and F1-score for Hyperparameter Set 1 - QLoRA')
plt.legend()
plt.grid(True)
plt.show()

# Plotting accuracy for Hyperparameter Set 2
plt.figure(figsize=(12, 6))

plt.plot(df_model1_hp2['epoch'], df_model1_hp2['accuracy'], label=model1+' - Accuracy', linestyle='-', color='blue', marker='o')
plt.plot(df_model1_hp2['epoch'], df_model1_hp2['f1_score'], label=model1+' - F1-score', linestyle='--', color='blue', marker='o')

plt.plot(df_model2_hp2['epoch'], df_model2_hp2['accuracy'], label=model2+' - Accuracy', linestyle='-', color='green', marker='o')
plt.plot(df_model2_hp2['epoch'], df_model2_hp2['f1_score'], label=model2+' - F1-score', linestyle='--', color='green', marker='o')

plt.plot(df_model3_hp2['epoch'], df_model3_hp2['accuracy'], label=model3+' - Accuracy', linestyle='-', color='orange', marker='o')
plt.plot(df_model3_hp2['epoch'], df_model3_hp2['f1_score'], label=model3+' - F1-score', linestyle='--', color='orange', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Accuracy / F1-score')
plt.title('Accuracy and F1-score for Hyperparameter Set 2 - QLoRA')
plt.legend()
plt.grid(True)
plt.show()