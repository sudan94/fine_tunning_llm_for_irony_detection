import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl

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
mpl.rcParams["font.size"] = 16
# Calculate average metrics for each model and hyperparameter set
metrics = {
    model1+' - HP Set 1': {
        'Accuracy': df_model1_hp1['accuracy'].mean(),
        'F1-score': df_model1_hp1['f1_score'].mean(),
        'Training Loss': df_model1_hp1['training_loss'].mean(),
        'Validation Loss': df_model1_hp1['validation_loss'].mean()
    },
    model1+' - HP Set 2': {
        'Accuracy': df_model1_hp2['accuracy'].mean(),
        'F1-score': df_model1_hp2['f1_score'].mean(),
        'Training Loss': df_model1_hp2['training_loss'].mean(),
        'Validation Loss': df_model1_hp2['validation_loss'].mean()
    },
    model2+' - HP Set 1': {
        'Accuracy': df_model2_hp1['accuracy'].mean(),
        'F1-score': df_model2_hp1['f1_score'].mean(),
        'Training Loss': df_model2_hp1['training_loss'].mean(),
        'Validation Loss': df_model2_hp1['validation_loss'].mean()
    },
    model2+' - HP Set 2': {
        'Accuracy': df_model2_hp2['accuracy'].mean(),
        'F1-score': df_model2_hp2['f1_score'].mean(),
        'Training Loss': df_model2_hp2['training_loss'].mean(),
        'Validation Loss': df_model2_hp2['validation_loss'].mean()
    },
    model3+' - HP Set 1': {
        'Accuracy': df_model3_hp1['accuracy'].mean(),
        'F1-score': df_model3_hp1['f1_score'].mean(),
        'Training Loss': df_model3_hp1['training_loss'].mean(),
        'Validation Loss': df_model3_hp1['validation_loss'].mean()
    },
    model3+' - HP Set 2': {
        'Accuracy': df_model3_hp2['accuracy'].mean(),
        'F1-score': df_model3_hp2['f1_score'].mean(),
        'Training Loss': df_model3_hp2['training_loss'].mean(),
        'Validation Loss': df_model3_hp2['validation_loss'].mean()
    }
}

# Create DataFrame from metrics
df_metrics = pd.DataFrame(metrics).T

# Sort DataFrame by the average accuracy
df_metrics_sorted = df_metrics.sort_values(by='F1-score', ascending=False)

# Plot heatmap
plt.figure(figsize=(16, 12))
sns.heatmap(df_metrics_sorted, annot=True, cmap="YlGnBu", fmt=".3f", linewidths=0.5)
plt.title('Performance Metrics Comparison (Average) - QLoRA')
plt.xlabel('Metrics')
plt.ylabel('Models and Hyperparameter Sets')
plt.tight_layout()
plt.show()
