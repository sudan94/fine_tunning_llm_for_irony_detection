import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import glob

# Path to the directory containing the CSV files
directory_path = 'output/binary/fine-tuned/'
# directory_path = 'output/multi-class/fine-tuned/'    # Uncomment this line for multi-class classification

# Get all CSV files in the directory
file_paths = glob.glob(os.path.join(directory_path, '*.csv'))
naming = directory_path.split('/')[1];

results = []
misclassifications_summary = []

for file_path in file_paths:
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Assuming the CSV has columns 'predicted_label', 'actual_label', and 'text'
    predicted_labels = df['predicted_label']
    actual_labels = df['actual_label']

    # Compute confusion matrix
    cm = confusion_matrix(actual_labels, predicted_labels)

    # Compute accuracy, precision, recall, and F1 score
    accuracy = accuracy_score(actual_labels, predicted_labels)
    report = classification_report(actual_labels, predicted_labels, output_dict=True)

    # Store the results
    model_name = os.path.basename(file_path).split('.')[0]
    misclassified_df = df[predicted_labels != actual_labels]
    num_misclassifications = len(misclassified_df)

    results.append({
        'model_name': model_name,
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'classification_report': report,
        'misclassified': misclassified_df,
        'num_misclassifications': num_misclassifications
    })

    misclassifications_summary.append({
        'model_name': model_name,
        'num_misclassifications': num_misclassifications
    })

# Identify the best model based on accuracy
best_model = max(results, key=lambda x: x['accuracy'])
best_model_name = 'FlanT5 - HP Set 1 - Qlora'
best_model_metrics = best_model

# Visualization of confusion matrix for the best model
cm = best_model_metrics['confusion_matrix']

plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix for {best_model_name}')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

print(f'Performance metrics for {best_model_name}:')
print(f"Accuracy: {best_model_metrics['accuracy']}")
print(pd.DataFrame(best_model_metrics['classification_report']).transpose())
print('\n' + '='*60 + '\n')

# Analyze misclassified instances for the best model
misclassified_df = best_model_metrics['misclassified']
if not misclassified_df.empty:
    print(f'Misclassified instances for {best_model_name}:')
    print(misclassified_df[['tweet', 'actual_label', 'predicted_label']])
    print('\n' + '='*60 + '\n')

# Summarize findings from textual examination for the best model
if not misclassified_df.empty:
    print(f'Common patterns in misclassified instances for {best_model_name}:')
    # Example: Group by actual and predicted labels to find common misclassifications
    summary = misclassified_df.groupby(['actual_label', 'predicted_label']).size().reset_index(name='count')
    print(summary)
    print('\n' + '='*60 + '\n')

# Save the misclassification summary to a CSV file
summary_df = pd.DataFrame(misclassifications_summary)
summary_df.to_csv(naming+'_misclassifications_summary.csv', index=False)