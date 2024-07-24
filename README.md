# Fine Tuning LLM for Irony Detection - Master Thesis

This project focuses on fine-tuning Large Language Models (LLMs) for the task of irony detection. The models are trained and evaluated using both binary and multi-class classification tasks.

## Project Structure

- **dataset/**: Contains the datasets for both binary and multi-class classification tasks.
  - **binary/**: Contains `train.csv`, `test.csv`, and `validation.csv` for binary classification.
  - **multi-class/**: Contains `train.csv`, `test.csv`, and `validation.csv` for multi-class classification.

- **data_split.py**: Script to split the dataset into training, validation, and test sets.

- **fine-tuning-scripts/**: Contains Jupyter notebooks for fine-tuning different models using LoRA and qLoRA techniques.
  - **binary/**: Fine-tuning scripts for binary classification.
  - **multi-class/**: Fine-tuning scripts for multi-class classification.

- **output/**: Contains the results of the fine-tuned and non-fine-tuned models.
  - **binary/**: Results for binary classification.
  - **multi-class/**: Results for multi-class classification.


## How to Use

### Running Fine-Tuning Scripts

The fine-tuning scripts are provided as Jupyter notebooks and can be run on Google Colab. Follow these steps to run the notebooks:

1. **Open Google Colab**: Go to [Google Colab](https://colab.research.google.com/).

2. **Upload the Notebook**: Click on "File" > "Upload Notebook" and upload the desired notebook from the `fine-tuning-scripts` directory.

3. **Connect to Runtime**: Ensure you are connected to a GPU runtime for faster training. You can do this by clicking on "Runtime" > "Change runtime type" and selecting "GPU".

4. **Create Hugginface account**: Create a hugginface account and get the API key for accessing the models. Make sure to add the dataset `train.csv`, `test.csv`, and `validation.csv` to the Huggingface dataset platform to access through its library.

5. **Run the Notebook**: Execute the cells in the notebook sequentially. Follow any additional instructions provided in the notebook.

## Data Source

This repository includes the dataset from the paper "[Sentiment Analysis in the Era of Large Language Models: A Reality Check](https://arxiv.org/abs/2305.15005)" originally available in the [LLM-Sentiment](https://github.com/DAMO-NLP-SG/LLM-Sentiment) repository by [DAMO-NLP-SG](https://github.com/DAMO-NLP-SG).

[![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg
