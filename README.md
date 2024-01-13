<!--
Copyright (c) Xuanting Chen.
Licensed under the MIT License.
-->

# Twitter Offensive Language Detection

This project focuses on detecting offensive language in tweets using various machine learning models. It allows users to train models on a dataset of tweets and evaluate the performance of these models in identifying offensive content.

## Getting Started

### Prerequisites

Before you can run this project, ensure you have Python installed on your machine. Additionally, you might need to install specific Python libraries. You can install these dependencies via pip (if your Python version is under 3.8, you might need `pickle5` for the latest features):

```bash
pip install -r requirements.txt
```


### Cloning the Repository

To clone the repository and run it on your local machine, execute the following command in your terminal:

```bash
git clone https://github.com/ChenXuanting/TwitterOffensiveLanguageDetection.git
```

This will download the code to your local machine.

### Training the Model

To train a model, navigate to the project directory and run `train_model.py` with the desired model name as an argument. For example:
```bash
python train_model.py albert-base-v1
```

This will start the training process for the specified model. The trained model and its predictions will be saved in the `cached-results` folder.

### Available Models

You can train the models with any of the following pre-trained model architectures:

- Albert: 'albert-base-v1', 'albert-large-v1', 'albert-xlarge-v1', 'albert-xxlarge-v1', 'albert-base-v2', 'albert-large-v2', 'albert-xlarge-v2', 'albert-xxlarge-v2'
- Roberta: 'roberta-base', 'roberta-large'
- XLM-Roberta: 'xlm-roberta-base', 'xlm-roberta-large'
- BERT: 'bert-base-uncased', 'bert-large-uncased'

Browse [Hugging Face Models](https://huggingface.co/models) for more pretrained models.

### Modifying Hyperparameters

To change the hyperparameters for model training, you can modify the `train_model.py` file. Adjust the parameters as needed to optimize the model's performance.

### Ensemble Model Performance

After training multiple models, you can evaluate their ensemble performance by running `ensemble.py`. This script combines the predictions from different models to potentially improve the overall accuracy.
```bash
python ensemble.py
```
### Our Results
All numbers are multiplied by 100 for clearer demonstration.
| Model                   | Macro F1 | Accuracy |
|-------------------------|----------|----------|
| All NOT                 | 41.93    | 72.21    |
| All OFF                 | 21.74    | 27.79    |
| *Single Models*         |          |          |
| BERT-base               | 90.93    | 92.26    |
| BERT-large              | 91.42    | 92.74    |
| RoBERTa-base            | 91.70    | 92.87    |
| RoBERTa-large           | **91.86**| **93.10**|
| RoBERTa-large MLM       | *91.99*  | *93.21*  |
| ALBERT-large-v1         | 91.50    | 92.15    |
| ALBERT-large-v2         | 91.49    | 92.13    |
| ALBERT-xxlarge-v1       | 91.39    | 92.42    |
| ALBERT-xxlarge-v2       | 91.55    | 92.91    |
| *Ensembles*             |          |          |
| BERT                    | 91.60    | 93.15    |
| RoBERTa                 | 91.83    | 93.03    |
| ALBERT-all              | **91.90**| **93.49**|
| ALBERT-xxlargea         | 91.58    | 93.27    |

## Contributing

Feel free to fork this repository and submit pull requests with any enhancements. You can also open issues for any bugs found or features you think would be beneficial.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

