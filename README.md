<!--
Copyright (c) Xuanting Chen.
Licensed under the MIT License.
-->

# Twitter Offensive Language Detection

This project focuses on detecting offensive language in tweets using various machine learning models. It allows users to train models on a dataset of tweets and evaluate the performance of these models in identifying offensive content.

## Getting Started

### Prerequisites

Before you can run this project, ensure you have Python installed on your machine. Additionally, you might need to install specific Python libraries. You can install these dependencies via pip:

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


## Contributing

Feel free to fork this repository and submit pull requests with any enhancements. You can also open issues for any bugs found or features you think would be beneficial.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

