# 🧠 DeepNLP_SynLSTM_Alerts

A multitask NLP system that performs **Named Entity Recognition (NER)** and **Sentiment Analysis (SA)** to generate real-time, structured alerts from social media posts or news articles. The architecture integrates custom modules including character-level embeddings, dependency-based Graph Convolutional Networks, and a SynLSTM encoder for linguistic context. 📄 For a complete theoretical and architectural explanation of the system, please refer to the [project report](./DeepNLP_SynLSTM_Alerts_Report.pdf).

---

## 💡 Overview

This system processes raw text to detect named entities (such as persons, organizations, monetary values, locations, etc.) and simultaneously classify the sentiment (positive, neutral, or negative). The outputs of both tasks are combined to generate contextual alerts, which can be used for trend detection, early warning systems, or monitoring pipelines.

---

## 🧱 Architecture

The core architecture supports flexible multitask modeling, with several variants:

- **Character-level BiLSTM** for fine-grained token representation
- **GCNs** over syntactic dependency trees
- **SynLSTM** to integrate syntactic roles into LSTM updates
- Independent or shared task-specific LSTMs for NER and SA

## 📁 Repository Structure

- `data/`: Contains the datasets used in the project.
- `src/`: Includes the source code.
- `conll2003_data.pkl`: Serialized file with the preprocessed CoNLL-2003 dataset.

---

## ⚙️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/miguelangelhuamani/DeepNLP_SynLSTM_Alerts.git
   ```

2. **Install dependencies**:

   It's recommended to use a virtual environment to manage dependencies:

   ```bash
   cd PracticaFinalDeepNLP
   python3 -m venv env
   source env/bin/activate  # On Windows: 'env\Scripts\activate'
   ```

   Then install the packages:

   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
3. **Download GloVe embeddings**

   To ensure the system works correctly, you need the pre-trained GloVe vectors (100 dimensions).

   1. Go to the following page — the .zip file will download automatically:
      [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip)
   
   2. Extract the contents of the ZIP file.
   
   3. Copy the file glove.6B.100d.txt to the root directory of the project, at the same level as src/, like this:
      ```bash
      /your-project/
      ├── src/
      ├── glove.6B.100d.txt
      ├── requirements.txt
      ├── venv/
      └── README.md
      ```


## 🚀 Usage

### Model Training

To train the NER and SA models:

```bash
python -m src.train
```

The trained weights will be saved in the `models/` directory.

### Evaluation

To evaluate the models:

```bash
python -m src.evaluate
```

### Alert Generation

To generate alerts from new input text:
Edit your input sentence directly in the `new_prediction.py` script and run:

```bash
python -m src.new_prediction
```
## 🧠 Generated Models

The system allows the creation of five different model variants, controlled by three configuration flags:

| Model    | use\_char\_embs | use\_separate\_lstms | use\_syn\_lstm |
|----------|------------------|-----------------------|----------------|
| Model A  | True             | True                  | True           |
| Model B  | False            | True                  | True           |
| Model C  | False            | False                 | True           |
| Model D  | False            | False                 | False          |
| Model E  | True             | False                 | True           |

These flags enable comparisons of model performance and complexity when using or excluding character embeddings, SynLSTM, and task-specific LSTM layers.

## 📊 Dataset

The system uses the CoNLL-2003 dataset for Named Entity Recognition. The file `conll2003_data.pkl` must be located in the project’s root directory.

## 🤝 Contributions

1. Fork the repository
2. Create a new branch: `git checkout -b feature/new-feature`
3. Make your changes and commit: `git commit -am 'Add new feature'`
4. Push to your branch: `git push origin feature/new-feature`
5. Open a Pull Request

## 📬 Contact

For questions or suggestions, feel free to contact the project team via GitHub Issues or Pull Requests.
