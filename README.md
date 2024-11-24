---

# Sentiment Analysis and Text Generation with Transformers

This repository demonstrates how to use Hugging Face's `transformers` library to perform tasks such as sentiment analysis, text generation, and speech-to-text transcription. The project utilizes pretrained models for analyzing tweets, generating poetic text, and transcribing audio files.

---

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Text Generation](#text-generation)
  - [Speech-to-Text Transcription](#speech-to-text-transcription)
- [Dependencies](#dependencies)
- [License](#license)

---

## Introduction
This repository showcases applications of Natural Language Processing (NLP) using pretrained transformer models. It includes:
1. Sentiment analysis on airline tweets.
2. Text generation based on prompts or poetic lines.
3. Speech-to-text transcription using the Wav2Vec2 model.

---

## Features
- Sentiment analysis using Hugging Face's `pipeline`.
- Text generation leveraging the GPT-based models.
- Speech-to-text transcription using `Wav2Vec2`.
- Visualization of confusion matrices and sentiment distributions.

---

## Setup

### Prerequisites
- Python 3.8 or higher
- Libraries listed in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/repo-name.git
   cd repo-name
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download sample datasets:
   - Place the `Tweets.csv` file for sentiment analysis in the project directory.
   - Place the `robert_frost_poems.csv` file for text generation in the project directory.
   - Place `audio.wav` for the speech-to-text demo in the project directory.

---

## Usage

### Sentiment Analysis
Analyze the sentiment of tweets from the dataset.
1. Load the dataset:
   ```python
   df = pd.read_csv("Tweets.csv")
   ```
2. Visualize sentiment distribution:
   ```python
   sns.countplot(df, x='airline_sentiment', palette='viridis')
   plt.show()
   ```
3. Run sentiment classification:
   ```python
   predictions = classifier(df['text'].tolist())
   ```
4. Evaluate model performance:
   ```python
   print(f"ROC AUC Score: {roc_auc_score(df['target'], probs)}")
   ```

### Text Generation
Generate text based on prompts or poetry.
1. Load poetic data:
   ```python
   poems = pd.read_csv("robert_frost_poems.csv")
   ```
2. Generate text:
   ```python
   gen("Example prompt", max_length=50)
   ```

### Speech-to-Text Transcription
Transcribe audio files into text.
1. Load and process the audio file:
   ```python
   audio, sampling_rate = librosa.load("audio.wav", sr=16000)
   ```
2. Perform transcription:
   ```python
   transcription = tokenizer.decode(predicted_ids[0])
   print(transcription)
   ```

---

## Dependencies
- Python 3.8+
- Libraries: `transformers`, `torch`, `pandas`, `numpy`, `seaborn`, `matplotlib`, `librosa`

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

## License
This project is licensed under the MIT License. See the LICENSE file for more information.

---
