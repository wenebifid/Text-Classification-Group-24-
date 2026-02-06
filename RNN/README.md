# RNN-Based Hate Speech Detection

**Part of Text Classification Group 24 Project**

A comprehensive implementation of Recurrent Neural Networks (RNN) for automated hate speech detection in social media text, comparing three different word embedding approaches: TF-IDF, Word2Vec Skip-gram, and Word2Vec CBOW.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Visual Results](#visual-results)
- [Key Findings](#key-findings)
- [Methodology](#methodology)
- [Reproducibility](#reproducibility)
- [Citation](#citation)
- [Contributors](#contributors)

---

## Project Overview

This project implements and evaluates RNN-based text classification models for detecting hate speech in tweets. The implementation follows a rigorous machine learning pipeline with:

- **Comprehensive data exploration** (4+ visualizations)
- **Systematic preprocessing** (tokenization, stemming, stopword removal)
- **Hyperparameter tuning** (17 configurations tested)
- **Multi-embedding comparison** (TF-IDF, Word2Vec Skip-gram, Word2Vec CBOW)
- **Thorough evaluation** (5 detailed tables, 6 visualizations)

The project demonstrates the effectiveness of semantic embeddings (Word2Vec) over traditional statistical methods (TF-IDF) for capturing linguistic nuances in hate speech detection.

---

## Dataset

**Source**: [Hate Speech and Offensive Language Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)
**Reference**: Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). *Automated Hate Speech Detection and the Problem of Offensive Language*. ICWSM.

### Dataset Characteristics:
- **Size**: 24,783 tweets
- **Classes**:
  - `0`: Hate Speech (~6%)
  - `1`: Offensive Language (~77%)
  - `2`: Neither (~17%)
- **Source**: Twitter API
- **Annotation**: Crowd-sourced via CrowdFlower (3+ annotators per tweet)
- **Challenge**: Severe class imbalance and high vocabulary overlap between hate speech and offensive language

### Data Format:
The dataset (`hatespeech-dataset/data/labeled_data.csv`) contains:
- `count`: Number of CrowdFlower annotators
- `hate_speech`: Number of hate speech annotations
- `offensive_language`: Number of offensive language annotations
- `neither`: Number of neither annotations
- `class`: Majority class label (0, 1, or 2)
- `tweet`: Raw tweet text

---

## Repository Structure

```
RNN/
├── RNN_Complete_Notebook.ipynb          # Main executable notebook (1.3 MB)
├── hatespeech-dataset/                  # Dataset directory (5.7 MB)
│   ├── README.md                        # Dataset documentation
│   └── data/
│       └── labeled_data.csv             # 24,783 annotated tweets
├── notebook-output-files/               # Execution results (1.7 MB)
│   ├── comprehensive_results_report.txt # Main results summary
│   ├── table1_overall_performance.csv   # Overall metrics comparison
│   ├── table2_per_class_performance.csv # Detailed per-class metrics
│   ├── table3_hate_speech_focus.csv     # Hate speech class analysis
│   ├── table4_hyperparameter_tuning.csv # Best configurations
│   ├── table5_qualitative_comparison.csv# Embedding characteristics
│   ├── figure1_overall_accuracy.png     # Accuracy comparison
│   ├── figure2_f1_macro.png             # Macro F1-score comparison
│   ├── figure3_hate_speech_f1.png       # Hate speech F1 comparison
│   ├── figure4_hate_speech_recall.png   # Recall comparison (critical metric)
│   ├── figure5_per_class_precision.png  # Per-class precision breakdown
│   └── figure6_per_class_recall.png     # Per-class recall breakdown
└── README.md                            # This file
```

---

## Installation & Requirements

### Environment Setup

1. **Google Colab (Recommended)**:
   - Upload `RNN_Complete_Notebook.ipynb` to Google Colab
   - Select GPU runtime: Runtime → Change runtime type → GPU
   - **Best GPU**: H100 (if available) for fastest execution

2. **Local Setup**:
   ```bash
   # Python 3.8+
   pip install tensorflow>=2.10.0
   pip install numpy pandas matplotlib seaborn
   pip install nltk scikit-learn gensim
   pip install tabulate
   ```

### Dataset Access

The dataset is included in this repository (`hatespeech-dataset/data/labeled_data.csv`). If needed, download separately from:
```
https://github.com/t-davidson/hate-speech-and-offensive-language
```

---

## Usage

### Running the Complete Analysis

1. **Open the notebook**:
   ```bash
   jupyter notebook RNN_Complete_Notebook.ipynb
   ```
   Or upload to Google Colab.

2. **Execute all cells**: Run → Run all cells (Estimated time: 30-45 minutes on H100)

3. **Outputs generated**:
   - 5 CSV tables saved to `notebook-output-files/`
   - 6 PNG visualizations saved to `notebook-output-files/`
   - Comprehensive text report: `comprehensive_results_report.txt`

### Notebook Sections

The notebook is organized into 10 comprehensive sections:

1. **Setup & Imports**: Environment configuration
2. **Problem Definition**: Dataset justification with citations
3. **Data Exploration**: 4+ visualizations (class distribution, text length, word frequencies)
4. **Preprocessing**: Tokenization, stemming, stopword removal
5. **Architecture Justification**: Why RNN for sequential hate speech detection
6. **RNN + TF-IDF**: 5 hyperparameter configurations
7. **RNN + Skip-gram**: 6 configurations (trainable vs frozen embeddings)
8. **RNN + CBOW**: 6 configurations
9. **Comprehensive Comparison**: Cross-embedding analysis with 5 tables & 6 figures
10. **Discussion**: Limitations, ethical considerations, future work

---

## Results Summary

### Overall Performance

| Embedding Type      | Test Accuracy | F1-Score (Macro) | F1-Score (Weighted) |
|---------------------|---------------|------------------|---------------------|
| TF-IDF              | 89.43%        | 0.6707           | 0.8800              |
| Word2Vec Skip-gram  | **89.85%**    | **0.7166**       | **0.8916**          |
| Word2Vec CBOW       | 89.31%        | 0.7110           | 0.8877              |

### Hate Speech Detection Performance (Critical Class)

| Embedding Type      | Precision | Recall  | F1-Score |
|---------------------|-----------|---------|----------|
| TF-IDF              | 0.5658    | 0.1503  | 0.2376   |
| Word2Vec Skip-gram  | **0.5197**| **0.2762** | **0.3607** |
| Word2Vec CBOW       | 0.4556    | 0.2867  | 0.3519   |

### Best Hyperparameters

**Word2Vec Skip-gram** (Best Overall):
- RNN Units: 64
- Dropout Rate: 0.2
- Learning Rate: 0.001
- Trainable Embeddings: True
- Validation F1 (Macro): 0.7205

**TF-IDF**:
- Embedding Dimension: 256
- RNN Units: 64
- Dropout Rate: 0.2
- Learning Rate: 0.001

**Word2Vec CBOW**:
- RNN Units: 128
- Dropout Rate: 0.3
- Learning Rate: 0.001
- Trainable Embeddings: True

---

## Visual Results

### Performance Comparison Visualizations

#### Overall Accuracy Comparison
![Overall Accuracy](notebook-output-files/figure1_overall_accuracy.png)
*Figure 1: Test accuracy across all three embedding approaches. Skip-gram achieves highest accuracy at 89.85%.*

#### Macro F1-Score Comparison
![Macro F1-Score](notebook-output-files/figure2_f1_macro.png)
*Figure 2: Macro F1-score treats all classes equally, revealing Skip-gram's superior balanced performance (0.7166).*

#### Hate Speech F1-Score
![Hate Speech F1](notebook-output-files/figure3_hate_speech_f1.png)
*Figure 3: F1-score specifically for hate speech detection - the most critical class. Skip-gram leads at 0.3607.*

#### Hate Speech Recall
![Hate Speech Recall](notebook-output-files/figure4_hate_speech_recall.png)
*Figure 4: Recall for hate speech (how much actual hate speech we catch). Skip-gram achieves 27.62% - an 84% improvement over TF-IDF.*

#### Per-Class Precision
![Per-Class Precision](notebook-output-files/figure5_per_class_precision.png)
*Figure 5: Precision breakdown for all three classes. Models excel at offensive language but struggle with hate speech precision.*

#### Per-Class Recall
![Per-Class Recall](notebook-output-files/figure6_per_class_recall.png)
*Figure 6: Recall breakdown showing strong performance on offensive language and neither, but limited hate speech detection.*

---

## Key Findings

### 1. Word2Vec Skip-gram is Best Overall
- **Highest macro F1-score**: 0.7166 (6.8% improvement over TF-IDF)
- **Best hate speech recall**: 27.62% (**84% improvement** over TF-IDF's 15.03%)
- **Semantic embeddings** capture linguistic nuances better than statistical TF-IDF

### 2. Class Imbalance is a Major Challenge
- Hate speech is only 6% of dataset (286/4,957 test samples)
- Models achieve high overall accuracy (89%+) but struggle with minority class
- **Macro F1 is critical metric** - accuracy alone is misleading

### 3. Hate Speech vs Offensive Language Confusion
- High vocabulary overlap between classes
- Even best model (Skip-gram) achieves only 36% F1 on hate speech
- Suggests need for more sophisticated models (LSTM, transformers)

### 4. Semantic Embeddings Outperform Statistical Methods
- Word2Vec (Skip-gram & CBOW) both exceed TF-IDF performance
- Dense semantic representations capture context better than frequency-based TF-IDF
- Skip-gram's focus on context prediction edges out CBOW's word prediction

### 5. Trainable Embeddings Improve Performance
- Fine-tuning Word2Vec embeddings on hate speech data improves F1
- Domain adaptation critical for specialized tasks

---

## Methodology

### Preprocessing Pipeline
1. **Lowercasing**: Normalize text case
2. **Tokenization**: RegexpTokenizer (alphanumeric tokens only)
3. **Stopword Removal**: NLTK English stopwords
4. **Stemming**: Porter Stemmer for morphological normalization
5. **Sequence Padding**: Max length 100 tokens

### Model Architecture
- **Input**: Padded sequences (max length: 100)
- **Embedding Layer**:
  - TF-IDF: 256-dimensional learned embeddings
  - Word2Vec: 100-dimensional pre-trained embeddings (trainable or frozen)
- **RNN Layer**: SimpleRNN (64-128 units) with dropout (0.2-0.3)
- **Output**: Dense layer with softmax (3 classes)
- **Loss**: Sparse categorical crossentropy
- **Optimizer**: Adam

### Train/Validation/Test Split
- **Train**: 60% (14,870 samples) - stratified
- **Validation**: 20% (4,956 samples) - stratified
- **Test**: 20% (4,957 samples) - stratified

### Hyperparameter Tuning
- **Total configurations tested**: 17
- **TF-IDF**: 5 configs (embedding dimensions, RNN units, dropout, learning rate)
- **Skip-gram**: 6 configs (RNN units, dropout, learning rate, trainable embeddings)
- **CBOW**: 6 configs (same parameters as Skip-gram)
- **Selection criterion**: Validation macro F1-score

### Evaluation Metrics
- **Accuracy**: Overall correctness (misleading with imbalance)
- **Precision**: True positives / Predicted positives
- **Recall**: True positives / Actual positives (critical for hate speech)
- **F1-Score**: Harmonic mean of precision and recall
- **Macro F1**: Unweighted average across classes (treats minority class equally)
- **Weighted F1**: Class-frequency weighted average
- **Confusion Matrix**: Detailed error analysis

---

## Reproducibility

### Ensuring Consistent Results

1. **Random Seeds Set**:
   ```python
   np.random.seed(42)
   tf.random.set_seed(42)
   ```

2. **Stratified Splits**: Class proportions maintained across train/val/test

3. **Fixed Hyperparameters**: All configurations documented in Table 4

4. **Pre-trained Embeddings**: Word2Vec models trained with fixed parameters:
   - Vector size: 100
   - Window: 5
   - Min count: 1
   - Workers: 4
   - Epochs: 10

### Replication Steps

1. Download repository
2. Ensure dataset at `hatespeech-dataset/data/labeled_data.csv`
3. Install requirements (see Installation section)
4. Run `RNN_Complete_Notebook.ipynb` from start to finish
5. Compare outputs in `notebook-output-files/` with provided results

---

## Citation

If you use this implementation or findings, please cite:

**Original Dataset**:
```bibtex
@inproceedings{davidson2017automated,
  title={Automated Hate Speech Detection and the Problem of Offensive Language},
  author={Davidson, Thomas and Warmsley, Dana and Macy, Michael and Weber, Ingmar},
  booktitle={Proceedings of the 11th International AAAI Conference on Web and Social Media},
  year={2017}
}
```

**Word2Vec**:
```bibtex
@inproceedings{mikolov2013distributed,
  title={Distributed Representations of Words and Phrases and their Compositionality},
  author={Mikolov, Tomas and Sutskever, Ilya and Chen, Kai and Corrado, Greg and Dean, Jeffrey},
  booktitle={Advances in Neural Information Processing Systems},
  year={2013}
}
```

---

## Contributors

**Text Classification Group 24**
- RNN Implementation: [This repository]
- Traditional ML Models: [Team member]
- LSTM Implementation: [Team member]
- GRU Implementation: [Team member]

---

## Limitations & Future Work

### Current Limitations
1. **Simple RNN architecture**: Limited long-term dependency modeling
2. **Class imbalance**: Minority class (hate speech) underrepresented
3. **Context insensitivity**: Cannot handle sarcasm, quotes, or reclaimed slurs
4. **Dataset bias**: May reflect annotator biases (e.g., AAE dialect discrimination)

### Recommended Improvements
1. **Advanced architectures**: LSTM/GRU for better memory, transformers (BERT) for context
2. **Class balancing**: SMOTE oversampling or class weights
3. **Contextual embeddings**: BERT, RoBERTa for dynamic word representations
4. **Ensemble methods**: Combine multiple embeddings for robust predictions
5. **Fairness audits**: Test across demographic groups to detect bias
6. **Threshold tuning**: Optimize decision boundary based on false positive/negative costs

### Ethical Considerations
- **Bias**: Models may perpetuate racial, gender, or cultural biases from training data
- **Context**: Cannot distinguish quotes, education, satire, or reclaimed language
- **Deployment**: Requires human oversight; automated moderation risks censorship
- **Transparency**: Model decisions should be explainable to affected users

---

## License

This project is for educational purposes as part of a university assignment. The dataset is provided by Davidson et al. (2017) under their original license terms.

---

## Acknowledgments

- **Dataset**: Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber
- **Frameworks**: TensorFlow, Keras, Gensim, NLTK, scikit-learn
- **Computational Resources**: Google Colab (GPU: H100)

---

**Last Updated**: February 2026
**Repository**: Text-Classification-Group-24-/RNN/
