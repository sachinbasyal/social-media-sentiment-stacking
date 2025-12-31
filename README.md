# Project Report: Evolutionary Approach to Sentiment Analysis

**Author:** Sachin Basyal 
**Date:** January 2026  
**Tech Stack:** Python, Scikit-Learn, XGBoost, LightGBM, MLflow, DagsHub, Optuna  

---

## 1. Executive Summary
This research project focused on building a robust Sentiment Analysis classifier for social media comments (Reddit/YouTube). The primary objective was to accurately classify comments into **Positive, Neutral, and Negative** categories, with a specific focus on overcoming the challenges of **class imbalance** and **noisy unstructured text**.

Through a series of six controlled experiments, we evolved our approach from a baseline Random Forest model (Accuracy: ~66%) to a sophisticated Stacking Ensemble (Accuracy: ~86%). This report documents the experimental methodology, the "failures" that led to deeper insights, and the final architectural decisions.

---

## 2. Table of Contents
1. [Phase 1: Exploratory Data Analysis (EDA)](#phase-1-exploratory-data-analysis-eda)
2. [Experiment 01 & 02: Baseline Modeling & Text Representation](#experiment-01--02-baseline-modeling--text-representation)
3. [Experiment 03: Feature Space Optimization](#experiment-03-feature-space-optimization)
4. [Experiment 04: Addressing Class Imbalance](#experiment-04-addressing-class-imbalance)
5. [Experiment 05: Gradient Boosting & Hyperparameter Tuning](#experiment-05-gradient-boosting--hyperparameter-tuning)
6. [Experiment 06: Stacking Ensemble (The Champion Model)](#experiment-06-stacking-ensemble-the-champion-model)
7. [Final Conclusion & Future Work](#final-conclusion--future-work)

---

<a name="phase-1-exploratory-data-analysis-eda"></a>
## Phase 1: Exploratory Data Analysis (EDA)
**Objective:** To understand the distribution, quality, and linguistic characteristics of the dataset before modeling.

**Key Findings:**
* **Class Imbalance:** The dataset was heavily skewed. Positive and Neutral comments dominated, while Negative comments (Class -1) were significantly underrepresented. This posed an early risk of the model biasing toward the majority classes.
* **Text Noise:** Raw data contained significant noise, including emojis, URLs, and slang. We implemented a cleaning pipeline to normalize text (lowercase, punctuation removal) while preserving semantic meaning.
* **Word Clouds:** Positive comments were characterized by words like "good," "great," and "love," while negative comments contained distinct markers like "bad," "hate," and "worst," confirming that vocabulary-based approaches (like TF-IDF) would be effective.

![Word Cloud Visualization](path_to_your_images/wordcloud_eda.png)
*(Placeholder: Upload your wordcloud image to the repo and link it here)*

---

<a name="experiment-01--02-baseline-modeling--text-representation"></a>
## Experiment 01 & 02: Baseline Modeling & Text Representation
**Hypothesis:** A simple Bag-of-Words approach with a Random Forest classifier will serve as a sufficient baseline.

**Methodology:**
We compared **Unigrams (single words)** vs. **Bigrams (word pairs)** vs. **Trigrams**. We utilized a **Random Forest Classifier** for its interpretability and resistance to overfitting.

**Results:**
* **Unigrams:** Failed to capture context (e.g., "not good" was treated as "not" and "good").
* **Bigrams (1,2):** Significantly outperformed Unigrams by capturing negations and short phrases.
* **Performance Ceiling:** The baseline model achieved approximately **66% Accuracy**.

**Conclusion:**
While Bigrams provided the best text representation, the Random Forest model hit a "performance wall." It struggled to distinguish subtle negative sentiments, likely due to the class imbalance identified in EDA.

---

<a name="experiment-03-feature-space-optimization"></a>
## Experiment 03: Feature Space Optimization
**Objective:** To determine the optimal vocabulary size (`max_features`) for the TF-IDF vectorizer.

**Methodology:**
We ran an optimization loop testing vocabulary sizes ranging from 1,000 to 10,000 features. We tracked the trade-off between **Accuracy** and **Computational Cost** using MLflow.

**Observation:**
Surprisingly, increasing the feature size did *not* linearly improve performance. The model reached an "elbow point" at **1,000 features**, where adding more rare words introduced noise (sparsity) rather than signal.

**Decision:**
We standardized on **1,000 features** for the intermediate experiments to maintain training speed, though we noted that advanced models (like Linear Regressors) might benefit from larger feature spaces later.

---

<a name="experiment-04-addressing-class-imbalance"></a>
## Experiment 04: Addressing Class Imbalance
**Problem:** The model was consistently misclassifying Negative comments as Neutral. Recall for the Negative class was unacceptably low (~0.04).

**Methodology:**
We compared three strategies to handle the skewed data:
1.  **Undersampling:** Removing majority samples (Result: Loss of information, poor accuracy).
2.  **Class Weights:** Penalizing the model for missing negative cases (Result: Moderate improvement).
3.  **SMOTE (Synthetic Minority Over-sampling Technique):** Synthetically generating new examples of Negative comments.

**Results:**
**SMOTE** was the decisive winner. By augmenting the minority class in the training set, we forced the model to learn the decision boundary for "Negative" comments. This experiment was the turning point where we began to see the model actually recognize hostility and criticism.

---

<a name="experiment-05-gradient-boosting--hyperparameter-tuning"></a>
## Experiment 05: Gradient Boosting & Hyperparameter Tuning
**Hypothesis:** Tree-based bagging (Random Forest) is too "shallow" for this task. Gradient Boosting (XGBoost), which learns sequentially from errors, will extract more signal.

**Methodology:**
* **Algorithm:** XGBoost (Extreme Gradient Boosting).
* **Optimization:** We employed **Optuna** (Bayesian Optimization) with **Stratified K-Fold Cross-Validation** to tune learning rate, tree depth, and estimators.
* **Correction:** During this phase, we identified and fixed a critical data pipeline bug where the Negative class mapping `{-1: 2}` was dropping rows.

**Results:**
* **Baseline (RF):** ~66.0%
* **XGBoost (Tuned):** **77.47%**

**Analysis:**
The switch to boosting provided a massive **+11.5% gain**. The model showed excellent generalization (CV score and Test score were within 1%), proving it was robust and not overfitting.

---

<a name="final-conclusion--future-work"></a>
## 7. Final Conclusion & Future Work

### Summary of Achievement
This project demonstrates a disciplined, scientific approach to Machine Learning. We did not simply "try models until one worked." Instead, we diagnosed specific bottlenecks (Imbalance, Sparsity, Model Bias) and applied targeted solutions (SMOTE, Feature Tuning, Stacking).

**Final Metrics (Test Set):**
* **Accuracy:** 86%
* **Precision (Weighted):** 0.86
* **Recall (Weighted):** 0.86

### Capabilities Demonstrated
* **MLops:** Utilized MLflow and DagsHub for experiment tracking and reproducibility.
* **Data Engineering:** Built robust pipelines to handle parsing errors, label mapping, and vectorization.
* **Advanced Modeling:** Implemented state-of-the-art Ensemble techniques (Stacking) and Bayesian Optimization (Optuna).

### Future Work
To further elevate this project from a research prototype to a production system, the following steps are proposed:

1.  **Deep Learning Integration:** Explore Transformer-based architectures (**BERT/RoBERTa**) to capture deep semantic context and bidirectional dependencies that TF-IDF vectors may miss.
2.  **Advanced MLOps Pipeline:** Integrate **DVC (Data Version Control)** to build end-to-end MLOps pipelines. This will ensure that large datasets are versioned alongside code, creating a fully reproducible training lineage.
3.  **Real-Time Deployment:** Develop a **Google Chrome Plugin** that utilizes our champion Stacking Model to perform real-time sentiment analysis on **YouTube Live Comments**, providing content creators with instant, actionable feedback on audience engagement.

---
*For more details, please view the [Notebooks](./notebooks) or contact the author.*
