
# Customer Churn Prediction

## Overview

Customer churn prediction is a classification problem that aims to identify customers who are likely to stop using a company's services. Early identification of at-risk customers enables businesses to take proactive retention measures and reduce customer loss.

The objective of this project was to build and evaluate multiple Machine Learning and Deep Learning models to accurately predict customer churn and select the most effective model based on performance metrics.

---

# Machine Learning Models

Several machine learning algorithms were trained and evaluated throughout the project.

## Logistic Regression

Logistic Regression served as a strong baseline model and demonstrated consistent performance across training and testing datasets. Due to its ability to generalize well and maintain a strong balance between Precision and Recall, it emerged as one of the top-performing models.

## Decision Tree

Decision Tree models were trained to capture non-linear relationships within the dataset. While they performed reasonably well on the training data, they tended to overfit and showed weaker generalization performance on unseen data compared to Logistic Regression.

## XGBoost

XGBoost was also evaluated due to its strong reputation in structured tabular datasets. Various hyperparameter configurations were tested; however, the model did not achieve a better overall F1 Score than Logistic Regression.

---

# Deep Learning Models

Deep Learning approaches were also explored to determine whether more complex architectures could extract additional patterns from the data.

## Wide Neural Network

A wide neural network architecture with a larger number of neurons per hidden layer was implemented and tested. Although the model achieved good training performance, it struggled to generalize effectively on validation and test data.

## Deep Neural Network

A deeper architecture containing multiple hidden layers was also developed and evaluated. Different configurations, activation functions, optimizers, and regularization techniques were tested.

Despite these efforts, the deep neural network exhibited noticeable overfitting. Training metrics improved significantly while validation and testing performance remained relatively weaker. This behavior is likely due to the limited dataset size, which restricted the model's ability to learn robust generalizable patterns.

---

# Handling Class Imbalance

Customer churn datasets often contain class imbalance, where non-churn customers significantly outnumber churn customers.

To address this challenge, multiple balancing techniques were explored:

## SMOTE

SMOTE (Synthetic Minority Oversampling Technique) was applied to generate synthetic samples for the minority class and improve class representation.

## Class Weighting

Different class weight configurations were tested to increase the importance of churned customers during model training.

Although both approaches improved certain metrics during experimentation, neither produced a better overall F1 Score than Logistic Regression.

---

# Results

After evaluating all Machine Learning and Deep Learning models, the best-performing model was **Logistic Regression**.

## Confusion Matrix

```text id="z8y31m"
[[1213  484]
 [ 100  528]]
```

## Performance Metrics

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 74.88% |
| Precision | 52.17% |
| Recall    | 84.08% |
| F1 Score  | 64.39% |

---

# Why Logistic Regression Was Selected

The final model selection was based primarily on the **F1 Score**, as customer churn prediction requires a balance between Precision and Recall rather than maximizing Accuracy alone.

Although Decision Tree, XGBoost, Wide Neural Networks, and Deep Neural Networks were thoroughly evaluated, Logistic Regression consistently delivered the best balance of performance metrics.

Key reasons for selecting Logistic Regression:

* Achieved the highest overall F1 Score among the tested models.
* Maintained strong Recall (84.08%), allowing the model to identify the majority of churned customers.
* Demonstrated better generalization compared to Deep Learning models, which showed signs of overfitting.
* Outperformed models trained using SMOTE and class weighting strategies.
* Produced stable and interpretable predictions suitable for deployment.

Based on comprehensive experimentation with multiple Machine Learning and Deep Learning techniques, Logistic Regression was selected as the final model for deployment because it provided the most reliable and balanced performance for customer churn prediction.


## 🚀 Live Demo

https://costumer-churn-prediction-by-adnan.streamlit.app/