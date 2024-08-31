

####  PREDICTION USING DECISION TREE CLASSIFIER

##### Dataset : Bank Marketing  dataset

##### Language: Python,Jupyter Notebook

#### Libraries:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations and handling arrays.
- **matplotlib.pyplot**: For creating static, animated, and interactive visualizations.
- **seaborn**: For statistical data visualization.
- **sklearn.model_selection**: For splitting the data into training and testing sets.
- **sklearn.preprocessing**: For scaling features and encoding categorical variables.
- **sklearn.impute**: For handling missing data.
- **sklearn.tree**: For training and visualizing the Decision Tree Classifier.
- **sklearn.metrics**: For evaluating the model's performance using metrics like confusion matrix, accuracy score, and classification report.


### **Overview**
===========

This project aims to predict whether a customer will subscribe to a term deposit (a bank product) using a Decision Tree Classifier. The dataset used is from the UCI Machine Learning Repository and includes various demographic and behavioral attributes of bank customers.


**Table of Contents**

1. [Overview](#overview)
2. [Installation](#installation)
3. [Features](#features)
4. [Visualizations and Insights](#visualizations-insights)
5. [Key Insights](#key-insights)
6. [Conclusion](#conclusion)
7. [Acknowledgments](#acknowledgments)
8. [Author Information](#Author-Information)

## Installation
To run this project,  you will need Python and Jupyter Notebook installed on your system.


## Features

The project utilizes the following features from the bank marketing dataset to predict whether a customer will purchase a product or service:

- **age**: Age of the customer
- **job**: Type of job (e.g., admin, blue-collar, etc.)
- **marital**: Marital status (e.g., married, single)
- **education**: Level of education (e.g., primary, secondary, tertiary)
- **default**: Whether the customer has credit in default (yes/no)
- **balance**: Account balance
- **housing**: Whether the customer has a housing loan (yes/no)
- **loan**: Whether the customer has a personal loan (yes/no)
- **contact**: Type of communication used for contact (e.g., cellular, telephone)
- **day**: Last contact day of the month
- **month**: Last contact month of the year
- **duration**: Duration of the last contact
- **campaign**: Number of contacts performed during this campaign
- **pdays**: Number of days since the client was last contacted from a previous campaign
- **previous**: Number of contacts performed before this campaign
- **poutcome**: Outcome of the previous marketing campaign
- **y**: Whether the customer subscribed to a term deposit (yes/no)

## Visualizations and Insights:

### Confusion Matrix

The confusion matrix provides insights into the model’s performance by showing the number of true positives, true negatives, false positives, and false negatives:


![download](https://github.com/user-attachments/assets/cd0d6752-54f9-49d2-a5bb-3ce78ca6a092)



- **True Negatives (TN):** 1110
- **False Positives (FP):** 95
- **False Negatives (FN):** 93
- **True Positives (TP):** 59
  
- **High Accuracy for 'No' Predictions**: The model performs well at identifying customers who will not buy the service.
- **Challenges with 'Yes' Predictions**: The model has difficulty identifying customers likely to purchase the service, evident from the lower counts in the 'Yes' category.
- **Imbalance**: Reflects the imbalance between the two classes, highlighting the need for potential model improvement or rebalancing techniques.

  
  ### Decision Tree Visualization

  The decision tree visualization shows the decision rules used by the classifier at each node, providing a clear understanding of how the model makes predictions based on the features.
  
 - **First decision tree**
```
plt.figure(figsize=(200,80))
plot_tree(clf, feature_names=X.columns, class_names=label_encoders['y'].classes_, filled=True)
plt.title('Decision Tree for Bank Marketing Prediction')
plt.show()
```


![download](https://github.com/user-attachments/assets/c5f5afed-cdd2-4144-8f78-df1617ea64db)



 - **Second decision tree**

 ```
plt.figure(figsize=(50,10))
plot_tree(clf, 
          filled=True, 
          rounded=True, 
          feature_names=X.columns,  # Use the feature names from the DataFrame
          class_names=['no', 'yes'],  # Target classes
          proportion=True,  # Proportion of samples at each node
          fontsize=10)
plt.show()
```  


![download](https://github.com/user-attachments/assets/cc4e2168-1335-4762-97ac-d676958bb31d)


## Key Insights

- **Confusion Matrix**: The model achieved 86.15% accuracy, effectively predicting most customer behaviors but struggled with the minority class (Yes).
- **Classification Report**: Precision, recall, and F1-score for the positive class (Yes) are lower due to class imbalance, though the model reliably predicts the majority class (No).
- **Decision Tree Visualization**: Provides clear decision rules and insights into how the model classifies customer data.

## Conclusion

The decision tree classifier effectively predicts whether a customer will purchase a product based on demographic and behavioral data from the UCI bank marketing dataset. It achieved an accuracy of 86.15%, demonstrating strong
performance in identifying non-purchasing customers. However, it struggles with predicting purchasing customers due to class imbalance, as reflected in lower precision, recall, and F1-score for the positive class. The decision tree visualization reveals how the model uses different features to make predictions. Overall, while the model is useful, improvements could be made by addressing class imbalance and exploring more advanced techniques for better accuracy on minority classes.

## Acknowledgments
Thanks to the contributors of the libraries used in this project: Pandas, NumPy, Matplotlib, and Seaborn.

Thanks to the creators of the Bank Marketing dataset for providing the data used in this analysis.

Special thanks to the Prodigy Infotech to provide me this opportunity to showcase my skills in Data loading & preprocessing ,train_test_split data , Train models like decision tree classifier, classification report , accuracy score , Visualize Confusion matrix &  decision tree classifier and forming meaningul insights.


### AUTHOR : ARYA S

### LinkedIn : www.linkedin.com/in/arya-dataanalyst

### Thank You for reading!



  


