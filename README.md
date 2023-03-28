# NLP Assignment 2 Report

## Text Preprocessing

For text preprocessing, I used the following libraries:


-   NLTK
-   BeautifulSoup
-   Regex

I performed the following preprocessing steps on the text data:

-   Removed special characters and punctuations 
		- **using Regex**
-   Removed stop words
		- **using NLTK lib**
-   Performed stemming
		- **using NLTK WordNetLemmatizer**
-	Remove any token not exists in the english dictionary
		-	**using NLTK words corpus**
		-	**this also include removing HTML tags and all other none word tokens**
## Text Vectorization

For text vectorization, I have used the following libraries:

-   scikit-learn (**TfidfVectorizer**) 

I performed the following vectorization techniques:
Embedding each document as a vector of : 
-   TF-IDF 
-  Feature Engineering 
	- **add doc_length feature for each row (Scaled using MinMaxScaler)** 
	- **for each row add 91 new columns as one hot encoded for labels words as 1 if the label name occuerd in the text and 0 if not**
## Machine Learning Algorithms

I used the following machine learning algorithms:

-   Logistic Regression
-   Naive Bayes
-   Support Vector Machine (SVM)

## Results

Sample results obtained after applying the three algorithms on the unseen dataset are as follows:

|Algorithm  |Accuracy On Train  |Accuracy On Test  |
|--|--|--|
| Logistic Regression| 0.76 |0.73 |
| Naive Bayes| 0.63 |0.61 |
| SVM| 0.80 |0.67 |


## Models Evaluation

I used the following evaluation metrics:

-   Micro-averaged F1-score
-   Macro-averaged F1-score
-   Accuracy

The evaluation metrics for the three algorithms are as follows:

| Algorithm | Micro-averaged F1-score | Macro-averaged F1-score |  
|--|--|--|
| Logistic Regression | 0.73 | 0.35 |
| Naive Bayes | 0.61 | 0.09 |
| SVM | 0.67 | 0.28 |

Based on the evaluation metrics, we can see that both logistic regression and SVM have their pros and cons. Although the training accuracy for the SVM model is higher than the other models, the test accuracy is lower compared to logistic regression. On the other hand, the logistic regression model has the best micro and macro-averaged F1-scores, indicating that it performs better in classifying all classes.

Therefore, we can conclude that while SVM has a higher training accuracy, logistic regression provides better generalization and performs better on the unseen test data, making it a more suitable model for this classification task.

## Model Enhancement

To improve performance, I can consider adding the following features:

-   sentiment analysis
	- **see the sentiment for the document if its good or bad or nutral may help with classification the subject**
- Topic modeling
	- **Using topic modeling techniques (e.g. LDA) to identify the main topics or themes in the text can provide additional features for classification.**

 - Word embeddings
	- **Using pre-trained word embeddings (e.g. GloVe, Word2Vec) to represent the text.**
