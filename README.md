# emotion_detection_analysis
Natural language processing helps computers understand speech and written text like a human being. This allows machines to compute necessary responses. One of the many NLP applications is emotion detection in text.

The emotion detection model is a type of model that is used to detect the type of feeling and attitude in a given text. It may be a feeling of joy, sadness, fear, anger, surprise, disgust, or shame.

An emotion detection model can classify a text into the following categories. By using emotion detection in text, businesses can know how customers feel about their brand and products. This helps businesses improve product quality and service delivery.

**Table of contents**

->Exploring our dataset
->Loading dataset
->Getting started with Neattext
->Importing machine learning packages
->Model features and labels
->Dataset splitting
->Pipeline approach
->Model fitting
->Calculating the accuracy score

**Loading dataset**

Use the following command to load the dataset:
df = pd.read_csv("emotion-dataset.csv")

To see how the dataset is structured, use this command:
df.head()

**Removing stopwords**

Stopwords is a list of all the commonly used words in any language. Stopwords carry very little helpful information and have minimal impact on the model during training. These words lead to model bias during training. Removing stopwords eliminates unimportant words, allowing the applications to focus on the essential words instead.

Common stopwords are like articles of a given language. They include the words, the, is, and and are in the English language.

**Importing machine learning packages**

Machine learning models have a problem comprehending raw text, they work well with numbers. Machines cannot process the raw text data, and it has to be converted into a matrix of numbers. CountVectorizer is used to convert the raw text into a matrix of numbers. This process depends on the frequency of each word in the entire text. During this process, CountVectorizer extracts important features from the text. They are used as input for the model during training.

**Dataset splitting**

We need to split our dataset into a train set and test set. The model will learn from the train set. We will use the test set to evaluate the model performance and measure the model’s knowledge capability.

We specify the test_size=0.3. This will split our dataset with 70% of data used for training and 30% for testing.

To make the process of training our model faster and automated, we will use a machine learning pipeline. Machine learning pipelines automate the machine learning workflows such as model fitting and training.

**Pipeline approach**

We import Pipeline using the following code:

from sklearn.pipeline import Pipeline
To use Pipeline we need to specify the machine learning stages we want to automate. In this tutorial, we have two processes we want to automate. The first stage is the CountVectorizer process. This stage converts the raw text dataset into a matrix of numbers that a machine can understand.

The second stage is the model training process.

**Model fitting**

To fit the pipeline stages into x_train and y_train, run this code:

pipe_lr.fit(x_train,y_train)

**Calculating the accuracy score**
When the accuracy score is expressed as a percentage, it becomes 82.0%. This is a high accuracy after the first phase of training. Through continuous training, the model will increase the accuracy score. The higher the accuracy score, the better model will be in making predictions.

Our model is now fully trained and tested.
