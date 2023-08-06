# flask_app_toxic_commet
## link to collab  https://colab.research.google.com/drive/1cd6dMsWIpZOrBOHmEO3Rq5JkqEhrP_Re?usp=sharing

## Toxic Comment Classification Flask App using LSTM Deep Learning

## Introduction:
The Toxic Comment Classification Flask App is a web application that utilizes a deep learning model built using LSTM (Long Short-Term Memory) to classify toxic comments. It provides a user-friendly interface where users can input a comment, and the app will predict whether the comment contains toxic content.

## Project Overview:

The main goal of this project is to build a machine learning model capable of identifying and flagging toxic comments to create a safer online community.
The model is built using LSTM, a type of recurrent neural network (RNN), which is well-suited for processing sequential data like text.
The Flask web framework is used to create the user interface and deploy the model as a web application.

## Data Collection and Preprocessing:

The model training data is collected from kaggle that contain labeled toxic and non-toxic comments.
The collected data is preprocessed, including text cleaning (removing special characters, punctuation, and lowercasing), tokenization, and padding sequences to ensure consistent input lengths.
## Model Architecture:

The LSTM-based deep learning model is used for text classification.
The model consists of an embedding layer to convert words into dense vectors, followed by LSTM layers for sequence processing, and finally, a dense layer with a sigmoid activation function for binary classification.

## Model Training:

The preprocessed data is split into training and validation sets.
The LSTM model is trained on the training set using appropriate optimization techniques (e.g., Adam optimizer) and binary cross-entropy loss.
Model performance is evaluated on the validation set, and hyperparameter tuning is performed to improve the model's accuracy.


## Web Application using Flask:

The Flask web application is created to provide an interactive interface for users to input comments and get toxicity predictions.
The user interface is designed using HTML, CSS, and JavaScript to ensure a responsive and user-friendly experience.
Flask routes handle user requests, and the comments are passed to the trained LSTM model for classification.
The model's prediction is then displayed on the web application, indicating whether the comment is toxic or non-toxic.

## GitHub Repository:

The project is hosted on GitHub, providing a central repository for the source code, data, model, and documentation.
The README file in the repository contains information about the project, instructions to run the Flask app locally, and any other relevant details.
## Conclusion:
The Toxic Comment Classification Flask App built using LSTM deep learning is a valuable tool to detect toxic comments and promote healthier online discussions. With the app hosted on GitHub, users can easily access this service to identify and address toxic content in various online platforms. The model can be continuously improved and updated to enhance its accuracy and usability. Additionally, the open-source nature of the GitHub repository encourages collaboration and contribution from the community to make the app more robust and effective in classifying toxic comments.
