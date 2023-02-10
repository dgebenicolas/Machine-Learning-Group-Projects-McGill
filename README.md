# Machine-Learning-Projects

Project 1 Summary :

Implemented and compared the performance of two machine learning methods (K-Nearest Neighbors and Decision Tree) on two benchmark datasets (Hepatitis and Diabetic Retinopathy Debrecen).
Determined the optimal hyperparameters for both models through experimentation and validation set analysis.
Extracted key features from both datasets and created decision boundary plots to visualize performance.
Achieved a test accuracy of 81.48% on the Hepatitis dataset and 63.28% on the Diabetic Retinopathy Debrecen dataset using the K-Nearest Neighbors method with hyperparameter of K=1 and Euclidean distance function.
Preprocessed the datasets by removing malformed data and standardizing the feature values.
Analyzed the class distributions and feature importance of both datasets to identify key factors affecting performance.

Project #2 Summary:


1) Implemented logistic regression and multi-class regression machine learning models from scratch for sentiment analysis and text classification.
2)  Conducted binary classification on IMDB movie reviews and multi-class classification on 20-news group datasets.
3) Compared performance of logistic regression and multi-class regression with K-Nearest Neighbors using AUROC and classification accuracy metrics.
4) Logistic regression significantly outperformed KNN with an AUROC of 0.8779 vs. KNN's 0.6343 in binary classification of IMDB movie reviews.
5)Multi-class regression outperformed KNN in multi-class classification on 20-news group dataset with a classification accuracy of 74.80% vs. KNN's 46.12%.
6)Conducted prepprocessing and feature selection, including removing rare and stop words, selecting top 100 features for logistic regression and one-hot encoding for multi-class regression.
7)Trained models with different learning rates and measured performance through learning curves, finding an optimal learning rate and iteration number to avoid overfitting.

Project #3 Summary:

Analyzed the performance of Multilayer Perceptron (MLP) and Convolutional Neural Network (CNN) on the Fashion MNIST data set
Experimented with different hyperparameters for MLP such as different number of hidden layers, activation functions, regularization techniques and unnormalized images to determine the optimal architecture for the highest accuracy.
Compared the accuracy of MLP with different architectures and a CNN consisting of 2 convolutional layers, 2 fully connected layers and ReLU activation function.
Implemented early stopping methods for the neural networks and found a new MLP architecture that outperformed all previous ones.
Observed an upward trend in prediction accuracy as the number of hidden layers increased, with the optimal number being 2 hidden layers of 128 neurons each, with a learning rate of 0.1 and a batch size of 128.
Achieved a validation accuracy of 87.98% for the optimal MLP architecture.
