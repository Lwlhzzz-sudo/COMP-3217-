import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score

def random_forest():
    # Load the training data
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]

    # Split the training data into training set and testing set
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100)
    rf_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = rf_classifier.predict(X_test_split)
   
    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)
    print("Accuracy on the Random forest testing set:", accuracy)

def linear_regression():
    # Load the training data
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]

    # Split the training data into training set and testing set
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a linear regression classifier
    linear_classifier = linear_model.LinearRegression()
    linear_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = linear_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions.round())



    print("Accuracy on the linear regression testing set:", accuracy)

def naive_bayes():
    # Load the training data
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]

    # Split the training data into training set and testing set
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a naive bayes classifier
    naive_bayes_classifier = GaussianNB()
    naive_bayes_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = naive_bayes_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)


    print("Accuracy on the naive bayes testing set:", accuracy)

def svm_classifier():
    # Load the training data
    scaler = StandardScaler()
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]
    X_train = scaler.fit_transform(X_train)
    # Split the training data into training set and testing set
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a svm classifier
    svm_classifier = SVC()
    svm_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = svm_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)


    print("Accuracy on the svm testing set:", accuracy)

def knn_classifier():
    # Load the training data
    scaler = StandardScaler()
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]
   
    X_train = scaler.fit_transform(X_train)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a svm classifier
    knn_classifier = KNeighborsClassifier()
    knn_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = knn_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)


    print("Accuracy on the knn testing set:", accuracy)


def logistic_regression():
    # Load the training data
    scaler = StandardScaler()
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]
   
    X_train = scaler.fit_transform(X_train)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a svm classifier
    logistic_classifier = LogisticRegression()
    logistic_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = logistic_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)

    print("Accuracy on the logistic testing set:", accuracy)

def decision_tree():
    # Load the training data
    scaler = StandardScaler()
    file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
    training_data = pd.read_csv(file_path, header=None)

    # Separate the features (columns 0-127) and the labels (column 128)
    X_train = training_data.iloc[:, 0:128]
    y_train = training_data.iloc[:, 128]
   
    X_train = scaler.fit_transform(X_train)
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train, y_train, test_size=0.2, random_state=38)

    # Train a svm classifier
    decision_classifier = DecisionTreeClassifier()
    decision_classifier.fit(X_train_split, y_train_split)

    # Make predictions on the testing set
    predictions = decision_classifier.predict(X_test_split)

    # Compute the accuracy on the testing set
    accuracy = accuracy_score(y_test_split, predictions)
    print("Accuracy on the decision tree testing set:", accuracy)

#logistic_regression()
#linear_regression()
#svm_classifier()
#naive_bayes()
#knn_classifier()
#decision_tree()
random_forest()


# Load the training data
file_path = "C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TrainingDataMulti.csv"
training_data = pd.read_csv(file_path, header=None)

# Separate the features (columns 0-127) and the labels (column 128)
X_train = training_data.iloc[:, 0:128]
y_train = training_data.iloc[:, 128]


# Train a Random Forest classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)

# Load the testing data
test_data = pd.read_csv("C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TestingDataMulti.csv", header=None)

# Make predictions on the testing set
predictions = rf_classifier.predict(test_data)

print("Predictions:", predictions)
#save the predictions to a csv file
result = pd.DataFrame(test_data)
result['Id'] = predictions
result.to_csv("C:\\Users\\lwlhzzz\\Desktop\\3217 -lab\\TestingResultsMulti.csv", index=False, header=False)
