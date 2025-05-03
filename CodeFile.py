import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load datasets
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
 
# Handle missing values in the training set
imputer_age = SimpleImputer(strategy='median')
train_data['Age'] = imputer_age.fit_transform(train_data[['Age']])

imputer_fare = SimpleImputer(strategy='median')
train_data['Fare'] = imputer_fare.fit_transform(train_data[['Fare']])

# Convert categorical variables to numerical representations in the training set
encoder = OneHotEncoder(drop='first', sparse_output=False)
train_encoded = pd.DataFrame(encoder.fit_transform(train_data[['Sex', 'Embarked']]), columns=encoder.get_feature_names_out(['Sex', 'Embarked']))

# Concatenate encoded features with the original data in the training set
train_data = pd.concat([train_data, train_encoded], axis=1)

# Drop irrelevant columns in the training set
columns_to_drop = ['Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', 'SibSp', 'Parch']
train_data.drop(columns_to_drop, axis=1, inplace=True)

# Split the training data into the training set
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Now handle missing values in the test set
test_data['Age'] = imputer_age.transform(test_data[['Age']])
test_data['Fare'] = imputer_fare.transform(test_data[['Fare']])

# Convert categorical variables to numerical representations in the test set
test_encoded = pd.DataFrame(encoder.transform(test_data[['Sex', 'Embarked']]), columns=encoder.get_feature_names_out(['Sex', 'Embarked']))

# Concatenate encoded features with the original data in the test set
test_data = pd.concat([test_data, test_encoded], axis=1)

# Drop irrelevant columns in the test set
test_data.drop(columns_to_drop, axis=1, inplace=True)

# Save PassengerId before dropping irrelevant columns
passenger_ids = test_data['PassengerId']

# Align feature names for both train and test data
X_train.columns = X_train.columns.astype(str)
X_valid.columns = X_valid.columns.astype(str)

# Re-align test data columns with the training data
missing_cols = set(X_train.columns) - set(test_data.columns)
for c in missing_cols:
    test_data[c] = 0

# Hyperparameter tuning for Support Vector Machine
svm_param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 'auto'], 'kernel': ['linear', 'rbf']}
svm_model = SVC(random_state=42)
svm_grid_search = GridSearchCV(svm_model, svm_param_grid, cv=3, scoring='accuracy')
svm_grid_search.fit(X_train, y_train)
best_svm_model = svm_grid_search.best_estimator_

# Hyperparameter tuning for Neural Network
nn_param_grid = {'hidden_layer_sizes': [(50,),(100,),(50,50),(100,100)], 'activation': ['logistic', 'tanh', 'relu']}
nn_model = MLPClassifier(random_state=42)
nn_grid_search = GridSearchCV(nn_model, nn_param_grid, cv=3, scoring='accuracy')
nn_grid_search.fit(X_train, y_train)
best_nn_model = nn_grid_search.best_estimator_

# Hyperparameter tuning for Random Forest
rf_param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]}
rf_model = RandomForestClassifier(random_state=42)
rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=3, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
best_rf_model = rf_grid_search.best_estimator_

# Use the selected models to make predictions on the validation set
valid_preds_svm = best_svm_model.predict(X_valid)
valid_preds_nn = best_nn_model.predict(X_valid)
valid_preds_rf = best_rf_model.predict(X_valid)

# Calculate accuracies
accuracy_svm = accuracy_score(y_valid, valid_preds_svm)
accuracy_nn = accuracy_score(y_valid, valid_preds_nn)
accuracy_rf = accuracy_score(y_valid, valid_preds_rf)

# Use the selected models to make predictions on the test set
test_preds_svm = best_svm_model.predict(test_data)
test_preds_nn = best_nn_model.predict(test_data)
test_preds_rf = best_rf_model.predict(test_data)

# Create separate DataFrames for each model's predictions
submission_svm = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds_svm})
submission_nn = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds_nn})
submission_rf = pd.DataFrame({'PassengerId': passenger_ids, 'Survived': test_preds_rf})

# Save the submission files for each model
submission_svm.to_csv('submission_svm.csv', index=False)
submission_nn.to_csv('submission_nn.csv', index=False)
submission_rf.to_csv('submission_rf.csv', index=False)

# Print accuracies
print("Support Vector Machine Accuracy:", accuracy_svm)
print("Neural Network Accuracy:", accuracy_nn)
print("Random Forest Accuracy:", accuracy_rf)

# Plot accuracies
models = ['Support Vector Machine', 'Neural Network', 'Random Forest']
accuracies = [accuracy_svm, accuracy_nn, accuracy_rf]

plt.bar(models, accuracies, color=['blue', 'orange', 'green'])
plt.ylabel('Accuracy')
plt.title('Model Accuracies')
plt.ylim([0, 1])
plt.show()