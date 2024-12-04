# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import data
df = pd.read_csv('/content/Loan_Modelling.csv')

# Header
print("Header:\n")
print(df.head())

# Descriptive statistics
print("\nDescriptive Statistics:\n")
print(df.describe())
print()

# Check for nulls
print("Nulls:\n")
print(df.isnull().sum())

# Filter data to only include minimum 'Experience' to 0 or above
df = df[df['Experience'] >= 0]
print("\nFiltered Data:\n")
print(df.head())

# New descriptive statistics
print("\nNew Descriptive Statistics:\n")
print(df.describe())

# 'Age' histogram
plt.hist(df['Age'], bins=20)
plt.title('Distribution of Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 'Experience' histogram
plt.hist(df['Experience'], bins=20)
plt.title('Distribution of Years of Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Frequency')
plt.show()

# 'Income' histogram
plt.hist(df['Income'], bins=20)
plt.title('Distribution of Income Levels')
plt.xlabel('Income (in 1000s)')
plt.ylabel('Frequency')
plt.show()

# 'Family' histogram
plt.hist(df['Family'], bins=10)
plt.title('Distribution of Family Sizes')
plt.xlabel('Family Size')
plt.ylabel('Frequency')
plt.show()

# 'CCAvg' histogram
plt.hist(df['CCAvg'], bins=20)
plt.title('Distribution of Credit Card Average Monthly Spending')
plt.xlabel('Credit Card Average Monthly Spending (in 1000s)')
plt.ylabel('Frequency')
plt.show()

# 'Mortgage' histogram
plt.hist(df['Mortgage'], bins=20)
plt.title('Distribution of Mortgage Values')
plt.xlabel('Mortgage Value (in 1000s)')
plt.ylabel('Frequency')
plt.show()

# Remove outliers for 'Income'
Q1_income = df['Income'].quantile(0.25)
Q3_income = df['Income'].quantile(0.75)
IQR_income = Q3_income - Q1_income
df = df[(df['Income'] >= Q1_income - 1.5 * IQR_income) & (df['Income'] <= Q3_income + 1.5 * IQR_income)]

# Remove outliers for 'CCAvg'
Q1_ccavg = df['CCAvg'].quantile(0.25)
Q3_ccavg = df['CCAvg'].quantile(0.75)
IQR_ccavg = Q3_ccavg - Q1_ccavg
df = df[(df['CCAvg'] >= Q1_ccavg - 1.5 * IQR_ccavg) & (df['CCAvg'] <= Q3_ccavg + 1.5 * IQR_ccavg)]

# Remove outliers for 'Mortgage'
Q1_mortgage = df['Mortgage'].quantile(0.25)
Q3_mortgage = df['Mortgage'].quantile(0.75)
IQR_mortgage = Q3_mortgage - Q1_mortgage
df = df[(df['Mortgage'] >= Q1_mortgage - 1.5 * IQR_mortgage) & (df['Mortgage'] <= Q3_mortgage + 1.5 * IQR_mortgage)]

# New descriptive statistics
print("\nNew Descriptive Statistics:\n")
print(df.describe())

# Assign the categories for the 'Education' attribute
education_categories = {
    1: 'Undergraduate',
    2: 'Graduate',
    3: 'Advanced/Professional'
}

# Apply mapping to the 'Education' column
df['Education'] = df['Education'].map(education_categories)

# Save data file
df.to_csv('cleaned_loan_data.csv', index=False)

# Average 'Personal_Loan' by Education
education_loan_counts = df.groupby('Education')['Personal_Loan'].value_counts().unstack().fillna(0)
print(education_loan_counts)
print()

# Median 'Income' by 'Personal_Loan'
income_loan_counts = df.groupby('Personal_Loan')['Income'].median()
print(income_loan_counts)
print()

# 'Median 'CCAvg' by 'Personal_Loan'
ccavg_loan_counts = df.groupby('Personal_Loan')['CCAvg'].median()
print(ccavg_loan_counts)
print()

# Visualize Average 'Personal_Loan' by 'Education'
plt.figure(figsize=(10, 8))
sns.barplot(x='Education', y='Personal_Loan', data=df, errorbar=None)
plt.title('% of Accepted Loans During Last Campaign by Education Level')
plt.xlabel('Education Level')
plt.ylabel('% of Accepted Loans')
plt.show()

# Visualize comparison of 'Personal_Loan' by Median 'Income'
plt.figure(figsize=(10, 8))
sns.barplot(x='Personal_Loan', y='Income', data=df, errorbar=None,
            estimator=np.median)
plt.title('Median Income by Personal Loan Acceptance')
plt.xlabel('Personal Loan Acceptance')
plt.ylabel('Median Income (in 1000s)')
plt.show()

# Visualize comparison of 'Personal Loan' by Median 'CCAvg'
plt.figure(figsize=(10, 8))
sns.barplot(x='Personal_Loan', y='CCAvg', data=df, errorbar=None,
            estimator=np.median)
plt.title('Median Credit Card Average Monthly Spending by Personal Loan Acceptance')
plt.xlabel('Personal Loan Acceptance')
plt.ylabel('Median Credit Card Average Monthly Spending (in 1000s)')
plt.show()

# Subset dataset to remove 'Education', 'ZipCode', and 'ID' features
df_subset = df.drop(['Education', 'ZIPCode', 'ID'], axis=1)

# Correlation Matrix
correlation_matrix = df_subset.corr()
correlation_matrix = correlation_matrix.round(2)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True)
plt.title('Correlation Matrix')
plt.show()

# Import necessary libraries for logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import statsmodels.api as sm
import numpy as np

# Import testing and training data
df_train_logistic = pd.read_csv('/content/Loan_train.csv')
df_test_logistic = pd.read_csv('/content/Loan_test.csv')

# Remove columns from data
df_train_logistic = df_train_logistic.drop(['ID', 'ZIPCode', 'Age', 'Experience', 'Mortgage'], axis=1)
df_test_logistic = df_test_logistic.drop(['ID', 'ZIPCode', 'Age', 'Experience', 'Mortgage'], axis=1)

# Create a binary 'Graduate_Degree' column based on 'Education'
df_train_logistic['Graduate_Degree'] = (df_train_logistic['Education'] == 'Graduate').astype(int)
df_test_logistic['Graduate_Degree'] = (df_test_logistic['Education'] == 'Graduate').astype(int)

# Create a binary 'Advanced/Professional' column based on 'Education'
df_train_logistic['Advanced_Professional'] = (df_train_logistic['Education'] == 'Advanced/Professional').astype(int)
df_test_logistic['Advanced_Professional'] = (df_test_logistic['Education'] == 'Advanced/Professional').astype(int)

# Drop the original 'Education' column
df_train_logistic = df_train_logistic.drop('Education', axis=1)
df_test_logistic = df_test_logistic.drop('Education', axis=1)

# Define features
X_train_logistic = df_train_logistic.drop('Personal_Loan', axis=1)
y_train_logistic = df_train_logistic['Personal_Loan']
X_test_logistic = df_test_logistic.drop('Personal_Loan', axis=1)
y_test_logistic = df_test_logistic['Personal_Loan']

# Add constant
X_train_with_constant_logistic = sm.add_constant(X_train_logistic)

# Build logistic regression model
model_logistic = sm.Logit(y_train_logistic, X_train_with_constant_logistic)
result_logistic = model_logistic.fit()

# Print summary
print(result_logistic.summary())
print()

# Print table of exponentiated coefficents
print("Exponentiated Coefficients:")
print(np.exp(result_logistic.params))
print()

# Apply model to test data
X_test_with_constant = sm.add_constant(X_test_logistic)
y_pred = result_logistic.predict(X_test_with_constant)
y_pred_logistic = (y_pred > 0.5).astype(int)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_logistic, y_pred_logistic))
print()
print("Classification Report:\n")
print(classification_report(y_test_logistic, y_pred_logistic))

# Import necessary libraries for ANN modeling
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

# Import data
ann_train = pd.read_csv('/content/Loan_train.csv')
ann_test = pd.read_csv('/content/Loan_test.csv')

# Drop irrelevant columns
ann_train = ann_train.drop(columns=['ID', 'ZIPCode'])
ann_test = ann_test.drop(columns=['ID', 'ZIPCode'])

# Dummy code 'Education' attribute
ann_train = pd.get_dummies(ann_train, columns=['Education'], drop_first=True)
ann_test = pd.get_dummies(ann_test, columns=['Education'], drop_first=True)

# Define the target variable (y) and features (X)
X_train_ann = ann_train.drop(columns=['Personal_Loan'])
y_train_ann = ann_train['Personal_Loan']
X_test_ann = ann_test.drop(columns=['Personal_Loan'])
y_test_ann = ann_test['Personal_Loan']

# Normalize the continuous variables using StandardScaler
scaler = StandardScaler()
X_scaled_ann = scaler.fit_transform(X_train_ann)

# Set seed
tf.random.set_seed(3)

# Build the ANN model
model_ann = Sequential([
    Dense(11, activation='relu', input_shape=(X_train_ann.shape[1],)),  # Input layer with 11 nodes
    Dense(6, activation='relu'),  # Hidden layer with 6 nodes (adjustable to 12 if needed)
    Dense(1, activation='sigmoid')  # Output layer with 1 node for binary classification
])

# Compile the model
model_ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model_ann.fit(X_train_ann, y_train_ann, validation_data=(X_test_ann, y_test_ann), epochs=50, batch_size=32, verbose=1)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

# Show charts
plt.show()

# Apply the ANN Model to testing data set
X_test_scaled_ann = scaler.transform(X_test_ann)
y_pred_ann = model_ann.predict(X_test_scaled_ann)
y_pred_binary_ann = (y_pred_ann > 0.5).astype(int)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test_ann, y_pred_binary_ann))
print()
print("Classification Report:\n")
print(classification_report(y_test_ann, y_pred_binary_ann))

# Import necessary libraries for decision tree modeling
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree

# Import training and testing data
train_tree = pd.read_csv('/content/Loan_train.csv')
test_tree = pd.read_csv('/content/Loan_test.csv')

# Convert 'Education' into numerical form
label_encoder = LabelEncoder()
train_tree['Education'] = label_encoder.fit_transform(train_tree['Education'])
test_tree['Education'] = label_encoder.transform(test_tree['Education'])

# Drop unnecessary columns
train_tree = train_tree.drop(columns=['ID', 'ZIPCode'])
test_tree = test_tree.drop(columns=['ID', 'ZIPCode'])

# Define features
X_train_tree = train_tree.drop(columns=['Personal_Loan'])
y_train_tree = train_tree['Personal_Loan']
X_test_tree = test_tree.drop(columns=['Personal_Loan'])
y_test_tree = test_tree['Personal_Loan']

# Define the decision tree model with prepruning parameters
decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5,              # Limit tree depth to prevent overfitting
    min_samples_split=10,     # Minimum samples required to split a node
    min_samples_leaf=5,       # Minimum samples required at each leaf
    max_leaf_nodes=20         # Limit the maximum number of leaf nodes
)

# Fit the model
decision_tree.fit(X_train_tree, y_train_tree)

# Plot the decision tree
plt.figure(figsize=(20, 8))
plot_tree(decision_tree, filled=True, feature_names=X_train_tree.columns, class_names=['No', 'Yes'])
plt.show()

# Create predictions
y_pred_tree = decision_tree.predict(X_test_tree)

# Evaluate model accuracy using test data and created predictions
val_accuracy_tree = accuracy_score(y_test_tree, y_pred_tree)
val_report_tree = classification_report(y_test_tree, y_pred_tree)

# Confusion matrix
conf_matrix_tree = confusion_matrix(y_test_tree, y_pred_tree)
print("Confusion Matrix:\n")
print(conf_matrix_tree)
print()

# Print accuracy scores and classification report
print("\nClassification Report:\n")
print(val_report_tree)

# Extract feature importance
feature_importances_tree = pd.DataFrame({
    'Feature': X_train_tree.columns,
    'Importance': decision_tree.feature_importances_
})

# Sort features by importance
feature_importances_tree = feature_importances_tree.sort_values(by='Importance', ascending=False)

# Display feature importance
print("Decision Tree Feature Importance:\n")
print(feature_importances_tree)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importances_tree['Feature'], feature_importances_tree['Importance'], color='orange')
plt.gca().invert_yaxis()  # Reverse order for better readability
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance in Decision Tree')
plt.show()