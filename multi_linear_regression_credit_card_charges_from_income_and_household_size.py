# Import libraries necessary for data analysis
import pandas as pd

# Import libraries necessary for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Define the file path for the 'Consumer' CSV file
file_path = 'C:/BAN 540/Consumer.csv'

# Read the CSV file into a DataFrame titled 'data'
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame 'data'
print(data.head())
print()

# Get the number of records (rows) and attributes (columns)
num_records = data.shape[0]
num_attributes = data.shape[1]

# Display the results
print(f'Number of records: {num_records}')
print(f'Number of attributes: {num_attributes}\n')

# Get the names of the attributes (columns)
attribute_names = data.columns.tolist()

# Display the attribute names
print('Names of the attributes in the dataset:')
print(attribute_names)
print()

# Calculate statistics for each numeric attribute
stats = data.describe().transpose()

# Select relevant statistics
stats_summary = stats[['min', 'max', 'mean', 'std']]

# Display the summary statistics
print('Summary statistics for each attribute:')
print(stats_summary)
print()

# Check for missing values in each column
missing_values = data.isnull().sum()

# Display the count of missing values for each column
print('Missing values in each column:')
print(missing_values)
print()

# HISTOGRAM FOR AMOUNT CHARGED

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a histogram for the "AmountCharged" column
plt.figure(figsize=(10, 6))
sns.histplot(data['AmountCharged'], bins=10, kde=True)  # kde=True adds a density curve
plt.title('Histogram of Amount Charged')
plt.xlabel('Amount Charged')
plt.ylabel('Frequency')

# Show the plot
plt.show()

# SCATTER PLOT FOR AMOUNT CHARGED VS. INCOME ($1000s)

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a scatter plot for Income vs Amount Charged
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Income($1000s)', y='AmountCharged', data=data, alpha=0.7)

# Add a regression line for better understanding
sns.regplot(x='Income($1000s)', y='AmountCharged', data=data, scatter=False, color='red')

# Customize the plot
plt.title('Impact of Income on Amount Charged')
plt.xlabel('Income ($1000s)')
plt.ylabel('Amount Charged')
plt.xlim(data['Income($1000s)'].min() - 5, data['Income($1000s)'].max() + 5)  # Add some padding
plt.ylim(data['AmountCharged'].min() - 5, data['AmountCharged'].max() + 5)  # Add some padding

# Show the plot
plt.show()

# SCATTER PLOT FOR AMOUNT CHARGED VS. HOUSEHOLD SIZE

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Create a scatter plot for Household Size vs Amount Charged
plt.figure(figsize=(10, 6))
sns.scatterplot(x='HouseholdSize', y='AmountCharged', data=data, alpha=0.7)

# Add a regression line for better understanding
sns.regplot(x='HouseholdSize', y='AmountCharged', data=data, scatter=False, color='red')

# Customize the plot
plt.title('Impact of Household Size on Amount Charged')
plt.xlabel('Household Size')
plt.ylabel('Amount Charged')
plt.xlim(data['HouseholdSize'].min() - 1, data['HouseholdSize'].max() + 1)  # Add some padding
plt.ylim(data['AmountCharged'].min() - 5, data['AmountCharged'].max() + 5)  # Add some padding

# Show the plot
plt.show()

# 3D SCATTER PLOT FOR AMOUNT CHARGED VS. INCOME (1000s) AND HOUSEHOLD SIZE

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(data['Income($1000s)'], data['HouseholdSize'], data['AmountCharged'], alpha=0.7)

# Set labels
ax.set_title('3D Scatter Plot of Income, Household Size, and Amount Charged')
ax.set_xlabel('Income ($1000s)')
ax.set_ylabel('Household Size')
ax.set_zlabel('Amount Charged')

# Show the plot
plt.show()

# Import library necessary for splitting data
from sklearn.model_selection import train_test_split

# Define independent variables (features) and dependent variable (target)
X = data[['Income($1000s)', 'HouseholdSize']]
y = data['AmountCharged']

# Split the data into training (80%) and testing (20%) datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the sizes of the datasets
print(f'Training dataset size: {X_train.shape[0]}')
print(f'Testing dataset size: {X_test.shape[0]}\n')

# Libraries necessary for linear regression and the model's
# coefficient of determination (R-squared)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create the linear regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Predict on the testing data
y_pred = model.predict(X_test)

# Calculate the R-squared value
r2 = r2_score(y_test, y_pred)

# Output the R-squared value
print(f'R-squared value: {r2:.4f}\n')

# Display the coefficients and intercept of the model
coefficients = model.coef_
intercept = model.intercept_

# Create a dictionary to hold the coefficients for easy viewing
coef_dict = {X.columns[i]: coefficients[i] for i in range(len(coefficients))}

# For loop that prints the coefficient value for each variable in the model
print("Coefficients:")
for variable, coefficient in coef_dict.items():
    print(f"{variable}: {coefficient:.4f}")

# Prints the model's intercept
print(f"Intercept: {intercept:.4f}\n")

# Prepare the regression formula
formula = "AmountCharged = {:.4f}".format(intercept)

# Add each coefficient to the formula
for variable, coefficient in coef_dict.items():
    formula += " + {:.4f} * {}".format(coefficient, variable)

# Display the regression formula
print("Regression Formula:")
print(formula)
print()