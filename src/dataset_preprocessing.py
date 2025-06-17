import pandas as pd
from sklearn.model_selection import train_test_split

house_df = pd.read_csv('/home/pranil/python_projects/machine_learning/linear_regression/dataset/house_price_regression_dataset.csv')
print(house_df.columns)
print(house_df.head(8))
print(house_df["Num_Bedrooms"].mean())
print(house_df["Square_Footage"].mean())
print(house_df["Square_Footage"].median())

#for information of dataset
print(house_df.info())
#for statistical summary
print(house_df.describe())

print(house_df.head())

#check for missing values
print(house_df.isnull().sum())

print(house_df.duplicated().sum())
house_df.drop_duplicates(inplace=True)

#features and target output
X = house_df.drop('House_Price', axis=1)
print(X)
y = house_df['House_Price']
print(y)

#split the dataset into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)