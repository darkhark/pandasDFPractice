import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.impute import KNNImputer
from fancyimpute import KNN

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train_df = pd.read_csv('../../TitanicData/train.csv')
test_df = pd.read_csv('../../TitanicData/test.csv')
combine = [train_df, test_df]

# Assignment 1

# Print out column headers
print(list(combine[0]))

# Determine if variable is categorical
for column in combine[0]:
    columnObj = combine[0][column]
    print('Column Name: ' + column)
    print('Data Type: ' + str(columnObj.dtype))

print('\n--------------Null or Empty--------------\n')

# Check for null, blank, or empty value in a column
print("Train Null ----------")
print(train_df.isnull().sum())
print("Test Null -------------")
print(test_df.isnull().sum())
print("Train NA ------------")
print(train_df.isna().sum())
print("Test NA --------------")
print(test_df.isna().sum())

print('\n--------------Describe--------------\n')
print(combine[0].describe())
print('\n')
print(combine[1].describe())
print('\n')
combinedDF = pd.merge(combine[0], combine[1], how='outer')
print(combinedDF.describe())

print('\n-----------Question 8----------------\n')
combinedDF['PassengerId'] = combinedDF.PassengerId.astype('category')
combinedDF['Pclass'] = combinedDF.Pclass.astype('category')
print(combinedDF.describe(include='all'))

# Assignment 2

print('\n------------Assignment 2----------------\n')
print('\n------------Question 9----------------\n')
pClass1DF = train_df.loc[train_df['Pclass'] == 1]
rowNumTot = len(pClass1DF.index)
rowNumSurvived = len(pClass1DF.loc[pClass1DF['Survived'] == 1].index)
survivedRatioPclass1 = rowNumSurvived / rowNumTot
print(survivedRatioPclass1)

print('\n------------Question 10----------------\n')
allSurvivors = train_df.loc[train_df['Survived'] == 1]
print(allSurvivors['Sex'].describe(include='all'))
print(342 - 233)

print('\n------------Question 11----------------\n')
survivedAgeGraphs = sns.FacetGrid(train_df, col='Survived')
# bins = 16 gives us age ranges by every 5 years
survivedAgeGraphs.map(plt.hist, 'Age', bins=16)
plt.show()
print("Plots were shown.")


print('\n------------Question 12----------------\n')
survivedPclassAgeGraphs = sns.FacetGrid(train_df, col='Survived', row='Pclass')
survivedPclassAgeGraphs.map(plt.hist, 'Age', bins=16)
survivedAgeGraphs.add_legend()
plt.show()
print("Six plots printed")

print('\n------------Question 13----------------\n')
survivedEmbSexFare = sns.FacetGrid(train_df, row='Embarked', col='Survived')
survivedEmbSexFare.map(sns.barplot, 'Sex', 'Fare', ci=None, order=None)
survivedEmbSexFare.add_legend()
plt.show()
print("Used bar graphs because historgrams wouldn't work")

print('\n------------Question 14----------------\n')
print(train_df["Ticket"].describe(include='all'))
print(1 - (681/891))

print('\n------------Question 15----------------\n')
print(combinedDF.isnull().sum())

print('\n------------Question 16----------------\n')
combinedDF['Sex'] = combinedDF['Sex'].map({'female': 1, 'male': 0}).astype(int)
print(combinedDF.head())

print('\n------------Question 17----------------\n')
# combinedDF['Age'] = KNN(k=3).fit_transform(combinedDF['Age'].values())
imputer = KNNImputer(n_neighbors=3)
combinedDF['Age'] = imputer.fit_transform(combinedDF['Age'].values.reshape(-1, 1))
print(combinedDF.isnull().sum())
print(combinedDF.head())
print("Notice how age now has 0 null values :)")

print('\n------------Question 18----------------\n')
modePort = train_df.Embarked.dropna().mode()[0]
combinedDF['Embarked'] = combinedDF['Embarked'].fillna(modePort)
print(combinedDF.isnull().sum())
print("Notice how embarked now has 0 null values :)")

print('\n------------Question 19----------------\n')
modeFare = train_df.Fare.dropna().mode()[0]
combinedDF['Fare'] = combinedDF['Fare'].fillna(modeFare)
print(combinedDF.isnull().sum())
print("Notice how fare now has 0 null values :)")

print('\n------------Question 20----------------\n')
combinedDF.loc[combinedDF['Fare'] <= 7.91, 'Fare'] = 0
combinedDF.loc[(combinedDF['Fare'] > 7.91) & (combinedDF['Fare'] <= 14.454), 'Fare'] = 1
combinedDF.loc[(combinedDF['Fare'] > 14.454) & (combinedDF['Fare'] <= 31), 'Fare'] = 2
combinedDF.loc[combinedDF['Fare'] > 31, 'Fare'] = 3
print(combinedDF.head())
