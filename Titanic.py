import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
train_df = pd.read_csv('../TitanicData/train.csv')
test_df = pd.read_csv('../TitanicData/test.csv')
combine = [train_df, test_df]

# Print out column headers
print(list(combine[0]))

# Determine if variable is categorical
for column in combine[0]:
    columnObj = combine[0][column]
    print('Column Name: ' + column)
    print('Data Type: ' + str(columnObj.dtype))

print('\n--------------Null or Empty--------------\n')

# Check for null, blank, or empty value in a column
print(train_df.isnull().sum())
print(test_df.isnull().sum())
print(train_df.isna().sum())
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
combinedDF['Pclass'] = combinedDF.PassengerId.astype('category')
print(combinedDF.describe(include='all'))



