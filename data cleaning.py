#task 2:
#do the data cleaning and EDA
import numpy as np
import pandas as pd
import seaborn as sns
gag= pd.read_csv('C:\\Users\\KRISHNAMIDULA K\\OneDrive\\Documents\\GitHub\\PRODIGY_DS_02\\Titanic-Dataset.csv')
gag.head()

#data cleaning:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#find the missing values:
gag.isnull().sum()
#age,cabin,embarked has missing values
# fill the missing values
#for age the the outliers difference is big hence we are using the median value to fill the missing data
gag['Age'].fillna(gag['Age'].median(),inplace=True)
print(gag['Age'])
print('the missing values after data cleaning is:',gag['Age'].isnull().sum())

#fill the missing values of cabin
#cabin has to many missing values and the can't be categorize with the help of mean ,median and mode hence we are dropping it:

#the output shows cabin column is found meant it is dropped already
gag.head()
#cabin is dropped

#fill the missing values of embarked with mode since it is a categorical data:
gag['Embarked'].fillna(gag['Embarked'].mode()[0],inplace=True)
gag['Embarked'].isnull().sum()

#now remove the duplicates
gag.drop_duplicates(inplace=True)
print('no of duplicates after removing ',gag.duplicated().sum())

#check the datatypes of columns
for column in gag.columns:
    print('columns_dtype:',gag[column].dtype)
#converting the datatypes
gag['Survived']=gag['Survived'].astype('int64')
gag['Fare']=gag['Fare'].astype('float64')
gag['Age']=gag['Age'].astype('int64')
gag['Embarked']=gag['Embarked'].astype('category')


#handling the outliers
#fare has so many outliers
#handling the outliers with interquartile range method:
q1=gag['Fare'].quantile(0.25)
q3=gag['Fare'].quantile(0.75)
IQR=q3-q1
#calculating the outliers bound
lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR
print('the outliers bound are',lower_bound,upper_bound)
gag['Fare']=np.where(gag['Fare']<lower_bound,lower_bound,gag['Fare'])
gag['Fare']=np.where(gag['Fare']>upper_bound,upper_bound,gag['Fare'])

print(gag['Fare'])

#checking the outliers
outliers = gag[(gag['Fare'] < lower_bound) | (gag['Fare'] > upper_bound)]
print(f"Number of outliers in 'Fare': {outliers.shape[0]}")

#encoding the categorical values as numerical values:
gag['Sex']=gag['Sex'].map({'male':0,'female':1})
gag['Embarked']=gag['Embarked'].map({'S':1,'C':0,'Q':2,})
gag.head()

#dropping the irrelevant columns:
#it doesn't provide any info about the likelihood of survival and stats about it
gag.drop(columns=['Name','Ticket'],inplace=True)

gag.head()

#saving the cleaned data
cleaned = 'cleaned_titanic_data.csv'
gag.to_csv(cleaned, index=False)
print(f"Data cleaning completed. Cleaned dataset saved to '{cleaned}'.")
