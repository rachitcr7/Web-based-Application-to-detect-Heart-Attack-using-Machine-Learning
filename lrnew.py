#Logistic Regression (85.25%)

#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from scipy import stats

#importing the dataset
df = pd.read_csv("heart1.csv")

#Exploration
df.head()
df.target.value_counts()
sns.countplot(x="target", data=df, palette="bwr") #Target values 0 vs 1
plt.show()

countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100)))

sns.countplot(x='sex', data=df, palette="mako_r") #Males vs Females
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

countFemale = len(df[df.sex == 0]) #Count female vs male
countMale = len(df[df.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(df.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(df.sex))*100)))

df.groupby('target').mean() #mean of each attribute acc. to target 0 vs 1

pd.crosstab(df.age,df.target).plot(kind="bar",figsize=(20,6)) #acc. to age representaion
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.ylabel('Frequency')
plt.savefig('heartDiseaseAndAges.png')
plt.show()

pd.crosstab(df.sex,df.target).plot(kind="bar",figsize=(15,6),color=['#1CA53B','#AA1111' ]) # acc. to sex
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.savefig('heartDiseaseSex.png')
plt.show()

plt.scatter(x=df.age[df.target==1], y=df.thalach[(df.target==1)], c="red") #scatter plot have vs haven't Max Heart Rate
plt.scatter(x=df.age[df.target==0], y=df.thalach[(df.target==0)],c="green")
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Maximum Heart Rate")
plt.savefig('heartDiseaseThalach.png')
plt.show()

pd.crosstab(df.slope,df.target).plot(kind="bar",figsize=(20,6),color=['#DAF7A6','#FF5733' ]) #slope for artery disease prediction
plt.title('Heart Disease Frequency for Slope')
plt.xlabel('The Slope of The Peak Exercise ST Segment ')
plt.xticks(rotation = 0)
plt.ylabel('Frequency')
plt.savefig('heartDiseaseSlope.png')
plt.show()

pd.crosstab(df.fbs,df.target).plot(kind="bar",figsize=(20,6),color=['#FFC300','#581845' ]) #fbs> 120 0 vs 1
plt.title('Heart Disease Frequency According To FBS')
plt.xlabel('FBS - (Fasting Blood Sugar > 120 mg/dl) (1 = true; 0 = false)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.savefig('heartDiseaseFBS.png')
plt.show()

pd.crosstab(df.trestbps,df.target).plot(kind="bar",figsize=(20,6),color=['#FFC300','#581845' ]) #fbs> 120 0 vs 1
plt.title('Heart Disease Frequency According To BP')
plt.xlabel('(Resting Blood Pressure range(80-120) mm/Hg)')
plt.xticks(rotation = 0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency of Disease or Not')
plt.savefig('heartDiseaseBP.png')
plt.show()

pd.crosstab(df.cp,df.target).plot(kind="bar",figsize=(15,6),color=['#11A5AA','#AA1190' ]) # chest pain
plt.title('Heart Disease Frequency According To Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.xticks(rotation = 0)
plt.ylabel('Frequency of Disease or Not')
plt.savefig('heartDiseaseChestPain.png')

plt.show()

#separating the dependent and independent attributes
y = df.target.values 
x_data = df.drop(['target'], axis = 1)

#normalizing the data
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values

#splitting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=3)

#fitting(training) the data
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
#prediction
y_pred = lr.predict(x_test)
coeff=lr.coef_.tolist()[0]
z_values=stats.zscore(df)
z_column=np.mean(z_values,axis=0)
desc=stats.sem(df,axis=0,ddof=0)

#confusion matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(5,5))
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)
plt.savefig('heartDiseaseCM_lr.png')

#accuracy
print("Test Accuracy {:.2f}%".format(lr.score(x_test,y_test)*100))