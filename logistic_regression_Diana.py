# Data Preprocessing

# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the data

dataset = pd.read_csv('Social_Network_Ads.csv')
#Creat Matrix of Features (array of features)
#iloc is a function of pandas, that will take the indexes 
#of columns that we want to extract from the dataset
x = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values


#taking care of missing value and encoding the variables wont be here.

# now we need to split the data set into training and test set
#library that does that for us
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

#Feature Scaling:

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test =sc_x.transform(x_test)

#fitting logistic regression to the training set
#import the correct library, logistic regression is a linear classifier
#here we are in 2D, therefore the 2 caegories will be separated by a straight line
from sklearn.linear_model import LogisticRegression
#as usual we creat an object from  this class which will be our classifier, that
#we are going to fit on our training set.
classifier = LogisticRegression(random_state=0)

# so our logistic regression object is ready, so we take it and fit it to the training set
#we use the fit method, we fit it to training set x and y. so that the classifier
#learns the correlation between x-train and y train, then it will be able to use
#this correlation to predict
classifier.fit(x_train, y_train)

# we check the powe of prediction on the test set
#predicting the test result
y_pred = classifier.predict(x_test)

#evaluating the performance of this logistic regression classifier that we made
# we make a confusion matrix, that holds the correct predictions of our model in test set
#as well as incorrect predictions
# we import a tool that helps us compute the confusion mtrix faster, this tool is a function

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#interesting way to evaluate the model performance: graphic visualizaion of our results
#including decision boundary and decision regions


from matplotlib.colors import ListedColormap
#for simplicity we name our training name to just set
x_set, y_set = x_train, y_train

x1 , x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max() +1, step = 0.01), np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step = 0.01))
# we draw the diagram in a range that changes from min of salary to max of salary
#and from min of age to the max of age
# -1 & +1 is for having a wider range in our graph so that our data points
#dont get smooshed into axes of diagram, step is th resoution of our meshgrids
# then we apply the classifier on all the pixel observation points
# we use contour function to color the 2 prediction regions
#then we use the predict function to predict if each of the pixel points belong to
#class 0 or class one
plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(), x2.max())

# with the loop we plot all the data points that we observe, plt.scatter is for
#making scatter plot

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set==j,1], c=ListedColormap(('red','green'))(i),label=j)
    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#Visualising the Test set results

from matplotlib.colors import ListedColormap

x_set, y_set = x_test, y_test

x1 , x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1, stop = x_set[:,0].max() +1, step = 0.01), np.arange(start=x_set[:,1].min()-1, stop=x_set[:,1].max()+1, step = 0.01))


plt.contourf(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),alpha = 0.75, cmap=ListedColormap(('red','green')))
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(), x2.max())


for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0], x_set[y_set==j,1], c=ListedColormap(('red','green'))(i),label=j)
    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
