import pandas as pd
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\ASHISH KUMAR\OneDrive\Desktop\Python\Dataset\Iris code\Iris.csv")
df.head()
# delete a column
df = df.drop(columns = ['Id'])
df.head()

# to display stats about data
df.describe()

# to display no. of samples on each class
df['Species'].value_counts()

# check for null values
df.isnull().sum()

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()

# df['Species'] = le.fit_transform(df['Species'])
# df.head()

from sklearn.model_selection import train_test_split
# train - 70
# test - 30
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


model.fit(x_train, y_train)

# print metric to get performance
print("Accuracy: ",model.score(x_test, y_test))

# save the model
import pickle
filename = 'savedmodel.sav'
pickle.dump(model, open(filename, 'wb'))

load_model = pickle.load(open(filename,'rb'))

x= load_model.predict([[7.6, 4.5, 1.3, 5.2]])
print(x)