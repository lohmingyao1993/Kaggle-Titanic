from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from tensorflow.keras.layers import Input,Dense,Activation,Dropout
from tensorflow.keras.models import Model

data=pd.read_csv("C:/Users/mingyao/Google Drive/Coursera/Projects/Titanic/train.csv")

# Female=0, Male=1
data['Sex'].replace('female',0,inplace=True)
data['Sex'].replace('male',1,inplace=True)


# sort age group
age_bins=[0,10,18,50,100]
age_labels=[0,1,2,3]
data['AgeGroup']=pd.cut(data['Age'],bins=age_bins,labels=age_labels,right=False)

data=data.drop(columns=['Name','Cabin','Fare','Ticket','Embarked','AgeGroup','PassengerId','SibSp','Parch'])
data=data.dropna()


def dummy_data(data, columns):
    for column in columns:
        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)
        data = data.drop(column, axis=1)
    return data


dummy_columns = ["Pclass"]
X=dummy_data(data, dummy_columns)
#test_data=dummy_data(test_data, dummy_columns)


from sklearn.preprocessing import MinMaxScaler

def normalize_age(data):
    scaler = MinMaxScaler()
    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))
    return data
X = normalize_age(X)  
#test_data = normalize_age(test_data)
#train_data.head()





y=data[['Survived']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# create model
input_layer = Input(shape=(X_train.shape[1],))
dense_layer_1 = Dense(100, activation='sigmoid')(input_layer)
dense_layer_2 = Dense(10, activation='sigmoid')(dense_layer_1)
output = Dense(y.shape[1], activation='softmax')(dense_layer_2)


model = Model(inputs=input_layer, outputs=output)

opt = keras.optimizers.Adam(lr=0.0001)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])

print(model.summary())


history = model.fit(X_train, y_train, batch_size=64, epochs=5, verbose=1, shuffle=True)


score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])
