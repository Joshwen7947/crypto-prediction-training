import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("ETH-USD.csv")

projection = 7
df['Prediction'] = df[['Close']].shift(-projection)

# print(df)

x = np.array(df[['close']])
x = x[:-projection]


y = df['Prediction'].values
y = y[:-projection]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.15)

linReg = LinearRegression()

linReg.fit(x_train,y_train)




