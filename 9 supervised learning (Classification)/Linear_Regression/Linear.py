from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from google.colab import files
import io
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


#for test on colab!!!
uploaded = files.upload()


ds = pd.read_csv(io.BytesIO(uploaded['insurance.csv']))
ds.head()

df = pd.DataFrame(ds)


label_encoder_x= LabelEncoder()
df['sex']= label_encoder_x.fit_transform(df['sex'])
df['smoker']= label_encoder_x.fit_transform(df['smoker'])
df['region']= label_encoder_x.fit_transform(df['region'])

df.drop_duplicates(inplace = True)


z = pd.DataFrame(df,columns=['age','sex','bmi','children','smoker','region'])
x, y = z, df['expenses']
x_train, x_test ,y_train, y_test = train_test_split(x, y, test_size = 0.2,
random_state=42)
reg=LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)


r_squared2 =r2_score(y_test, y_pred)
print("R2=",r_squared2)
