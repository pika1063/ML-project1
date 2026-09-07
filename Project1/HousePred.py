import pandas as pd
df = pd.read_csv(r"C:\Users\mrfau\Downloads\House Price Prediction Dataset - House Price Prediction Dataset.csv")

print(df.head())

print(df.isnull().sum())

print(df.shape)
df["Condition"]=df["Condition"].map({
  "Excellent":5,
  "Good":4,
  "Fair":2,
  "Poor":1

})
x=df[["Area","Bedrooms","Floors","YearBuilt","Condition"]]
y=df["Price"]
print("--------x shape----------")
print(x.shape)
print()
print("--------y shape----------")
print(y.shape)
print()
print("------train and test split-------")

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42
)

print("---- x_test ----")
print(x_test)

print()

print("---- x_train ----")
print(x_train)

print()

print("---- y_test ----")
print(y_test)

print()

print("---- y_train ----")
print(y_train)

print()

print("-------linear regression------")
from sklearn.linear_model import LinearRegression
model=LinearRegression()
print(model)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("-----y_pred-----")
print(y_pred)
print()
print("-----r2_score-----")
from sklearn.metrics import r2_score, mean_absolute_error
r2_score = r2_score(y_test,y_pred)
print(r2_score)
print()
print("-----mean_absolute_error-----")
mae=mean_absolute_error(y_test,y_pred)
print(mae)