import pandas as pd

df = pd.read_csv(r"C:\Users\mrfau\Downloads\Student_EDA_Practice_Dataset - Student_EDA_Practice_Dataset.csv")
# print(type(df))
print("////// EDA PROCESSES //////")

print("-------for first 5 rows-------")
print(df.head())
print(" ")

print("-------for last 5 rows-------")
print(df.tail())
print(" ")

print("-----column and datatype check------")
print(df.info())
print(" ")

print("------shape check------")
print(df.shape)
print(" ")

print("------statistical analysis------")
print(df.describe())
print(" ")

print("------missing values------")
print(df.isnull())
print(" ")

print(df.isnull().sum())
print(" ")

print(df.isna())
print(" ")

