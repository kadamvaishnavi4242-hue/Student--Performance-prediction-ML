import pandas as pd
from sklearn.preprocessing import LabelEncoder
df=pd.read_csv(r"C:\Users\kadam\Downloads\student succes predictor.csv",encoding="latin1")
print("sample data")
print(df.head())
print("structure of data")
print(f"rows:{df.shape[0]},columns:{df.shape[1]}")
print("dataset info")
print(df.info())
print("summary statactics")
print(df.describe())
print("missing values")
print(df.isnull().sum())
le=LabelEncoder()
df["internet"]=le.fit_transform(df["internet"])
df["passes"]=le.fit_transform(df["passes"])
print("after encoding")
print(df.head())
print("data type")
print(df.dtypes)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import seaborn as sns
features=["studyhours","attendence","pastscore","sleephours"]
scaler=StandardScaler()
df_scaled=df.copy()
df_scaled[features]=scaler.fit_transform(df[features])
X=df_scaled[features]
y=df_scaled["passes"]
print(y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42,test_size=0.2)
model=LogisticRegression()
print("y_train shape",y_train.shape)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
print("classification report")
print(classification_report(y_test,y_pred))
confu_matrix=confusion_matrix(y_test,y_pred)
print(confu_matrix)
sns.heatmap(confu_matrix,annot=True,fmt="d",cmap="Blues",xticklabels=["fail","pass"],yticklabels=["fail"])
plt.xlabel("predicted")
plt.ylabel("actual")
plt.title("confusion matrix")
plt.tight_layout()
plt.show()
print("predict your result")
study_hours=float(input("enter your studyhours"))
Attendence=float(input("enter your Attendence"))
past_score=float(input("enter your pastscore"))
sleep_hour=float(input("enter your sleephour"))
user_input_df=pd.DataFrame([{
    "studyhours":study_hours,
    "attendence":Attendence,
    "pastscore":past_score,
    "sleephours":sleep_hour
}])
user_input_scaled=scaler.transform(user_input_df)
prediction= model.predict(user_input_scaled)[0]
result="pass" if prediction==1 else "fail"
print(f"based on your input{result}")
