
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
train_path ="C:\\Users\\Kshore N\\Downloads\\train (1).csv"
test_path ="C:\\Users\\Kshore N\\Downloads\\test (1).csv"
train_df=pd.read_csv(train_path)
test_df=pd.read_csv(test_path)
print(train_df.head())
print(test_df.head())
def extract_titles(name : str) -> str:
    if pd.isna(name):
        return "Unknown"
    return name.split(",")[1].split(".")[0].strip()
def feature_engineering(df :pd.DataFrame) -> pd.DataFrame:
    df=df.copy()
    df['Title']=df['Name'].apply(extract_titles)
    rare_titles=['Lady',"Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
    df["Title"]=df["Title"].replace({
        "Mlle":"Miss",
        "Mme":"Mrs",
        "Ms" :"Miss"

    })
    df['Title'] =df['Title'].replace(rare_titles, "Rare")
    df["FamilySize"]=df['SibSp'] +df['Parch']+1
    df['Alone']=0
    df.loc[df['FamilySize']==1,'Alone']=1
    df.drop(columns=["Ticket","Cabin","Name"],errors='ignore')
    return df
train_fe=feature_engineering(train_df)
test_fe=feature_engineering(test_df)
target_col='Survived'
id_col='PassengerId'
x=train_fe.drop(columns=[target_col])
y=train_fe[target_col]
x_test_final=test_fe.copy()
numerical_features =["Age","Fare","SibSp",'Parch',"FamilySize","Alone"]
categorical_features =["Pclass","Sex","Embarked","Title"]
numerical_pipeline =Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
categorical_pipeline =Pipeline(steps=[
    ("imputer",SimpleImputer(strategy="constant")),
    ("onehot",OneHotEncoder(handle_unknown="ignore"))
])
preprocessor=ColumnTransformer(
    transformers=[
        ("numerical",numerical_pipeline,numerical_features),
        ("categorical",categorical_pipeline,categorical_features)
    ]
)
clf=RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=1
)
model =Pipeline(steps=[
    ("preprocessor",preprocessor),
    ("classifier",clf)
])
scores =cross_val_score(model, x, y ,cv =5,n_jobs=1,verbose=2,scoring='accuracy')
print(f"accuracy score : {scores.mean() *100:.2f}%  || {scores.std()*100:.2f}%")
param_grid={
    "classifier__n_estimators":[100,200,300],
    "classifier__max_depth":[None,5,10,15],
    "classifier__min_samples_split":[2,4,6],
    "classifier__min_samples_leaf":[1,2,3]
}
grid_search=GridSearchCV(model,param_grid,cv=5,scoring='accuracy',n_jobs=1,verbose=2)
grid_search.fit(x,y)
print(f" best  cv scores :{grid_search.best_score_}")
print(f" best parameters :{grid_search.best_params_}")
best_model = grid_search.best_estimator_
final_model =best_model
predictions =final_model.predict(x_test_final)
submission= pd.DataFrame({
    "PassengerId" :x_test_final[id_col],
    "Survived":predictions.astype(int)
})
submission_path ="submission.csv"
print(f" submission file is saved at {submission_path}")
print(submission.head())







