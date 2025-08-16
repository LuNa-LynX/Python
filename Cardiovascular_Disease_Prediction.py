import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv(r"C:\Users\Swathi Krishna\Downloads\cardio_train.csv",sep=';')
df.head()
df.drop(columns=['id'], inplace=True) 
df.drop_duplicates(inplace=True)
df.fillna(method='ffill', inplace=True)


df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)




plt.figure(figsize=(8, 4))
sns.histplot(df['age'] / 365, bins=30, kde=True, color='salmon')
plt.title('Age Distribution (in years)')
plt.xlabel('Age')
plt.show()


plt.figure(figsize=(6, 4))
sns.countplot(x='cholesterol', hue='cardio', data=df, palette='Set2')
plt.title('Cholesterol Levels vs Cardiovascular Disease')
plt.show()


plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.show()


X = df.drop(columns=['cardio'])
y = df['cardio']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


models = {
    'Logistic Regression': LogisticRegression(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier()
}

accuracy_results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracy_results[name] = acc
    print(f"\n {name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))


plt.figure(figsize=(8, 5))
sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()), palette='viridis')
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0.7, 1.0)
plt.show()


final_model = RandomForestClassifier()
final_model.fit(X_train, y_train)
final_pred = final_model.predict(X_test)

print("\n Final Model: Random Forest")
print("Accuracy:", accuracy_score(y_test, final_pred))
