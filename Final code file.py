import pandas as pd  
import numpy as np  
import seaborn as s  
import matplotlib.pyplot as plt  
import sklearn
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.metrics import ( accuracy_score,confusion_matrix,classification_report,roc_curve,auc, precision_recall_curve )


pk_dataset = pd.read_csv(r"C:\Users\nakul\OneDrive\Desktop\ACM ML PROJECT\RAW DATA\archive\Pokemon.csv")  
pk_working = pk_dataset.copy()  
pk_working["Mega_Evolution"] = pk_working["Name"].apply(lambda x: 1 if "Mega" in x else 0)

#To remove collums that are not needed for model training from the working data set
pk_working = pk_working.drop(columns=["Name", "Type 1", "Type 2", "Generation", "Legendary"])  


#To Define Features and Target
X = pk_working.drop(columns=["Mega_Evolution"])
y = pk_working["Mega_Evolution"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Training and prediction
algomodel = RandomForestClassifier(n_estimators=100, random_state=777)
algomodel.fit(X_train, y_train)
y_prediction = algomodel.predict(X_test)
y_scores = algomodel.predict_proba(X_test)[:, 1]  # Probability of class 1 (Mega)

#Plotting of Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_prediction)
plt.figure(figsize=(10,5))
s.heatmap(conf_matrix, annot=True, fmt="d", cmap="Purples", xticklabels=["Normal", "Mega"], yticklabels=["Normal", "Mega"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

#ROC curve computation
fp_rate, tp_rate,_ = roc_curve(y_test, y_scores)
area_roc = auc(fp_rate, tp_rate)

#Plotting of ROC curve
plt.figure(figsize=(10, 5))
plt.plot(fp_rate, tp_rate, color='purple', label=f'ROC Curve (AUC = {area_roc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

accuracy = accuracy_score(y_test, y_prediction)
print(f"Accuracy of Model: {accuracy:.2f}")

#Prediction of Mega Evolution(full dataset)
pk_dataset["Mega_Evolution"] = algomodel.predict(pk_working.drop(columns=["Mega_Evolution"])) 
pk_dataset["Mega_Evolution"] = pk_dataset["Mega_Evolution"].apply(lambda x: "Yes" if x == 1 else "No")

#CSV Output
output_file_path = r"C:\Users\nakul\OneDrive\Desktop\ACM ML PROJECT\RAW DATA\archive\pokemon_mega_predictions.csv"

pk_dataset[["Name", "Mega_Evolution"]].to_csv(output_file_path, index=False)

print(f"Otput saved at : {output_file_path}")

#Plotting of Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
plt.plot(recall, precision,color = "purple" ,label='Precision Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.legend()
plt.show()