import the necessary library and the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
​
diabetes=pd.read_csv('diabetes.csv')
print(diabetes.columns)
Index(['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'],
      dtype='object')
diabetes.head()
Pregnancies	Glucose	BloodPressure	SkinThickness	Insulin	BMI	DiabetesPedigreeFunction	Age	Outcome
0	6	148	72	35	0	33.6	0.627	50	1
1	1	85	66	29	0	26.6	0.351	31	0
2	8	183	64	0	0	23.3	0.672	32	1
3	1	89	66	23	94	28.1	0.167	21	0
4	0	137	40	35	168	43.1	2.288	33	1
print("dimension  of the data: {}".format(diabetes.shape))
dimension  of the data: (768, 9)
Basic Data Analysis
#grouping data based on outcome
print(diabetes.groupby('Outcome').size())
Outcome
0    500
1    268
dtype: int64
import seaborn as sns
sns.countplot(x=diabetes['Outcome'], label="Count")
<Axes: xlabel='Outcome', ylabel='count'>

# some information of our data
diabetes.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
K-Nearest Neighbours to Predict Diabetes
from sklearn.model_selection import train_test_split
​
x_train, x_test, y_train, y_test=train_test_split(diabetes.loc[:,diabetes.columns!='Outcome'], diabetes['Outcome'], random_state=80, train_size=.6)
​
from sklearn.neighbors import KNeighborsClassifier
​
train_accuracy=[]
test_accuracy=[]
​
nbd=range(1,15)
​
for n_nbd in nbd:
    #build the model
    knn=KNeighborsClassifier(n_neighbors=n_nbd)
    knn.fit(x_train, y_train)
    
    #record the accuracy 
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))
    
plt.plot(nbd, train_accuracy, label="train accuracy")
plt.plot(nbd, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("nbd")
​
plt.legend()
<matplotlib.legend.Legend at 0x1831a69cfd0>

knn=KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test, y_test))
0.782608695652174
0.7337662337662337
Decision Tree Classifier to Predict Diabetes
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(random_state=0)
​
tree.fit(x_train, y_train)
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))
1.0
0.6883116883116883
tree=DecisionTreeClassifier(max_depth=3, random_state=0)
tree.fit(x_train, y_train)
​
print(tree.score(x_train, y_train))
print(tree.score(x_test, y_test))
0.7913043478260869
0.7532467532467533
Feature Importance in Decision Trees
print(tree.feature_importances_)
[0.04598683 0.64752304 0.         0.         0.         0.24454402
 0.06194612 0.        ]
diabetes_features=diabetes.loc[:,diabetes.columns!='Outcome']
def plot_FI(model):
    plt.figure(figsize=(8,6))
    features=8
    plt.barh(range(features),model.feature_importances_)
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.yticks(np.arange(features), diabetes_features)
    
    plt.ylim(-1,features)
​
plot_FI(tree)

Deep Learning to Predict Diabetes
from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier(random_state=45)
​
mlp.fit(x_train, y_train)
​
print(mlp.score(x_train, y_train))
print(mlp.score(x_test, y_test))
0.7717391304347826
0.7045454545454546
from sklearn.preprocessing import StandardScaler
scale=StandardScaler()
x_train_scale=scale.fit_transform(x_train)
x_test_scale=scale.fit_transform(x_test)
​
mlp=MLPClassifier(random_state=0)
​
mlp.fit(x_train_scale, y_train)
​
print(mlp.score(x_train_scale, y_train))
print(mlp.score(x_test_scale, y_test))
0.8152173913043478
0.775974025974026
C:\Users\Hp\anaconda3\Lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:691: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  warnings.warn(
plt.figure(figsize=(20,20))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
​
plt.yticks(range(8),diabetes_features)
plt.xlabel("Weight matrix")
plt.ylabel("Features")
​
plt.colorbar()
<matplotlib.colorbar.Colorbar at 0x267a70902d0>

Setting Up the GUI with Tkinter
Import Libraries for GUI and Machine Learning:
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
Load and Prepare Data:
# Load the dataset
diabetes = pd.read_csv('diabetes.csv')
​
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    diabetes.drop(columns='Outcome'), diabetes['Outcome'], random_state=80, train_size=0.6
)
​
# Standardize the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
​
# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train_scaled, y_train)
​
KNeighborsClassifier(n_neighbors=11)
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
Create the GUI:
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
​
# Initialize the Tkinter app
root = tk.Tk()
root.title("Diabetes Prediction App")
root.geometry("420x550")  # Set the window size
root.configure(bg="#cbe8f7")  # Light sky blue background
​
# Function to create a gradient background
def create_gradient(canvas, width, height, color1, color2):
    for i in range(height):
        ratio = i / height
        new_color = f"#{int(ratio * int(color2[1:3], 16) + (1 - ratio) * int(color1[1:3], 16)):02x}" \
                    f"{int(ratio * int(color2[3:5], 16) + (1 - ratio) * int(color1[3:5], 16)):02x}" \
                    f"{int(ratio * int(color2[5:7], 16) + (1 - ratio) * int(color1[5:7], 16)):02x}"
        canvas.create_line(0, i, width, i, fill=new_color)
​
# Create a Canvas for the gradient background
canvas = tk.Canvas(root, width=420, height=550, highlightthickness=0)
canvas.pack(fill="both", expand=True)
create_gradient(canvas, 420, 550, "#cbe8f7", "#ffffff")
​
# Frame for input fields with a shadow effect
input_frame = tk.Frame(root, bg="#ffffff", padx=20, pady=20, relief="raised", borderwidth=2)
input_frame.place(relx=0.5, rely=0.5, anchor="center")
​
# Title Label with custom font and shadow effect
title_label = tk.Label(
    input_frame, text="Diabetes Prediction", font=("Verdana", 16, "bold"), bg="#ffffff", fg="#004080"
)
title_label.grid(row=0, columnspan=2, pady=(0, 20))
​
# Input labels and entry fields with a custom look
labels = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
          'Insulin', 'BMI', 'Diabetes Pedigree Function', 'Age']
entries = []
​
for i, label in enumerate(labels):
    tk.Label(input_frame, text=label, font=("Verdana", 10, "bold"), bg="#ffffff", fg="#0073e6").grid(
        row=i + 1, column=0, padx=10, pady=5, sticky="w"
    )
    entry = tk.Entry(input_frame, font=("Verdana", 10), bg="#f0f8ff", fg="#333333", relief="flat", borderwidth=1)
    entry.grid(row=i + 1, column=1, padx=10, pady=5, ipadx=5, ipady=3)
    entries.append(entry)
​
# Function for prediction
def predict_diabetes():
    try:
        # Gather input data from the user
        input_data = [float(entry.get()) for entry in entries]
        input_data_scaled = scaler.transform([input_data])
​
        # Make a prediction
        prediction = knn.predict(input_data_scaled)
        outcome = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
​
        # Show the result
        messagebox.showinfo("Prediction", f"The patient is likely: {outcome}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")
​
# Stylish Prediction Button with hover effect
def on_hover(event):
    predict_button.config(bg="#ff944d", fg="white")
​
def on_leave(event):
    predict_button.config(bg="#ff6600", fg="white")
​
predict_button = tk.Button(
    root, text="Predict", command=predict_diabetes, bg="#ff6600", fg="white",
    font=("Verdana", 12, "bold"), relief="raised", bd=3, padx=15, pady=5
)
predict_button.place(relx=0.5, y=480, anchor="center")
​
# Bind hover effects to the button
predict_button.bind("<Enter>", on_hover)
predict_button.bind("<Leave>", on_leave)
​
# Run the app
root.mainloop()
​
C:\Users\Hp\anaconda3\Lib\site-packages\sklearn\base.py:464: UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
  warnings.warn(
Comprehensive Tkinter GUI for Machine Learning Workflow
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
​
# Initialize the main application window
root = tk.Tk()
root.title("Machine Learning Workflow")
root.geometry("700x500")  # Set a larger window size for better layout
root.configure(bg="#e6f7ff")  # Light blue background
​
# Custom Styles
style = ttk.Style()
style.configure("TNotebook", background="#ffffff", padding=5)
style.configure("TNotebook.Tab", font=("Helvetica", 10, "bold"), background="#cce7ff", padding=10)
style.map("TNotebook.Tab", background=[("selected", "#80bfff")])
​
# Using a Notebook widget to create tabs
notebook = ttk.Notebook(root)
notebook.pack(pady=20, expand=True)
​
# Global variables
dataset = None
x_train, x_test, y_train, y_test = None, None, None, None
model = None
scaler = None
​
# --- Module 1: Dataset Selection ---
def load_dataset():
    global dataset
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        dataset = pd.read_csv(file_path)
        messagebox.showinfo("Dataset Loaded", "Dataset loaded successfully!")
        print(dataset.head())
​
frame1 = ttk.Frame(notebook, padding=20)
notebook.add(frame1, text="1. Dataset Selection")
load_button = tk.Button(frame1, text="Load Dataset", command=load_dataset, bg="#4CAF50", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
load_button.pack(pady=20)
​
# --- Module 2: Dataset Preprocessing ---
def preprocess_data():
    global x_train, x_test, y_train, y_test, scaler
    if dataset is not None:
        x = dataset.drop(columns='Outcome')
        y = dataset['Outcome']
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=80, train_size=0.6)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        messagebox.showinfo("Preprocessing", "Data preprocessed successfully!")
    else:
        messagebox.showerror("Error", "Please load a dataset first.")
​
frame2 = ttk.Frame(notebook, padding=20)
notebook.add(frame2, text="2. Data Preprocessing")
preprocess_button = tk.Button(frame2, text="Preprocess Data", command=preprocess_data, bg="#FF5733", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
preprocess_button.pack(pady=20)
​
# --- Module 3: Data Visualization ---
def visualize_data():
    if dataset is not None:
        sns.pairplot(dataset, hue='Outcome')
        plt.show()
    else:
        messagebox.showerror("Error", "Please load a dataset first.")
​
frame3 = ttk.Frame(notebook, padding=20)
notebook.add(frame3, text="3. Data Visualization")
visualize_button = tk.Button(frame3, text="Visualize Data", command=visualize_data, bg="#0073e6", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
visualize_button.pack(pady=20)
​
# --- Module 4: Feature Engineering ---
def feature_engineering():
    if dataset is not None:
        # Example of feature engineering: adding a new feature (e.g., BMI category)
        dataset['BMI_Category'] = pd.cut(dataset['BMI'], bins=[0, 18.5, 25, 30, np.inf], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
        messagebox.showinfo("Feature Engineering", "Feature Engineering completed!")
        print(dataset.head())
    else:
        messagebox.showerror("Error", "Please load a dataset first.")
​
frame4 = ttk.Frame(notebook, padding=20)
notebook.add(frame4, text="4. Feature Engineering")
feature_engineering_button = tk.Button(frame4, text="Apply Feature Engineering", command=feature_engineering, bg="#9C27B0", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
feature_engineering_button.pack(pady=20)
​
# --- Module 5: Model Training ---
def train_model():
    global model
    if x_train is not None:
        model = KNeighborsClassifier(n_neighbors=11)
        model.fit(x_train, y_train)
        messagebox.showinfo("Model Training", "Model trained successfully!")
    else:
        messagebox.showerror("Error", "Please preprocess the data first.")
​
frame5 = ttk.Frame(notebook, padding=20)
notebook.add(frame5, text="5. Model Training")
train_button = tk.Button(frame5, text="Train Model", command=train_model, bg="#2196F3", fg="white", font=("Helvetica", 12, "bold"), relief="raised")
train_button.pack(pady=20)
​
# --- Module 6: Model Testing ---
def test_model():
    if model is not None:
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        messagebox.showinfo("Model Testing", f"Model Accuracy: {accuracy:.2f}")
    else:
        messagebox.showerror("Error", "Please train the model first.")
​
frame6 = ttk.Frame(notebook, padding=20)
notebook.add(frame6, text="6. Model Testing")
test_button = tk.Button(frame6, text="Test Model", command=test_model, bg="#FFC107", fg="black", font=("Helvetica", 12, "bold"), relief="raised")
test_button.pack(pady=20)
​
# --- Module 7: Performance Evaluation ---
def evaluate_performance():
    if model is not None:
        predictions = model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        performance_text = f"Model Accuracy: {accuracy:.2f}\n"
        messagebox.showinfo("Performance Evaluation", performance_text)
    else:
        messagebox.showerror("Error", "Please train the model first.")
​
frame7 = ttk.Frame(notebook, padding=20)
notebook.add(frame7, text="7. Performance Evaluation")
evaluate_button = tk.Button(frame7, text="Evaluate Performance", command=evaluate_performance, bg="#8BC34A", fg="black", font=("Helvetica", 12, "bold"), relief="raised")
evaluate_button.pack(pady=20)
​
# Run the app
root.mainloop()
​
​
