from tkinter import *
import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from imblearn.combine import SMOTEENN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense

global filename
global df, X_train, X_test, y_train, y_test
global ada_acc, rf_acc, mlp_acc, lstm_acc, gru_acc, smote_enn_acc

def upload():
    global filename, df
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    df = pd.read_csv(filename)
    
    # Replace '?' with NaN
    df.replace('?', np.nan, inplace=True)

    # Fill missing values with mode for each column
    df.fillna(df.mode().iloc[0], inplace=True)
    
    text.delete('1.0', END)
    text.insert(END, 'Dataset loaded\n')
    text.insert(END, "Dataset Size: " + str(len(df)) + "\n")

def splitdataset(): 
    global df, X_train, X_test, y_train, y_test

    # Encode string columns to numerical values
    label_encoder = LabelEncoder()
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = label_encoder.fit_transform(df[column])

    X = np.array(df.drop(["Class"], axis=1))
    y = np.array(df["Class"])
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)    
    # Display dataset split information
    text.delete('1.0', END)
    text.insert(END, "Dataset split\n")
    text.insert(END, "Splitted Training Size for Machine Learning : " + str(len(X_train)) + "\n")
    text.insert(END, "Splitted Test Size for Machine Learning    : " + str(len(X_test)) + "\n")
    
    # Display shapes of X_train, X_test, y_train, y_test
    text.insert(END, "\nShape of X_train: " + str(X_train.shape) + "\n")
    text.insert(END, "Shape of X_test: " + str(X_test.shape) + "\n")
    text.insert(END, "Shape of y_train: " + str(y_train.shape) + "\n")
    text.insert(END, "Shape of y_test: " + str(y_test.shape) + "\n\n")

def adaboost():
    global ada_acc
    ada = AdaBoostClassifier(n_estimators=100, random_state=0)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    ada_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for AdaBoost is {ada_acc * 100}%\n'
    text.insert(END, result_text)

def random_forest():
    global rf_acc
    rf = RandomForestClassifier(n_estimators=100, random_state=0)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for Random Forest is {rf_acc * 100}%\n'
    text.insert(END, result_text)


def mlp():
    global mlp_acc
    mlp = MLPClassifier()
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mlp_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for MLP is {mlp_acc * 100}%\n'
    text.insert(END, result_text)

def lstm():
    global lstm_acc
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    _, lstm_acc = model.evaluate(X_test, y_test)
    result_text = f'Accuracy for LSTM is {lstm_acc * 100}%\n'
    text.insert(END, result_text)

def gru():
    global gru_acc
    model = Sequential()
    model.add(GRU(units=50, input_shape=(X_train.shape[1], 1)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    _, gru_acc = model.evaluate(X_test, y_test)
    result_text = f'Accuracy for GRU is {gru_acc * 100}%\n'
    text.insert(END, result_text)

def smote_enn():
    global smote_enn_acc,mlp
    smote_enn = SMOTEENN()
    X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
    mlp = MLPClassifier()
    mlp.fit(X_resampled, y_resampled)
    y_pred = mlp.predict(X_test)
    smote_enn_acc = accuracy_score(y_test, y_pred)
    result_text = f'Accuracy for SMOTE-ENN is {smote_enn_acc * 100}%\n'
    text.insert(END, result_text)

def plot_bar_graph():
    algorithms = ['AdaBoost', 'Random Forest', 'MLP', 'LSTM', 'GRU', 'SMOTE-ENN']
    accuracies = [ada_acc * 100, rf_acc * 100, mlp_acc * 100, lstm_acc * 100, gru_acc * 100, smote_enn_acc * 100]
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'cyan']
    
    plt.bar(algorithms, accuracies, color=colors)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Machine Learning Algorithms')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def predict():
    # Open file manager to select CSV file
    filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if filename:
        # Read the selected CSV file
        input_data = pd.read_csv(filename)

        # Fill missing values with mode for each column (assuming similar preprocessing as done in other functions)
        input_data.fillna(input_data.mode().iloc[0], inplace=True)

        # Preprocess input data (if needed)
        label_encoder = LabelEncoder()
        for column in input_data.columns:
            if input_data[column].dtype == 'object':
                input_data[column] = label_encoder.fit_transform(input_data[column])

        # Perform prediction using MLP model
        y_pred = mlp.predict(input_data)

        # Display the prediction result
        if y_pred[0] == 1:
            messagebox.showinfo("Prediction Result: ", "Fraudulent Transaction Detected")
        else:
            messagebox.showinfo("Prediction Result: ", "Non-Fraudulent Transaction Detected")


main = tk.Tk()
main.title("A DEEP LEARNING ENSEMBLE WITH DATA RESAMPLING FOR CREDIT CARDS FRAUD DETECTION") 
main.geometry("1600x900")

font = ('times', 16, 'bold')
title = tk.Label(main, text='A DEEP LEARNING ENSEMBLE WITH DATA RESAMPLING FOR CREDIT CARDS FRAUD DETECTION',font=("times"))
title.config(bg='Dark Blue', fg='white')
title.config(font=font)           
title.config(height=3, width=145)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = tk.Text(main, height=20, width=180)
scroll = tk.Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50, y=120)
text.config(font=font1)

font1 = ('times', 13, 'bold')
button_bg_color = "lightgrey"
button_fg_color = "black"
button_hover_bg_color = "grey"
button_hover_fg_color = "white"
bg_color = "#32d1a7"  # Light blue-green background color

# Define button configurations
button_config = {
    "bg": button_bg_color,
    "fg": button_fg_color,
    "activebackground": button_hover_bg_color,
    "activeforeground": button_hover_fg_color,
    "width": 15,
    "font": font1
}

uploadButton = tk.Button(main, text="Upload Dataset", command=upload, **button_config)
pathlabel = tk.Label(main)
splitButton = tk.Button(main, text="Split Dataset", command=splitdataset, **button_config)
adaboostButton = tk.Button(main, text="AdaBoost", command=adaboost, **button_config)
rfButton = tk.Button(main, text="Random Forest", command=random_forest, **button_config)
mlpButton = tk.Button(main, text="MLP", command=mlp, **button_config)
lstmButton = tk.Button(main, text="LSTM", command=lstm, **button_config)
gruButton = tk.Button(main, text="GRU", command=gru, **button_config)
smote_ennButton = tk.Button(main, text="SMOTE-ENN", command=smote_enn, **button_config)
plotButton = tk.Button(main, text="Plot Results", command=plot_bar_graph, **button_config)
predict_button = tk.Button(main, text="Prediction", command=predict, **button_config)

uploadButton.place(x=50, y=600)
pathlabel.config(bg='DarkOrange1', fg='white', font=font1)  
pathlabel.place(x=250, y=600)
splitButton.place(x=450, y=600)
adaboostButton.place(x=50, y=650)
rfButton.place(x=250, y=650)
mlpButton.place(x=450, y=650)
lstmButton.place(x=650, y=650)
gruButton.place(x=850, y=650)
smote_ennButton.place(x=1050, y=650)
plotButton.place(x=50, y=700)
predict_button.place(x=250, y=700)

main.config(bg=bg_color)
main.mainloop()
