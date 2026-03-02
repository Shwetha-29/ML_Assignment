# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# ==============================
# LOAD DATASET
# ==============================
data = pd.read_csv("Vehicle_Maintenance _Prediction.csv")

data = data.drop(["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

le = LabelEncoder()
data["Type"] = le.fit_transform(data["Type"])

X = data[[
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]]

y = data["Machine failure"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))


# ==============================
# PREDICTION FUNCTION
# ==============================
def predict_failure():
    try:
        type_input = type_entry.get().strip().upper()

        if type_input not in ["L", "M", "H"]:
            result_label.config(text="Enter Type as L, M, or H", fg="red")
            return

        type_val = le.transform([type_input])[0]

        air_temp = float(air_temp_entry.get())
        process_temp = float(process_temp_entry.get())
        speed = float(speed_entry.get())
        torque = float(torque_entry.get())
        wear = float(wear_entry.get())

        input_data = pd.DataFrame([[
            type_val,
            air_temp,
            process_temp,
            speed,
            torque,
            wear
        ]], columns=X.columns)

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        if prediction == 1:
            result_label.config(
                text=f"⚠ Maintenance Required\nProbability: {probability:.2f}",
                fg="red"
            )
        else:
            result_label.config(
                text=f"✅ No Maintenance Required\nProbability: {probability:.2f}",
                fg="green"
            )

    except:
        result_label.config(text="Please enter numeric values correctly", fg="red")


# ==============================
# CREATE UI
# ==============================
root = tk.Tk()
root.title("Vehicle Maintenance Prediction")
root.geometry("500x600")
root.configure(bg="#f4f6f7")

title = tk.Label(root, text="Vehicle Maintenance Prediction",
                 font=("Arial", 20, "bold"),
                 bg="#f4f6f7")
title.pack(pady=20)

label_font = ("Arial", 14)

tk.Label(root, text="Machine Type (L/M/H)", font=label_font, bg="#f4f6f7").pack()
type_entry = tk.Entry(root, font=label_font, width=20)
type_entry.pack(pady=5)

tk.Label(root, text="Air Temperature [K]", font=label_font, bg="#f4f6f7").pack()
air_temp_entry = tk.Entry(root, font=label_font, width=20)
air_temp_entry.pack(pady=5)

tk.Label(root, text="Process Temperature [K]", font=label_font, bg="#f4f6f7").pack()
process_temp_entry = tk.Entry(root, font=label_font, width=20)
process_temp_entry.pack(pady=5)

tk.Label(root, text="Rotational Speed [rpm]", font=label_font, bg="#f4f6f7").pack()
speed_entry = tk.Entry(root, font=label_font, width=20)
speed_entry.pack(pady=5)

tk.Label(root, text="Torque [Nm]", font=label_font, bg="#f4f6f7").pack()
torque_entry = tk.Entry(root, font=label_font, width=20)
torque_entry.pack(pady=5)

tk.Label(root, text="Tool Wear [min]", font=label_font, bg="#f4f6f7").pack()
wear_entry = tk.Entry(root, font=label_font, width=20)
wear_entry.pack(pady=5)

predict_button = tk.Button(root,
                           text="Predict",
                           font=("Arial", 16, "bold"),
                           bg="#2e86c1",
                           fg="white",
                           width=15,
                           command=predict_failure)
predict_button.pack(pady=20)

result_label = tk.Label(root,
                        text="",
                        font=("Arial", 16, "bold"),
                        bg="#f4f6f7")
result_label.pack(pady=20)

root.mainloop()