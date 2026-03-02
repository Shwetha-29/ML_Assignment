# ==============================
# IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import tkinter as tk
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ==============================
# LOAD DATASET
# ==============================
data = pd.read_csv("Vehicle_Maintenance _Prediction.csv")

data = data.drop(["UDI", "Product ID", "Machine failure",
                  "TWF", "HDF", "PWF", "OSF", "RNF"], axis=1)

# Encode Type
le = LabelEncoder()
data["Type"] = le.fit_transform(data["Type"])

X = data

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==============================
# APPLY PCA
# ==============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# ==============================
# PREDICTION FUNCTION
# ==============================
def transform_input():
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
        input_pca = pca.transform(input_scaled)

        pc1 = input_pca[0][0]
        pc2 = input_pca[0][1]

        result_label.config(
            text=f"PC1: {pc1:.2f}\nPC2: {pc2:.2f}",
            fg="blue"
        )

    except:
        result_label.config(text="Enter numeric values correctly", fg="red")


# ==============================
# UI
# ==============================
root = tk.Tk()
root.title("PCA - Machine Feature Reduction")
root.geometry("500x600")
root.configure(bg="#f4f6f7")

title = tk.Label(root,
                 text="Principal Component Analysis",
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

tk.Button(root,
          text="Transform to PCA",
          font=("Arial", 16, "bold"),
          bg="#e67e22",
          fg="white",
          width=18,
          command=transform_input).pack(pady=20)

result_label = tk.Label(root,
                        text="",
                        font=("Arial", 16, "bold"),
                        bg="#f4f6f7")
result_label.pack(pady=20)

root.mainloop()