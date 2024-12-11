# -*- coding: utf-8 -*-
"""
Created on Thu May 30 10:01:08 2024

@author: ASUS
"""

import tkinter as tk #GUI(grafisel kullanıcı arayüz ) oluşturmak için kullanılan kütüpkanedir.
import pandas as pd #veri işlemede kullandik
import numpy as np#veri işlemede kullandık
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Veriyi yükleyin ve ön işlemleri gerçekleştirdik
data = pd.read_csv("diabetes.csv")#veri setini içe aktardık
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].replace(0, np.NaN)
data.fillna(data.mean(), inplace=True)#Eksik verileri sütunların ortalama değerleri ile doldurduk.
x = data[['Glucose', 'BMI', 'Age', 'Pregnancies']]
y = data.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=166, max_depth=8)
clf = clf.fit(x_train, y_train)

# Tkinter arayüzü
root = tk.Tk()
root.title("Diabetes Prediction")#tikenter pencerrisin başlığı

labels = ['Glucose', 'BMI', 'Age', 'Pregnancies'] #
entries = []

# Giriş alanları oluştur
for label in labels: #kullanıcının gireceği değişken isimlerini içerir
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=5)

    lbl = tk.Label(frame, text=label)
    lbl.pack(side=tk.LEFT)

    entry = tk.Entry(frame)
    entry.pack(side=tk.LEFT)
    entries.append(entry)

# Tahmin fonksiyonu
def predict():
    try:
        # Kullanıcıdan alınan girdileri toplayın
        input_data = [float(entry.get()) for entry in entries]#Kullanıcıdan alınan giriş verilerini toplar ve bir diziye dönüştürür.
        input_data = np.array(input_data).reshape(1, -1)#Veriyi modele uygun hale getirmek için yeniden şekillendirir.
        
        # Model ile tahmin yapın
        prediction = clf.predict(input_data)[0]
        result_text = "Diabetes Prediction: Positive" if prediction == 1 else "Diabetes Prediction: Negative"#Model ile tahmin yapar ve sonucu (Positive veya Negative) ekrana yazdırır.
    except ValueError:
        result_text = "Lütfen geçerli sayısal değerler girin." #Girişlerde hata olduğunda (geçersiz değerler girildiğinde) hata mesajı gösterir.
    
    result_label.config(text=result_text)

# Tahmin butonu ve sonuç etiketi
button = tk.Button(root, text="Predict", command=predict)#Tahmin işlemini başlatacak bir buton oluşturulur ve GUI'ye eklenir.
button.pack(pady=20)

result_label = tk.Label(root, text="") #Tahmin sonucunu gösterecek bir etiket oluşturulur ve GUI'ye eklenir.
result_label.pack()

# Ana döngüyü başlat
root.mainloop()#Tkinter ana döngüsü başlatılarak GUI'nin çalışması sağlanır.
