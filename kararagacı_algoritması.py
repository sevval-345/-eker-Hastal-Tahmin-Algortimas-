# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 21:25:53 2024

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Şevval MIKÇI 
"""

import pandas as pd # veri işleme için kullanılan bir kütüphanedir. 
import numpy as np # matematik işlemleri için kullanılan bir kütüphanedir.
import seaborn as sns # verileri görselleştirmek için kullanılan kütüphanedir.
from sklearn.ensemble import RandomForestClassifier # kullandığımız karar ağacı algoritması.
from sklearn.model_selection import train_test_split # verileri test ve eğitim verileri olarak ayırmak için kullanılır. 
from sklearn import metrics # doğruluk oranını hesaplamak için kullanılan kütüphanedir.
import matplotlib.pyplot as plt  # veri görselleştirmede kullanılır.
from sklearn import tree # ağaç yapısını oluşturduk.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score



# Veri yükledik.
data = pd.read_csv("diabetes.csv")

# Veri setinde eksik veri var mı kontrol ettik.
print((data.isnull().sum()))

# Sıfır değerini içeren verilere bakalım.
print((data.eq(0).sum()))

# Eksik verileri görselleştirdik.
sns.heatmap(data.eq(0), cbar=False)

# Sıfır olan kolonları NaN yazdık.
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction','Age']].replace(0, np.NaN)

# Eksik verileri ortalama ile doldurduk.
data.fillna(data.mean(), inplace=True)

# Sıfır değerlerini tekrar kontrol ettik.
print((data.eq(0).sum()))

# Verinin korelasyon matrisini hesaplayıp görselleştirdik.
print(data.corr())
sns.heatmap(data.corr(), annot=True)

# En yüksek korelasyona sahip 5 özellik
print(data.corr().nlargest(5, 'Outcome').index)

# Verileri şeker değerine göre çıktı durumunu veren matris.
plt.title("Glucose ve diyabet")
sns.scatterplot(x="Glucose", y="Outcome", data=data, s=30, edgecolor="red")
plt.plot()

# Veri setini x ve y olarak ayırdık 
x = data[['Glucose','BMI','Age','Pregnancies']]
y = data.iloc[:,8]

# Veriyi eğitim ve test olarak ayırmak 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
clf = RandomForestClassifier(n_estimators=166, max_depth=8) # 166 karar ağacı, 8 derinlik
clf = clf.fit(x_train, y_train)

# Test seti üzerinde tahmin 
y_pred = clf.predict(x_test)

# Doğruluk oranını yazdırır
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Karar ağacı görselleştirme işlemi uyguladık
f_n = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
t_n = ["0", "1"]
fig = plt.figure(figsize=(60, 20), dpi=100)
plot = tree.plot_tree(clf.estimators_[5], feature_names=f_n, class_names=t_n, filled=True)
fig.savefig("Tree1.png")


# Performans metriklerini hesaplayan fonksiyon
def calculate_confusion_matrix_rates(cm):
    # True Positives, False Positives, False Negatives, True Negatives hesaplamaları
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    # Her sınıf için doğruluk, hassasiyet, duyarlılık ve F1 skoru hesaplamaları
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
# Karar Ağacı için karmaşıklık matrisi
cm_tree = confusion_matrix(y_test, y_pred)
print("Karar Ağacı Karmaşıklık Matrisi:")
print(cm_tree)

# Karar Ağacı için performans metrikleri
class_rates_tree = calculate_confusion_matrix_rates(cm_tree)
print("Karar Ağacı Performans Metrikleri:")
print(f"Doğruluk (Accuracy): {class_rates_tree['accuracy']}")
print(f"Hassasiyet (Precision): {class_rates_tree['precision']}")
print(f"Duyarlılık (Recall): {class_rates_tree['recall']}")
print(f"F1 Skoru: {class_rates_tree['f1']}")
