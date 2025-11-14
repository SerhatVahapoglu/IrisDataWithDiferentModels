# IrisDataWithDiferentModels
# 🌸 Iris Dataset Classification & Regression Analysis

Bu proje, klasik **Iris veri seti** üzerinde farklı makine öğrenimi algoritmalarının performanslarını karşılaştırmak için hazırlanmıştır.  
Amaç, hem **sınıflandırma (Naive Bayes)** hem de **regresyon (Linear Regression & SVR)** yöntemlerini uygulayıp değerlendirmektir.

---

##  İçindekiler
- [Proje Amacı](#proje-amacı)
- [Veri Keşfi (EDA)](#veri-keşfi-eda)
- [Modelleme Aşamaları](#modelleme-aşamaları)
  - Gaussian Naive Bayes
  - Linear Regression
  - Support Vector Regression (SVR)
  - SVR + GridSearchCV (Hyperparameter Tuning)
- [Sonuçlar](#sonuçlar)
- [Yorum ve Çıkarımlar](#yorum-ve-çıkarımlar)
- [Yazar](#yazar)

---

## 🎯 Proje Amacı

Bu çalışmada aşağıdaki sorulara yanıt aranmıştır:
1. Iris çiçek türlerini (setosa, versicolor, virginica) sınıflandırmak için hangi model daha etkilidir?  
2. Sepal/Petal uzunluk ve genişlikleri üzerinden doğrusal ve doğrusal olmayan regresyon modelleri nasıl sonuç verir?  
3. GridSearchCV ile hiperparametre optimizasyonu modeli ne kadar iyileştirir?

---

🔍 Veri Keşfi (EDA)

Kullanılan veri: 11-iris.csv

150 örnek, 5 nitelik:

SepalLengthCm

SepalWidthCm

PetalLengthCm

PetalWidthCm

Species (hedef)

Tür	Örnek Sayısı
Iris-setosa	50
Iris-versicolor	50
Iris-virginica	50
📊 Görselleştirme

Pairplot grafikleri: türler arasında PetalLength ve PetalWidth değişkenlerinin güçlü ayırt edici olduğunu gösteriyor.

Scatter plot’larda Iris-setosa net ayrılırken, versicolor ve virginica kısmen örtüşüyor.

🤖 Modelleme Aşamaları
1️⃣ Gaussian Naive Bayes (Classification)
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)
y_pred = gnb.predict(X_test_scaled)


Sonuçlar:

Metrik	Değer
Accuracy	1.0 (100%)
Precision	1.00
Recall	1.00
F1-score	1.00

✅ Tüm sınıflar mükemmel biçimde sınıflandırılmıştır.
Bu, Iris veri setinin Naive Bayes ile oldukça kolay ayrılabildiğini gösterir.

2️⃣ Linear Regression (Regression)
linreg = LinearRegression()
linreg.fit(X_train_scaled, y_train)


Sonuçlar:

Metrik	Değer
Mean Absolute Error	0.1572
R² Score	0.9339

📈 Model, verideki varyansın %93’ünü açıklayabiliyor.

3️⃣ Support Vector Regression (SVR)
svr = SVR()
svr.fit(X_train_scaled, y_train)


Sonuçlar:

Metrik	Değer
Mean Absolute Error	0.1370
R² Score	0.9528

RBF kernel sayesinde doğrusal olmayan ilişkilere daha iyi uyum sağlamış.
R² değeri Linear Regression’a göre biraz daha yüksek.

4️⃣ SVR + GridSearchCV (Hyperparameter Optimization)
param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['rbf', 'linear']
}
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3, n_jobs=-1)


En iyi parametreler:

{'C': 1, 'gamma': 0.1, 'kernel': 'rbf'}


Sonuçlar:

Metrik	Değer
Mean Absolute Error	0.1319
R² Score	0.9498

GridSearchCV sonrası skorlar benzer kaldı; model zaten optimizeye oldukça yakındı.

📈 Sonuç Karşılaştırma Tablosu
Model	Tür	Accuracy / R²	MAE	Açıklama
GaussianNB	Classification	1.00	–	Tüm türleri hatasız ayırdı
Linear Regression	Regression	0.9339	0.157	Doğrusal ilişki güçlü
SVR	Regression	0.9528	0.137	RBF kernel ile daha iyi sonuç
SVR (GridSearchCV)	Regression	0.9498	0.132	Parametre ayarıyla minimal iyileşme

