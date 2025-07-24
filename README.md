# 🏆 Futbol Maç Tahmin Sistemi

Makine öğrenmesi algoritmaları kullanarak futbol maçlarının sonuçlarını tahmin eden gelişmiş bir masaüstü uygulaması.

## 📋 İçindekiler

- [Özellikler](#özellikler)
- [Teknolojiler](#teknolojiler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Model Mimarisi](#model-mimarisi)
- [Veritabanı Yapısı](#veritabanı-yapısı)
- [Tahmin Türleri](#tahmin-türleri)
- [Dosya Yapısı](#dosya-yapısı)
- [Katkıda Bulunma](#katkıda-bulunma)

## ✨ Özellikler

### 🎯 Tahmin Yetenekleri

- **İlk Yarı Tahminleri**: Kazanan, skor, gol sayısı, karşılıklı gol
- **Maç Sonu Tahminleri**: Sonuç, skor, toplam gol, gol aralığı
- **İkinci Yarı Tahminleri**: Kazanan, gol istatistikleri, karşılıklı gol
- **Özel Tahminler**: Gol farkı, en çok gol olan yarı, takım bazlı gol tahminleri

### 🧠 Gelişmiş ML Algoritmaları

- **Ensemble Learning**: RandomForest + XGBoost + LightGBM kombinasyonu
- **Feature Engineering**: 50+ özellik ile derinlemesine analiz
- **Cross Validation**: 5-fold CV ile model doğrulaması
- **Hyperparameter Optimization**: Optimize edilmiş model parametreleri

### 🎨 Kullanıcı Arayüzü

- Modern PyQt5 GUI
- Koyu tema tasarım
- Responsive layout
- Real-time tahmin sonuçları

## 🛠 Teknolojiler

### Backend

- **Python 3.8+**
- **Scikit-learn**: Temel ML algoritmaları
- **XGBoost**: Gradient boosting
- **LightGBM**: Microsoft'un hızlı GB algoritması
- **Pandas & NumPy**: Veri işleme

### Veritabanı

- **SQLite**: Maç verileri ve istatistikler

### GUI

- **PyQt5**: Masaüstü arayüzü

### Model Persistence

- **Joblib**: Model kaydetme/yükleme

## 📦 Kurulum

### Gereksinimler

```bash
pip install pandas numpy scikit-learn xgboost lightgbm joblib PyQt5 matplotlib seaborn
```

### Proje Kurulumu

```bash
git clone https://github.com/kullaniciadi/futbol-tahmin-sistemi.git
cd futbol-tahmin-sistemi
```

### Veritabanı Kontrolü

```bash
python analysis.py  # Veritabanı analizi
```

### Model Eğitimi

```bash
python model_training.py  # Tüm ligler için model eğitimi
```

### Uygulamayı Çalıştırma

```bash
python app.py
```

## 🚀 Kullanım

1. **Lig Seçimi**: Dropdown menüden istediğiniz ligi seçin
2. **Takım Seçimi**: Ev sahibi ve deplasman takımlarını belirleyin
3. **Maç Detayları**: Tarih, saat ve sezon bilgilerini girin
4. **Tahmin**: "Tahmin Yap" butonuna tıklayın
5. **Sonuçlar**: Detaylı tahmin sonuçlarını görüntüleyin

## 🧬 Model Mimarisi

### Feature Engineering (feature_engineering.py)

```python
# Takım formu analizi
- Son 6 maç performansı
- Gol ortalamaları
- Temiz çarşaf oranları
- Karşılıklı gol istatistikleri

# Head-to-Head istatistikleri
- Karşılıklı geçmiş performans
- Ortalama gol sayıları
- Kazanma oranları

# Sezon içi performans
- Mevcut sezon istatistikleri
- Ev/deplasman avantajı
- Mevsimsel etkiler
```

### Model Training (model_training.py)

```python
# Ensemble Architecture
VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier()),
        ('xgb', XGBClassifier()),
        ('lgb', LGBMClassifier())
    ],
    voting='soft'  # Olasılık bazlı oylama
)

# Model Optimization
- 800-1200 estimators
- Balanced class weights
- Grid search optimized parameters
```

### Prediction Engine (prediction.py)

```python
# Tahmin Türleri
- Binary Classification (Alt/Üst, Var/Yok)
- Multi-class Classification (Kazanan, Skor)
- Regression (Gol sayıları)
- Probability Estimation (Güven skorları)
```

## 🗄 Veritabanı Yapısı

### Tablolar

- **leagues**: Lig bilgileri
- **countries**: Ülke bilgileri
- **teams**: Takım bilgileri
- **fixtures**: Maç sonuçları
- **players**: Oyuncu bilgileri (opsiyonel)

### Veri Kalitesi

- Tutarlılık kontrolleri
- Eksik veri analizi
- Sezon geçiş validasyonu

## 🎯 Tahmin Türleri

### İlk Yarı

- Kazanan (Ev/Deplasman/Berabere)
- Tam skor tahmini
- Toplam gol (0.5, 1.5, 2.5 alt/üst)
- Takım bazlı gol tahminleri
- Karşılıklı gol durumu
- Gol farkı

### Maç Sonu

- Final sonucu
- Tam skor tahmini
- Toplam gol (1.5, 2.5, 3.5 alt/üst)
- Gol aralığı (0-1, 2-3, 4-5, 6+)
- Karşılıklı gol durumu
- En çok gol olan yarı

### İkinci Yarı

- Kazanan taraf
- Gol istatistikleri
- Karşılıklı gol durumu
- Gol farkı analizi

## 📁 Dosya Yapısı

```
futbol-tahmin-sistemi/
├── app.py                    # Ana GUI uygulaması
├── analysis.py               # Veritabanı analiz modülü
├── feature_engineering.py    # Özellik mühendisliği
├── model_training.py         # Model eğitim sistemi
├── prediction.py             # Tahmin motoru
├── sports_data.db           # SQLite veritabanı
├── models/                  # Eğitilmiş modeller
│   ├── league_*_model.joblib
│   └── league_*_scaler.joblib
└── README.md               # Proje dokümantasyonu
```

## 🎲 Model Performansı

### Doğruluk Oranları

- **Maç Sonucu**: ~%65-70
- **Toplam Gol Alt/Üst**: ~%60-65
- **Karşılıklı Gol**: ~%58-63
- **Gol Aralığı**: ~%40-45
- **Tam Skor**: ~%15-20

### Optimizasyon Teknikleri

- Class balancing
- Feature scaling
- Cross-validation
- Ensemble methods
- Hyperparameter tuning

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- Scikit-learn topluluğu
- XGBoost ve LightGBM geliştiricileri
- PyQt5 framework
- Futbol veri sağlayıcıları

## 📞 İletişim

Herhangi bir sorunuz varsa, lütfen bir issue açın veya benimle iletişime geçin.

---

⭐ **Bu projeyi beğendiyseniz, lütfen star verin!**
