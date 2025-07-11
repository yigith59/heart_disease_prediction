# heart_disease_prediction
Kalp Hastalığı Tahmin Modeli

Bu proje, makine öğrenimi modellerini kullanarak kalp hastalığı varlığını tahmin etmeyi amaçlamaktadır. Farklı coğrafi bölgelerden toplanan çok çeşitli hasta verilerini bir araya getirerek, daha genellenebilir ve güvenilir bir tahmin modeli geliştirmeye odaklanılmıştır. 

Veri Seti
Projede beş farklı kalp hastalığı veri seti kullanılmıştır:

processed.cleveland.data 
kaynak = https://archive.ics.uci.edu/dataset/45/heart+disease

processed.switzerland.data
kaynak = https://archive.ics.uci.edu/dataset/45/heart+disease

processed.va.data
kaynak = https://archive.ics.uci.edu/dataset/45/heart+disease
 
processed.hungarian.data
kaynak = https://archive.ics.uci.edu/dataset/45/heart+disease

Cardiovascular_Disease_Dataset.csv (Hindistan)
kaynak = https://data.mendeley.com/datasets/dzz48mvjht/1


clevland,switzerland,va ve hungary veri seti kendi aralarında birleştirilmiş olup bu veri setlerinin hazır hale getirilmesinde kullanılan kodlar ucl.py dosyasında bulunmakta. düzenlemeler sonucu oluşturulan dataframe df_diğer.csv olarak kaydedildi
hindistana ait veri setine yapılan düzenlemeler ise india.py dosyasında bulunmakta . düzenlemeler sonucu oluşturulan dataframe df_india.csv olarak kaydedildi

Proje Aşamaları için Kısa Açıklamalar

1. Aşama: Hindistan Veri Seti Üzerine Odaklanma
Bu aşama, Hindistan'dan gelen kalp hastalığı veri seti üzerine odaklanmıştır. Amacımız, tek bir coğrafi bölgeye özgü verileri kullanarak bir temel model oluşturmak ve veri ön işleme, eksik değer yönetimi, aykırı değer kontrolü ve detaylı özellik mühendisliği adımlarını test etmekti.

Bu aşamada geliştirilen özel özellikler arasında age_gender kombinasyonları, komorbite, rest ve egzersiz_toleransi_skoru bulunmaktadır.


2. Aşama: Diğer Ülkelerin Veri Setleri Üzerine Odaklanma
Bu kısım, Cleveland, İsviçre, VA ve Macaristan'dan gelen dört ayrı kalp hastalığı veri setinin birleştirilmesiyle oluşturulan veri seti üzerinde çalışmayı içerir. Bu aşamada, ilk aşamadaki gibi benzer veri ön işleme, eksik değer doldurma, aykırı değer kontrolü ve özellik mühendisliği adımları uygulandı.


3. Aşama: Tüm Veri Setlerinin Birleştirilmesi ve Nihai Model Oluşturma
Bu aşama, projenin en kritik ve sonuç odaklı bölümüdür. İlk iki aşamada ayrı ayrı işlenen Hindistan veri seti ve diğer dört ülkenin birleştirilmiş veri seti tek bir büyük veri havuzunda birleştirilmiştir. birleştirmeye ait kodlar heart_last.py dosyasındadır

Bu birleşik veri seti üzerinde, daha önce geliştirilen tüm kapsamlı veri ön işleme ve özellik mühendisliği adımları tutarlı bir şekilde uygulanmıştır. Ardından, veri seti eğitim ve test setlerine ayrılmış, Standard Scaler ile ölçeklendirilmiştir. Çeşitli makine öğrenimi modelleri (LightGBM, XGBoost, LogisticRegression vb.) çapraz doğrulama ile recall metriği üzerinden karşılaştırılmış ve LightGBM modeli GridSearchCV ile optimize edilmiştir.


Veri Seti Kolonları ve Özellikleri

age = yaş bilgisi
gender = cinsiyet bilgisi
chestpain = göğüs ağrısı tipi
restingBP = dinlenme kan basıncı
serumcholestrol = serum kolesterol değeri
fastingbloodsugar = kan şekerinin 120 nin üzerinde olup olmadığı
restingelectro = dinlenme ekg sonucu
maxheartrate = tesste ulaşılan max kalp hızı
exerciseangia = egzersizde angina olup olmadığı
oldpeak = ST segmenti değişikliği

komorbite: Hastanın serum kolesterolü ve açlık kan şekeri seviyelerine göre türetilmiş komorbidite yükü
rest: Dinlenme elektrokardiyografik sonuçları ve yüksek dinlenme kan basıncına göre türetilmiş özellik
max_hr: Hastanın yaşına göre beklenen maksimum kalp atış hızına kıyasla ulaştığı maksimum kalp atış hızı seviyesi
target = kalp hastalığı varlığı
egzersiz_toleransi_skoru: Egzersize bağlı anjina, max_hr seviyesi ve oldpeak değerleri dikkate alınarak oluşturulmuş hastanın egzersiz toleransını özetler
age_gender değişkenleri = yaş grubu ve cinsiyete göre oluşturulan kombinasyonlara one hot encoding uygulanmış hali

target = hastalık varlığı


Model test verisi üzerinde dengeli ve yüksek başarı göstermiştir:
Accuracy (Doğruluk): %85
ROC AUC Skoru: 0.9133 → Modelin pozitif (hasta) ve negatif (sağlıklı) sınıfları ayırt etme gücü oldukça yüksek.
Recall (Pozitif sınıf / hasta): %90 → Kalp hastası bireyleri doğru tespit etme oranı yüksek.
Precision (Pozitif sınıf): %86 → "Hasta" olarak tahmin edilen bireylerin %86’sı gerçekten hasta



















