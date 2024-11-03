from PIL import Image
import pandas as pd
import numpy as np 
from IPython.display import Image as show_image
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Görüntüyü açma ve boyutunu değiştirme
# Resim, belirli bir dosya yolundan okunur ve boyutu 299x299 piksele ayarlanır.
img = Image.open("C:/Users/ASAF/Desktop/Datasets/araba.jpeg").resize((299,299))

# Görüntüyü numpy dizisine dönüştürme
# Bu adım, görüntüyü modelin kabul edebileceği bir format olan numpy dizisine dönüştürür.
img = np.array(img)

# Numpy dizisinin şeklini modelin beklediği şekle dönüştürme
# Modelin girdi olarak beklediği şekil: (numara, yükseklik, genişlik, kanal sayısı)
img = img.reshape(-1,299,299,3)

# Görüntünün şekli hakkında bilgi yazdırma
x = img.shape
print(x)

# Ön işleme yapma
# Bu adım, modelin daha doğru tahminler yapabilmesi için görüntüyü normalleştirir.
img = preprocess_input(img)

# InceptionResNetV2 modelini yükleme
# Model, önceden eğitilmiş "imagenet" ağırlıklarıyla yüklenir.
model = InceptionResNetV2(weights="imagenet",classes=1000) 

# Modelin mimarisini özetleme
# Bu, modelin katmanlarının ve parametrelerinin bir özetini verir.
model.summary()

# Orijinal görüntüyü gösterme
# Jupyter Notebook gibi bir ortamda görüntünün gösterilmesi için kullanılır.
show_image("C:/Users/ASAF/Desktop/araba.jpeg")

# Modeli kullanarak görüntü üzerinde tahmin yapma
# Model, ön işlenmiş görüntü üzerinde çalışarak sınıf tahminleri yapar.
pred = model.predict(img)

# Tahmin sonuçlarını çözme ve yorumlama
# Modelin çıktısını, anlaşılabilir etiketlere dönüştürür ve en olası 5 sınıfı listeler.
preds = decode_predictions(pred, top=5)

# Tahmin edilen sınıfları ve olasılıklarını yazdırma
print(preds)