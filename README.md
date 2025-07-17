**Yolov5 Model Çıktıları Analizi**

**mAP (Mean Average Precision):** Nesne tespitte modelin doğruluğunu ölçen standart metriktir.

\*Precision-Recall eğrisinin altında kalan alanın ortalamasıdır.

**mAP@0.5:** IoU (Intersection over Union) eşiği %50 kabul edilerek hesaplanır.

**mAP@0.5:0.95**: IoU 0.5’ten 0.95’e kadar 0.05 artışlarla hesaplanıp ortalaması alınır (daha zordur, COCO standardıdır).

- **0 ile 1 arasında değer alır.**
- 1’e ne kadar yakınsa model o kadar iyidir.

**R (Recall):**

Gerçek pozitiflerin ne kadarını modelin yakalayabildiğini gösterir.

Veya ne kadarını kaçırdığını gösterir.

Yani gerçekte var olan nesnelerin ne kadarını bulduğudur.

\*Recall= TruePositives/ ( TruePositives+FalseNegatives) \*

- 0 ile 1 arasında değer alır.
- 1’e ne kadar yakınsa model o kadar fazla gerçek nesneyi bulmuş demektir.

​

**P (Precision):**

P (Hassasiyet): Tespit edilen nesnelerin doğruluğu, kaç tespitin doğru olduğunu gösterir.

\*Precision= TruePositives /(TruePositives+FalsePositives)\*

- ​ 0 ile 1 arasında değer alır.
- 1’e ne kadar yakınsa model bulduğu nesnelerde o kadar az hata yapıyor demektir.

**F1 Skoru:**

F1 Skoru: F1 Skoru, kesinlik ve hatırlamanın harmonik ortalamasıdır ve hem yanlış pozitifleri hem de yanlış negatifleri dikkate alarak bir modelin performansının dengeli bir değerlendirmesini sağlar.

Precision (Kesinlik) ve Recall (Duyarlılık) arasında dengeyi ölçer.

\*F1= 2× (Precision+Recall)/(Precision×Recall)\*

- **​ 0 ile 1 arasında değer alır.**
- 1’e ne kadar yakınsa precision ve recall dengesi o kadar iyidir.

**Aralarındaki ilişki:**

**Precision yüksek, Recall düşük ise:** Model az ama doğru nesneleri buluyor.

**Recall yüksek, Precision düşük ise:** Model çok nesne buluyor ama yanlışları fazla.

**F1 skoru**, bu ikisinin dengelenmiş ölçüsünü verir.

**mAP**, modelin genel doğruluğunu IoU eşiklerine göre ölçer.

**Epoch:**

Tüm eğitim veri setinin modele bir kez gösterilmesi işlemidir.

1 epoch: Tüm train verilerinin forward + backward propagasyon işlemi tamamlanmasıdır.

Model genellikle çoklu epoch’larda eğitilir (ör: 50, 100, 300 epoch) ki öğrenme gerçekleşsin.Fazlası overfittinge azı da modelin iyi eğitilememesine sebep olur.

**box_loss (Bounding Box Loss):**

Modelin tahmin ettiği bounding box (kutu) koordinatlarının gerçek kutulara ne kadar yakın olduğunu ölçer.

Gelişmiş versiyonlarda CIoU veya GIoU loss kullanılarak hesaplanır.

**obj_loss (Objectness Loss):**

Modelin o bölgede bir nesne olup olmadığını doğru tahmin edip edemediğini ölçer.

Nesne olan yerlere yüksek olasılık, nesne olmayan yerlere düşük olasılık vermeyi öğrenmesini sağlar.

**cls_loss (Classification Loss)**

Tespit edilen nesnenin hangi sınıfa ait olduğunun doğru tahmin edilip edilmediğini ölçer.

Eğer yalnızca 1 sınıf varsa (ör: sadece insan), genellikle **cls_loss** sıfıra yakın olur.

**GPU (Graphics Processing Unit):**

YOLOv5 ve diğer deep learning modellerinin hızlı eğitilmesi için kullanılan donanım birimidir.

Çok çekirdekli paralel işlem yapabilmesi sayesinde:

Eğitim süresini ciddi oranda kısaltır.

Daha büyük batch size ile eğitim imkanı sağlar.

**Ozan Yılmaz**



# Yolov5objctdetection
This project uses the OpenCV, Torch, and Yolov5 libraries in Python to perform object recognition on the selected image and saves it to a specified path. Additionally, this project connects to the computer camera using Yolov5 to perform object recognition on the video.
