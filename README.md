# License Plate Gate Access System
 YOLOv8 ile plaka tespiti/kırpma ve PaddleOCR kullanılarak oluşturulmuş license plate gate access system. Sistem, okunan plakayı `license_plate_gate_access_system/whitelist.txt` içindeki beyaz listeye göre kontrol eder. Beyaz listedeki plakalar için onay verilir ve Raspberry Pi üzerindeki LED 5 saniye yanar. Beyaz listede olmayan plakalar `license_plate_gate_access_system/unauthorized_plates.txt` dosyasına zaman bilgisiyle kaydedilir; bu durumda LED kapalı kalır.
 
![aaa](https://user-images.githubusercontent.com/87595266/179367849-7d33fd32-be4f-43b3-ac47-adb27b5d861b.png)

# Nasıl Çalışır
 `main.py` önce USB kameraları OpenCV ile tarar. Görüntü alınabilen USB kamera varsa sistem bu kameraların hepsinden düzenli olarak kare alır ve kareleri eş zamanlı işler. Çalışan USB kamera yoksa Raspberry Pi CSI/pin kamera için `picamera2` fallback olarak devreye girer. OCR ile okunan plaka `whitelist.txt` içinde varsa onay verilir; yoksa `unauthorized_plates.txt` dosyasına kaydedilir. Gerekli paket kurulumları yapıldıktan sonra aşağıdaki komutla çalıştırın:

```bash
.venv/bin/python license_plate_gate_access_system/main.py
```

Bu projede hazirlanan aktif sanal ortam proje kokundeki `.venv` dizinidir. Ortam `--system-site-packages` ile olusturuldu; Raspberry Pi kamera/sistem paketlerini sistemden, YOLO/Torch CPU ve PaddleOCR paketlerini `.venv` icinden kullanir.

```bash
.venv/bin/python license_plate_gate_access_system/main.py
```

`requirements.txt` dosyasi bu dogrulanmis ortamin kritik paket envanteridir. Raspberry Pi icin hazirlanan `torch==2.12.0+cpu` ve `torchvision==0.27.0+cpu` paketlerini normal `pip install torch torchvision` ile ezmeyin; gerektiginde `CODEX_NOTES.md` icindeki sanal ortam notunu kontrol edin.

Raspberry Pi OS üzerinde PiCamera fallback için kamera paketi sistemden kurulmuş olmalıdır:

```bash
sudo apt install python3-picamera2
```

PiCamera kullanılacaksa kamera önceden `libcamera-hello` ile görüntü veriyor olmalıdır.

Varsayılan PiCamera modu `2592x1944` ve `15 FPS` olarak sabitlenmiştir. USB kameralar için artık `v4l2-ctl` ile bildirilen en yüksek destekli çözünürlük otomatik seçilir; bu bilgi alınamazsa uygulama yaygın çözünürlükleri büyükten küçüğe deneyip kameranın gerçekten verdiği en yüksek modu kullanır. Gerekirse `PLAKA_USB_WIDTH` ve `PLAKA_USB_HEIGHT` ile çözünürlüğü elle sabitleyebilirsiniz. Uygulama canlı video akışı yerine varsayılan olarak her kameradan saniyede 2 kare yakalar ve plaka okuma işlemini bu kareler üzerinde çalıştırır.

USB kamera taraması varsayılan olarak `/dev/video0` karşılığı olan `0` indeksinden `9` indeksine kadar yapılır. USB kameralar belirli aralıklarla tekrar taranır; sonradan takılan USB kamera PiCamera fallback çalışırken de öncelik alır. Gerekirse şu değişkenlerle ayarlanabilir:

```bash
PLAKA_USB_INDEX_LIMIT=16 PLAKA_USB_RESCAN_SECONDS=5 PLAKA_USB_WIDTH=1920 PLAKA_USB_HEIGHT=1080 .venv/bin/python license_plate_gate_access_system/main.py
```

Plaka tespiti için hazır YOLOv8 modeli `license_plate_gate_access_system/models/license_plate_detector.pt` yolunda bulunur. Farklı bir model kullanmak için:

```bash
PLAKA_YOLO_MODEL=/path/to/license_plate_detector.pt .venv/bin/python license_plate_gate_access_system/main.py
```

YOLO güven eşiği varsayılan olarak `0.25` değerindedir. Değiştirmek için:

```bash
PLAKA_YOLO_CONF=0.4 .venv/bin/python license_plate_gate_access_system/main.py
```

YOLO adayları artık OCR'a gönderilmeden önce plaka benzeri oranlarla filtrelenir. Varsayılan olarak kutu en-boy oranı `2.0-6.5`, kare alan oranı ise `0.003-0.08` arasında olmalıdır. Gerekirse şu değişkenlerle ayarlanabilir:

```bash
PLAKA_MIN_ASPECT=2.0 PLAKA_MAX_ASPECT=6.5 PLAKA_MIN_AREA=0.003 PLAKA_MAX_AREA=0.08 .venv/bin/python license_plate_gate_access_system/main.py
```

Terminale başarıyla okunan ve formatı doğru olan her plaka kamera adıyla birlikte yazdırılır. Plaka `whitelist.txt` içinde varsa terminalde `ONAY` görünür ve LED 5 saniye yanar. Plaka listede yoksa terminalde `RED` görünür, `unauthorized_plates.txt` dosyasına yeni zaman bilgisiyle eklenir ve LED kapalı kalır. Bir karede birden fazla plaka tespit edilirse her plaka ayrı kırpılır ve ayrı OCR'a gönderilir. OCR metin okuyup plaka formatını reddederse ham ve temizlenmiş OCR çıktısı da terminale basılır; böylece PaddleOCR'ın plakayı nasıl yanlış algıladığı görülebilir. YOLO plaka bulamazsa tekrar eden normal başarısızlıklar yalnızca değiştiğinde yazdırılır.

## Beyaz Liste ve GPIO LED

Onay verilecek plakaları `license_plate_gate_access_system/whitelist.txt` dosyasına her satıra bir plaka gelecek şekilde yazın:

```text
34ABC034
34PAS98
```

Boşluklar yok sayılır ve plaka büyük harfe çevrilir. İsterseniz farklı bir beyaz liste dosyası kullanabilirsiniz:

```bash
PLAKA_WHITELIST_FILE=/path/to/whitelist.txt .venv/bin/python license_plate_gate_access_system/main.py
```

Onaya bağlı LED çıkışı varsayılan olarak BCM `17` GPIO pinidir. Bu pin Raspberry Pi 4 üzerinde fiziksel pin `11` karşılığıdır. Onaylı plaka sonrası yalnızca bu pin `HIGH` yapılır ve LED yanma süresi sonunda tekrar `LOW` yapılır. Ayrıca proje başladığında sabit çıkış olarak BCM `27` `HIGH` yapılır; Raspberry Pi 4 üzerinde fiziksel pin `13` karşılığıdır ve uygulama boyunca `HIGH` kalır. LED yanma süresi varsayılan olarak 5 saniyedir:

```bash
PLAKA_LED_GPIO_PIN=17 PLAKA_LED_ON_SECONDS=5 .venv/bin/python license_plate_gate_access_system/main.py
```

GPIO bağlantısı:

```text
Raspberry Pi fiziksel pin 11 (GPIO17 / BCM17) -> 220-330 ohm direnç -> LED uzun bacak/anot (+)
Raspberry Pi fiziksel pin 13 (GPIO27 / BCM27) -> proje boyunca sabit HIGH çıkış
LED kısa bacak/katot (-) -> Raspberry Pi fiziksel pin 9 (GND)
```

Alternatif toprak için fiziksel pin `6`, `14`, `20`, `25`, `30`, `34` veya `39` da kullanılabilir. LED'i dirençsiz bağlamayın.

Kamera tarafında en iyi sonuç için kamera ve plaka sabit durmalı, plaka kadrajda tamamen görünmeli ve parlama mümkün olduğunca azaltılmalıdır. Telefon/tablet ekranından okuma yapılacaksa ekran parlaklığı ve açı sabitlenmeli; gerçek plaka okuması ekrandan okuma denemelerine göre daha güvenilir sonuç verir.

## OpenCV / OCR hata ayıklama

Uygulama varsayılan olarak `debug_frames/` klasörüne örnek kareler kaydeder. Program her başlatıldığında eski `debug_frames/` içeriği temizlenir ve yalnızca yeni çalıştırmanın debug görüntüleri saklanır. Her kayıt klasöründe şu dosyalar bulunur:

```text
01_frame_raw.jpg          Kameradan gelen tam kare
03_frame_annotated.jpg    YOLO plaka kutuları ve seçilen plaka kutusu
04_crop_gray.jpg          YOLO'nun plaka diye kırptığı gri görüntü
05_crop_ocr_input.jpg     PaddleOCR'a verilen görüntü
metadata.json             OCR çıktısı, sonuç ve seçilen YOLO kutusu bilgileri
```

`03_frame_annotated.jpg` içinde yeşil kutu doğru plakayı göstermiyorsa sorun YOLO tespit/kırpma tarafındadır. Yeşil kutu ve `04_crop_gray.jpg` doğru plakayı gösteriyor ama `metadata.json` içindeki `raw_ocr` yanlışsa sorun OCR veya OCR ön işleme tarafındadır.

Debug kaydı her 2 saniyede bir yapılır; başarılı plaka okumaları ayrıca her zaman kaydedilir. Kapatmak için:

```bash
PLAKA_DEBUG=0 .venv/bin/python license_plate_gate_access_system/main.py
```

Kayıt aralığını değiştirmek için:

```bash
PLAKA_DEBUG_INTERVAL=5 .venv/bin/python license_plate_gate_access_system/main.py
```

Kayıt klasörünü değiştirmek için:

```bash
PLAKA_DEBUG_DIR=/tmp/plaka_debug .venv/bin/python license_plate_gate_access_system/main.py
```
# Youtube Önizleme Videosu
 https://youtu.be/HI5iR_xi_zY
