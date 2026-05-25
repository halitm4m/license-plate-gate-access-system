# Codex Notes

Bu dosya Codex oturumlari icin proje ici hatirlatma notudur.

## Calisma Kurali

- Codex bu projede bir dosya degisikligi yaptiginda, is bitmeden once ilgili degisiklikleri git commit olarak kaydetmelidir.
- Commit mesaji yapilan isi kisa ve acik anlatmalidir.
- Commit atmadan once `git status --short` ile kapsam kontrol edilmelidir.
- Kullaniciya ait veya calisma aninda olusmus ilgisiz dosyalar commit'e dahil edilmemelidir.
- Runtime ciktilari, kamera/OCR deneme verileri, cache dosyalari ve debug gorselleri kaynak kod degisikligi degilse commit'e alinmamalidir.
- Bir sonraki Codex cagrisi once `git log --oneline -n 10` ve `git status --short` ile onceki islemleri kontrol etmelidir.

## Bu Notun Amaci

Onceki Codex islemlerinin sadece calisma agacinda kaybolmasini engellemek ve sonraki oturumlarda yapilan islerin git gecmisinden izlenebilmesini saglamak.

## Terminal Log Kurali

- Terminale kamera baslangic ayarlari disinda yalnizca plaka sonuc ve basarisiz deneme loglari yazilmalidir.
- Onayli plaka log formati: `++ 34PAS98`.
- Onaysiz plaka log formati: `-- 04ASD89`.
- Basarisiz deneme log formati: `BD: 2`.
- Terminale traceback, hata ayrintisi, GPIO durumu, kamera devreden cikti mesaji veya baska aciklama yazdirilmamalidir.
- Bu terminal loglama sistemi kullanici acikca istemedikce degistirilmemelidir.

## Python Sanal Ortam Notu

- Aktif ve birlestirilmis sanal ortam proje kokundeki `.venv` dizinidir.
- Calistirma komutu: `.venv/bin/python PlakaOkuma-NumberPlateRecognition/main.py`
- Bu `.venv`, Raspberry Pi sistem paketlerini gorebilmek icin `--system-site-packages` ile olusturulmustur.
- 2026-05-24 tarihinde eski `/home/pi/projects/plate_env` ortamindaki calisan YOLO/Torch kurulumu yeni `.venv` icine tasindi; PaddleOCR kurulumu yeni `.venv` icinde korundu.
- Dogrulanan kritik paketler: `opencv-python==4.13.0.92`, `numpy==2.3.5`, `setuptools==81.0.0`, `torch==2.12.0+cpu`, `torchvision==0.27.0+cpu`, `ultralytics==8.4.53`, `ultralytics-thop==2.0.19`, `polars==1.40.1`, `polars-runtime-32==1.40.1`, `paddleocr==2.10.0`, `paddlepaddle==3.1.1`, `matplotlib==3.10.9`, `sympy==1.14.0`, `mpmath==1.3.0`.
- Torch/Torchvision paketleri Raspberry Pi icin CPU-only kurulumdur. Bunlari normal `pip install torch torchvision` ile ezme; gerekmedikce yeniden kurulum yapma.
- Proje, PaddleOCR cache dosyalarini `.paddlex_cache/` ve `.paddleocr_cache/`, Matplotlib cache dosyalarini `.matplotlib_cache/` altinda tutacak sekilde ayarlidir.
