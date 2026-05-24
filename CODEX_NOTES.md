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
