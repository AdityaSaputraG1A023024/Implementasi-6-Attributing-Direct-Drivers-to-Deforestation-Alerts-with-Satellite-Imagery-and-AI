# Data EuroSAT

Data citra berukuran besar tidak disimpan di Git. Jalankan `python -m src.data_loader --download` untuk mengunduh arsip subset RGB EuroSAT dari sumber yang dicatat di `config.yaml`. Berkas raw masuk ke `data/raw/` dan diabaikan oleh Git. Format data adalah Parquet dengan kolom image dan label; decoder gambar dipakai saat membaca.

Versi dataset dan metadata aktualnya (jumlah contoh, nama/proporsi kelas, mode dan ukuran citra) dicatat dalam `reports/dataset_metadata.json` setelah dataset diunduh/dibaca. Rujukan:

- EuroSAT upstream, versi multispektral dan RGB: https://github.com/phelber/EuroSAT
- EuroSAT RGB mirror yang dipakai kode: https://huggingface.co/datasets/nielsr/eurosat-demo
- Metadata versi mirror: https://huggingface.co/datasets/nielsr/eurosat-demo/resolve/main/dataset_infos.json

Jangan mengunggah arsip raw EuroSAT ke repositori. Hasil eksperimen dan cache lokal juga sebaiknya ditinjau sebelum commit.
