# Panduan Pengujian

## Persiapan
Untuk menjalankan pengujian, pastikan Anda telah menginstal dependensi yang diperlukan:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov
```

Kemudian, instal proyek dalam mode pengembangan:

```bash
pip install -e .
```

## Menjalankan Pengujian
Untuk menjalankan semua pengujian:

```bash
pytest
```

Untuk menjalankan dengan cakupan kode:

```bash
pytest --cov=src tests/
```

Untuk menghasilkan laporan cakupan HTML:

```bash
pytest --cov=src --cov-report=html tests/
```
Laporan akan tersedia di direktori `htmlcov/`.

## Struktur Pengujian

Folder `tests/` berisi pengujian yang diorganisir sesuai dengan modul yang diuji:

- `test_utils.py`: Pengujian utilitas teknis
- `test_data_preprocessing.py`: Pengujian pemrosesan data
- `test_model.py`: Pengujian model prediksi
- `test_trading_strategies.py`: Pengujian strategi trading
- `test_backtest.py`: Pengujian backtesting
- `test_optimizer.py`: Pengujian pengoptimalan strategi 