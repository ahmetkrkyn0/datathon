# ============================================================================
# TIER-3 — multilingual-e5-large embedding (COLAB GPU, tek hucre)
# ============================================================================
# AMAC: bert_embed.py (CPU) ile BIREBIR AYNI emb_train.npy + emb_test.npy'yi GPU'da
#       ~1-2 dk'da uretmek. Cikti row-bagimsiz + frozen -> fold-leakage YOK; CV pipeline
#       (e5_ridge.py / ensemble.py) bu .npy'leri yukler, torch'a gerek kalmaz.
#
# KULLANIM (Colab):
#   1) Runtime -> Change runtime type -> Hardware accelerator = GPU (T4 yeterli).
#   2) Google Drive'a su iki dosyayi yukle:  data/train.csv  ve  data/test_x.csv
#      (repodaki KANONIK dosyalar; Kaggle'dan degil. Satir sirasi BIREBIR korunmali.)
#   3) Asagidaki DRIVE_DIR'i dosyalari koydugun klasore gore ayarla.
#   4) Bu hucreyi calistir. Cikti AYNI klasore: emb_train.npy, emb_test.npy
#   5) O iki .npy'yi indir -> repoda artifacts/ icine koy.
#
# DETERMINIZM: query-prefix + normalize=True + float32; SEED=42. GPU float'i CPU'dan ~1e-6
#   sapabilir ama row-bagimsiz embedding + Ridge'e gore ihmal edilebilir (sonuc gecerli).
# ============================================================================

# --- 0) Drive mount + klasor ---
from google.colab import drive
drive.mount('/content/drive')

# !!! BURAYI KENDI KLASORUNE GORE AYARLA !!!
DRIVE_DIR = '/content/drive/MyDrive/datathon26'   # train.csv + test_x.csv burada olmali

# --- 1) Bagimlilik (Colab'da torch+GPU zaten var; sadece sentence-transformers) ---
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
                'sentence-transformers==3.0.1'], check=True)

# --- 2) Sabitler (bert_embed.py ile AYNI sozlesme) ---
import os
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

SEED = 42
HF_MODEL_ID = 'intfloat/multilingual-e5-large'
E5_PREFIX = 'query: '
EMB_DIM = 1024
TEXT_COL = 'mentor_feedback_text'
ID_COL = 'student_id'

torch.manual_seed(SEED)
np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device =', device, '|', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU-ONLY (GPU runtime sec!)')

# --- 3) KANONIK veri (utf-8-sig; bert_embed.py = cv.load_train/test ile ayni okuma) ---
train = pd.read_csv(os.path.join(DRIVE_DIR, 'train.csv'), encoding='utf-8-sig')
test = pd.read_csv(os.path.join(DRIVE_DIR, 'test_x.csv'), encoding='utf-8-sig')
print('train', train.shape, '| test', test.shape)
# Satir sirasi/ID kontrol (fold hizalamasi icin KRITIK)
assert list(train[ID_COL])[:1] == ['STU_000001'], 'train ID sirasi beklenenden farkli!'
assert list(test[ID_COL])[:1] == ['STU_010001'], 'test ID sirasi beklenenden farkli!'
assert len(train) == 10000 and len(test) == 10000, 'satir sayisi 10000 degil!'

# --- 4) e5 prefix + GPU encode (deterministik, sira korunur) ---
def prefixed(texts):
    return [E5_PREFIX + ('' if t is None else str(t)) for t in texts]

model = SentenceTransformer(HF_MODEL_ID, device=device)
model.eval()

with torch.no_grad():
    emb_tr = model.encode(prefixed(train[TEXT_COL].values), batch_size=64,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    emb_te = model.encode(prefixed(test[TEXT_COL].values), batch_size=64,
                          normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)

emb_tr = np.ascontiguousarray(emb_tr, dtype=np.float32)
emb_te = np.ascontiguousarray(emb_te, dtype=np.float32)

# --- 5) Dogrulama (bert_embed.py assert'leriyle ayni) ---
assert emb_tr.shape == (10000, EMB_DIM), emb_tr.shape
assert emb_te.shape == (10000, EMB_DIM), emb_te.shape
assert np.isfinite(emb_tr).all() and np.isfinite(emb_te).all(), 'NaN/Inf var!'
print('norm (birim olmali): train', round(float(np.linalg.norm(emb_tr, axis=1).mean()), 5),
      '| test', round(float(np.linalg.norm(emb_te, axis=1).mean()), 5))

# --- 6) Kaydet (AYNI Drive klasorune) ---
np.save(os.path.join(DRIVE_DIR, 'emb_train.npy'), emb_tr)
np.save(os.path.join(DRIVE_DIR, 'emb_test.npy'), emb_te)
print('YAZILDI ->', DRIVE_DIR, ': emb_train.npy', emb_tr.shape, '| emb_test.npy', emb_te.shape)
print('Bu iki .npy dosyasini indirip repoda artifacts/ icine koy.')
