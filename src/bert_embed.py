"""
TIER-3 SPEKULATIF — multilingual-e5-large embedding CIKARIMI (IZOLE, tek-seferlik).
====================================================================================

    python src/bert_embed.py            # cache yoksa cikar, varsa atla
    python src/bert_embed.py --force    # cache'i yeniden uret

NE YAPAR: `mentor_feedback_text`'i intfloat/multilingual-e5-large ile 1024-boyutlu yogun
vektore cevirir ve KANONIK artefakt olarak cache'ler:
    artifacts/emb_train.npy   (10000, 1024)  float32
    artifacts/emb_test.npy    (10000, 1024)  float32

NEDEN IZOLE / FOLD-SAFE:
  * Model FROZEN (egitim yok) ve hesap GLOBAL (her satir digerlerinden BAGIMSIZ kodlanir) ->
    fold-leakage YAPISAL OLARAK IMKANSIZ. Embedding bir satirin KENDI metninin deterministik
    fonksiyonudur; hedefe/diger satirlara/fold'a bakmaz.
  * Bu .npy'ler kanonik reproducible artefakttir: CV pipeline (e5_ridge.py / ensemble.py) bunlari
    yukler ve torch'a IHTIYAC DUYMAZ. requirements.txt KIRLENMEZ (torch yalniz requirements-embed.txt).

E5 SOZLESMESI: model her metne "query: " prefix'i bekler (asimetrik retrieval encoder; tek-cumle
  gomme icin query-prefix standardi). normalize_embeddings=True -> birim-norm (cosine ~ dot).

DETERMINIZM (SEED=42): torch.manual_seed + CPU + eval(). encode() sirasi giris sirasiyla ayni
  (convert_to_numpy, no shuffle). Tek-thread BLAS (cv.set_seed env) + CPU -> ayni .npy.

GIT: models/ ve HF cache .gitignore'da (model COMMIT EDILMEZ). bert_embed.py kendi indirir.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# transformers'in TF/Flax backend'ini PROBE etmesini ENGELLE: TF import'u ~1-2GB RAM yer ve bu
# makinede (15.8GB toplam, ~4GB bos) e5-large encode'uyla birlikte OOM'a sokuyordu (sessiz kill).
# torch-only zorla: hem setdefault hem KESIN set (= ile) + tensorflow'u import-aninda gorunmez yap.
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "0")

import numpy as np

import cv  # SEED, yol sozlesmesi, load_train/load_test (utf-8-sig, KANONIK satir sirasi)

# --------------------------------------------------------------------------- #
# Sabitler & yol sozlesmesi
# --------------------------------------------------------------------------- #
HF_MODEL_ID = "intfloat/multilingual-e5-large"
MODELS_DIR = cv.ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"  # .gitignore'da (models/ kapsami); model burada cache'lenir

EMB_TRAIN_PATH = cv.ARTIFACTS_DIR / "emb_train.npy"
EMB_TEST_PATH = cv.ARTIFACTS_DIR / "emb_test.npy"

E5_PREFIX = "query: "  # e5 zorunlu prefix (tek-cumle gomme)
EMB_DIM = 1024         # multilingual-e5-large gizli boyut
BATCH_SIZE = 8         # CPU mini-batch (kucuk -> aktivasyon RAM'i dusuk; deterministik, sira korunur)
CHUNK_SIZE = 1000      # encode'u parcalara bol + her parca sonrasi belleği serbest birak (OOM korumasi)


def _prefix(texts) -> list[str]:
    """Her metne e5 'query: ' prefix'i ekle (None/NaN -> bos string güvenli)."""
    return [E5_PREFIX + ("" if t is None else str(t)) for t in texts]


def _load_model():
    """e5-large'i yukle. Yerel models/e5-large varsa ORADAN; yoksa HF'den indir + yerel kaydet.

    CPU + eval + torch.manual_seed(42) -> deterministik. Yerel kayit -> sonraki kosular internet
    GEREKTIRMEZ (reproducibility; .npy cache zaten kanonik, bu yalniz model indirme tekrarini onler).
    """
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(cv.SEED)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    # HF cache'inden yukle (models/hf_cache, .gitignore'da). Cache yoksa SentenceTransformer otomatik
    # indirir + cache'ler -> sonraki kosular internet GEREKTIRMEZ. Ayri .save() YAPILMAZ: kanonik
    # reproducible artefakt zaten .npy cache; 2.2GB duplike yerel-kayit gereksiz (ve yarim-kayit
    # onceki kosuda load-from-local'i bozmustu) -> tek kaynak HF cache.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    src = "yerel HF cache" if (HF_CACHE_DIR / f"models--{HF_MODEL_ID.replace('/', '--')}").exists() else "HF (indirme)"
    print(f"[bert_embed] model yukleniyor ({src}): {HF_MODEL_ID}  cache={HF_CACHE_DIR}", flush=True)
    model = SentenceTransformer(HF_MODEL_ID, device="cpu", cache_folder=str(HF_CACHE_DIR))

    model.eval()
    return model


def _encode(model, texts, tag: str = "") -> np.ndarray:
    """query-prefix'li metinleri 1024-dim birim-norm float32 embedding'e cevirir (deterministik).

    BELLEK GUVENLI: CHUNK_SIZE'lik parcalara bolerek encode + her parca sonrasi GC. Boylece pik RAM
    = model + tek-parca aktivasyonu (10k'yi tek seferde encode -> OOM sessiz-kill, bu makinede ~4GB
    bos). Parcalama SIRAYI korur (giris sirasiyla ayni) -> satir-hizali + deterministik.
    """
    import gc

    import torch

    prefixed = _prefix(texts)
    n = len(prefixed)
    out = np.empty((n, EMB_DIM), dtype=np.float32)  # onceden ayrilmis (parca buyumesi yok)
    for start in range(0, n, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, n)
        with torch.no_grad():
            chunk = model.encode(
                prefixed[start:end],
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,   # birim-norm (cosine ~ dot)
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        out[start:end] = np.asarray(chunk, dtype=np.float32)
        del chunk
        gc.collect()
        print(f"[bert_embed]   {tag} encode {end}/{n}", flush=True)
    assert out.ndim == 2 and out.shape[1] == EMB_DIM, f"beklenen (N,{EMB_DIM}), gelen {out.shape}"
    assert np.isfinite(out).all(), "embedding'de NaN/Inf var."
    return np.ascontiguousarray(out)


def main() -> None:
    force = "--force" in sys.argv
    cv.set_seed()  # PYTHONHASHSEED + tek-thread BLAS/OpenMP (determinizm)
    cv.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if EMB_TRAIN_PATH.exists() and EMB_TEST_PATH.exists() and not force:
        a = np.load(EMB_TRAIN_PATH)
        b = np.load(EMB_TEST_PATH)
        print(f"[bert_embed] cache MEVCUT (atlaniyor): emb_train{a.shape} emb_test{b.shape}. "
              f"Yeniden uretmek icin --force.")
        return

    train = cv.load_train()
    test = cv.load_test()
    print(f"[bert_embed] metin: train={len(train)} test={len(test)}  model={HF_MODEL_ID}")

    model = _load_model()

    # Train'i encode + HEMEN kaydet (kismi ilerleme OOM'da bile korunur), sonra serbest birak.
    print("[bert_embed] train embedding cikariliyor ...", flush=True)
    emb_tr = _encode(model, train[cv.TEXT_COL].values, tag="train")
    assert emb_tr.shape == (len(train), EMB_DIM), emb_tr.shape
    np.save(EMB_TRAIN_PATH, emb_tr)
    tr_norm = float(np.linalg.norm(emb_tr, axis=1).mean())
    print(f"[bert_embed] YAZILDI: {EMB_TRAIN_PATH.name}{emb_tr.shape}  mean||v||={tr_norm:.4f}", flush=True)
    del emb_tr
    import gc as _gc
    _gc.collect()

    print("[bert_embed] test embedding cikariliyor ...", flush=True)
    emb_te = _encode(model, test[cv.TEXT_COL].values, tag="test")
    assert emb_te.shape == (len(test), EMB_DIM), emb_te.shape
    np.save(EMB_TEST_PATH, emb_te)
    te_norm = float(np.linalg.norm(emb_te, axis=1).mean())
    print(f"[bert_embed] YAZILDI: {EMB_TEST_PATH.name}{emb_te.shape}  mean||v||={te_norm:.4f}", flush=True)
    print(f"[bert_embed] BITTI. norm kontrol (birim olmali): train={tr_norm:.4f}  test={te_norm:.4f}", flush=True)


if __name__ == "__main__":
    main()
