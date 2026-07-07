# DIRILIS BIRLESTIRME — Sonuc (blend_dirilis)

KARAR METRIGI = nested recency-weighted OOF MSE (rw-OOF). Overfit kapisi YOK (kullanici karari): en ufak nested dusus bile alinir.

- ham 12-model blend nested rw-OOF = **82.239761**
- birlesik final (blend_dirilis) nested rw-OOF = **82.174541**
- TOPLAM delta = **-0.065220**
- uygulanan adim sirasi = `['rv_ftshaped(base-col)', 'pp_xlmrgate', 'pp_winsor', 'rv_sigmaclip']`
- final unweighted CV = 72.7584 (std 2.7186)

## 1) Adaylarin guncel-tabandaki deltasi (eski vs yeni)

| aday | tip | eski delta (eski taban) | yeni delta (guncel taban) | sonuc |
|---|---|---|---|---|
| rv_ftshaped | base-kolon (13-model ridge_pos) | -0.010936 @blend | -0.010936 (ALINDI) | ALINDI |
| rv_sigmaclip | post-process (E[clip(N(mu,c*sigma))]) | -0.002601 @ppstack | -0.002521 (ALINDI) | ALINDI |

## 2) Birlesik zincir (kumulatif)

| adim | bilesem | step-delta | kumulatif rw | nested/frozen info |
|---|---|---|---|---|
| 1 | rv_ftshaped(base-col) | -0.010936 | 82.228825 | 13-model ridge_pos; ftshaped agirlik=0.0006; standalone_rw=87.0283 |
| 2 | pp_xlmrgate | -0.049024 | 82.179801 | a_mode=0.3 a_dist={0.2: 1, 0.3: 9, 0.4: 1, 0.5: 4} q_thr_med=17.326; frozen q_thr=21.125 a=0.5 |
| 3 | pp_winsor | -0.002738 | 82.177062 | p=0.0%:1/15, p=0.1%:3/15, p=0.25%:11/15; frozen p=0.25% (lo=43.3171) |
| 4 | rv_sigmaclip | -0.002521 | 82.174541 | nested c-pick=[c=0.3:15/15]; frozen_c=0.3 |

rv_ftshaped standalone base rw = 87.0283 (12-model blend uyelerinden zayif, ama ortogonal sinyal blend'e net kolon-katkisi yapabilir).

## 3) DURUST YORUM

- **Kapisiz secim:** Bu birlesik zincir overfit kapisindan (0.25*cv_std) GECMEDI; kullanicinin acik talimati uzerine kapisiz, en-ufak-nested-dusus mantigiyla kuruldu. Her bilesen NESTED (hucre-disindan fit) -> in-sample iyimserlik elimine edildi; AMA delta'lar cv_std'nin (~3 MSE) cok altinda ve paired/blocked-bootstrap ile 'anlamli' DEGIL.
- **Private-belirsiz:** Bu mikro-kazanclarin isaretinin private'da korunacagi GARANTI DEGIL. Kazanc rastgele CV-gurultusu seviyesinde; private %40 bolmesinde ters donebilir.
- **Ne kadari gercek mekanizma vs meta-overfit:**
  * rv_ftshaped (base-kolon): t-shaped teknik-skor std/mean/range gercek bir FE sinyali; ama 12-model blend (xlmr/ourteam_tf/GBDT'ler) bu profil-tutarliligini buyuk olcude zaten yakaliyor -> katki ya mikroskobik ya redundant. ridge_pos agirligi/delta bunu gosterir.
  * rv_sigmaclip (post-process): E[clip(N(mu,c*sigma))] censored-normal ust/alt kutleye analitik kalibrasyon; MEKANIZMA gecerli AMA xlmr-gate + winsor zaten ust-kuyrugu ittikten sonra ek kazanc cogunlukla mikroskobik (nested c-grid secim gurultusunden ayirt edilemez).
- **SUB onerisi:** blend_dirilis'i ancak base blend ile yapisal es-degerli (ayni govde + ince postproc/kolon) gorup private risk dagitiminda DIKKATLE kullan. SUB-1 (sade catboost_full) AYRI/sade tutulmali; bu zincir SUB-2 (ensemble) kuyrugu icindir.
