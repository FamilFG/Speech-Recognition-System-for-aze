# Azerbaycan Dili ucun Nitq Tanima Sistemi (ASR)

Bu layihe **R.I.S.K. Company AI Engineer Intern** tapsirigi cercevesinde hazirlanmisdir. Layihe nin meqsedi Mozilla Common Voice datasetinden istifade ederek Azerbaycan dili ucun Avtomatik Nitq Tanima (ASR) pipeline-i qurmaq ve optimallasdirmaqdir.

## Layihe Strukturu

* **part_a**: ASR baza tetbiqi (baseline). Modelin ilkin veziyyetde performansi
* **part_b**: Tekmilleshdirilmis sistem (prompting ve metn temizleme / normalization)
* **results**: Metrikalar (WER/CER), neticeler ve qrafikler
* **report.pdf**: Analitik hesabat (Hisse C)
* **requirements.txt**: Lazim olan kitabxanalar

## Texnologiyalar ve Model

* **Model:** OpenAI Whisper `small`
* **Dataset:** Mozilla Common Voice 17.0 (Azerbaycan dili)
* **Metrikalar:** Word Error Rate (WER) ve Character Error Rate (CER)
* **Kitabxanalar:** `openai-whisper`, `jiwer`, `pandas`, `matplotlib`

## Neticelerin Muqayisesi

| Metrika          | Part A (Baseline) | Part B (Improved) | Ferq       |
| ---------------- | ----------------- | ----------------- | ---------- |
| **Ortalama WER** | 32.71%            | 31.28%            | **-1.43%** |
| **Ortalama CER** | 8.73%             | 8.31%             | **-0.42%** |

**Part B-de yaxsilasma inference-time optimizasiya usullari ile elde edilmisdir (initial_prompt ve metn temizleme).**

## Qurasdirma ve Ise Salma

### 1. Kitabxanalari yukle

```bash id="az1"
pip install -r requirements.txt
```

### 2. Baza modeli ise sal (Part A)

```bash id="az2"
python part_a/run_inference.py
python part_a/evaluate.py
```

### 3. Tekmilleshdirilmis pipeline-i ise sal (Part B)

```bash id="az3"
python part_b/train.py
python part_b/run_ft_inference.py
python part_b/plot_metrics.py
```
