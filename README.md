# Azərbaycan Dili üçün Nitq Tanıma Sistemi (ASR)

Bu layihə **R.I.S.K. Company AI Engineer Intern** tapşırığı çərçivəsində hazırlanmışdır. Layihənin məqsədi Mozilla Common Voice datasetindən istifadə edərək Azərbaycan dili üçün Avtomatik Nitq Tanıma (ASR) pipeline-i qurmaq və optimallaşdırmaqdır.

## Layihə Strukturu

* **part_a**: ASR baza tətbiqi (baseline). Modelin ilkin vəziyyətdə performansı  
* **part_b**: Təkmilləşdirilmiş sistem (prompting və mətn təmizləmə / normalization)  
* **results**: Metrikalar (WER/CER), nəticələr və qrafiklər  
* **report.pdf**: Analitik hesabat (Hissə C)  
* **requirements.txt**: Lazım olan kitabxanalar  

## Texnologiyalar və Model

* **Model:** OpenAI Whisper `large-v2`  
* **Dataset:** Mozilla Common Voice 17.0 (Azərbaycan dili)  
* **Metrikalar:** Word Error Rate (WER) və Character Error Rate (CER)  
* **Kitabxanalar:** `openai-whisper`, `jiwer`, `pandas`, `matplotlib`  

## Nəticələrin Müqayisəsi

| Metrika          | Part A (Baseline) | Part B (Improved) | Fərq       |
|------------------|-------------------|-------------------|------------|
| **Ortalama WER** | 32.71%            | 31.28%            | **-1.43%** |
| **Ortalama CER** | 8.73%             | 8.31%             | **-0.42%** |

**Part B-də yaxşılaşma inference-time optimizasiya üsulları ilə əldə edilmişdir (initial_prompt və mətn təmizləmə).**

## Quraşdırma və İşə Salma

### 1. Kitabxanaları yüklə

```bash
pip install -r requirements.txt
```

### 2. Baza modeli işə sal (Part A)

```bash id="az2"
python part_a/run_inference.py
python part_a/evaluate.py
```

### 3. Təkmilləşdirilmiş pipeline-i işə sal (Part B)

```bash id="az3"
python part_b/train.py
python part_b/run_ft_inference.py
python part_b/plot_metrics.py
```
