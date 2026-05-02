import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer
import os
import re

results_file = "results/part_b/ft_results.csv"
log_file = "results/part_b/fine_tuning_log.csv"

def clean(text):
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text) 
    return " ".join(text.split())


if os.path.exists(results_file):
    df = pd.read_csv(results_file)
    wers = []
    cers = []
    
    for row in df.itertuples():
        ref = clean(row.reference)
        pred = clean(row.prediction)
        
        if not ref:
            continue
            
        wers.append(wer(ref, pred))
        cers.append(cer(ref, pred))
    
    avg_wer = sum(wers) / len(wers)
    avg_cer = sum(cers) / len(cers)
    
    os.makedirs("results/part_b", exist_ok=True)
    with open("results/part_b/ft_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Avg WER: {round(avg_wer, 4)}\n")
        f.write(f"Avg CER: {round(avg_cer, 4)}")
    
    print(f"Metrics saved, WER: {avg_wer:.2%}, CER: {avg_cer:.2%}")

if os.path.exists(log_file):
    log_df = pd.read_csv(log_file)
    os.makedirs("results/part_b/plots", exist_ok=True)

    plt.figure()
    plt.plot(log_df["epoch"], log_df["train_loss"], label='Train Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("results/part_b/plots/loss.png")
    
    plt.figure()
    plt.plot(log_df["epoch"], log_df["val_wer"], label='Val WER')
    plt.title("Validation WER")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.legend()
    plt.savefig("results/part_b/plots/wer.png")
    plt.show()