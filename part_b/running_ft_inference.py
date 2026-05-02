import whisper
import pandas as pd
import os

model = whisper.load_model("large-v2")
data_path = "data"
clips_path = os.path.join(data_path, "clips")
tsv_file = os.path.join(data_path, "test.tsv") 
output_file = "results/part_b/ft_results.csv"

df_test = pd.read_csv(tsv_file, sep="\t")
results = []
count = 0

for row in df_test.itertuples():
    audio_path = os.path.join(clips_path, row.path)
    try:
        result = model.transcribe(audio_path, language="az", initial_prompt="Azərbaycan dili.")
        
        results.append({
            "reference": row.sentence,
            "prediction": result["text"]
        }) 
        
        count += 1
        if count % 10 == 0:
            print(f"Processed {count}")
            
    except Exception as e:
        print(f"Error {e}")
        continue

os.makedirs("results/part_b", exist_ok=True)
results_df = pd.DataFrame(results)
results_df.to_csv(output_file, index=False, encoding="utf-8")