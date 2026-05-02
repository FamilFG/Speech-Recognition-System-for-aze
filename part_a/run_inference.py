import whisper
import csv
import os

modelsize = 'large-v2' #i have a good gpu so i used this one, but medium also gave a good enough result
language = 'az'
clipspath = 'data/clips'
tsvfile = 'data/test.tsv'
outputpath = 'results/base_results.csv'

model = whisper.load_model(modelsize)

with open(tsvfile, encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    rows = list(reader)

results = []

for i, row in enumerate(rows):
    print(f'{i+1} / {len(rows)} is processing')

    audio_filename = row['path']
    audio_path = os.path.join(clipspath, audio_filename)
    reference = row['sentence']

    try:
        result = model.transcribe(audio_path, language=language)
        my_prediction = result['text'].strip()
        results.append({'reference': reference, 'prediction': my_prediction})
    except Exception as e:
        print(f'Error: {e}')

os.makedirs('results', exist_ok=True)

with open(outputpath, 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['reference', 'prediction'])
    writer.writeheader()
    writer.writerows(results)