import pandas as pd
import matplotlib.pyplot as plt
from jiwer import wer, cer

df = pd.read_csv('results/base_results.csv')
wers = []
cers = []

for row in df.itertuples():
    ref = str(row.reference).lower().strip()
    hyp = str(row.prediction).lower().strip()
    
    wers.append(wer(ref, hyp))
    cers.append(cer(ref, hyp))
df['wer'] = wers
df['cer'] = cers


avgwer = sum(wers) / len(wers)
avgcer = sum(cers) / len(cers)

df_sorted = df.sort_values('wer')

#best worst
best_5  = df_sorted.head(5)
worst_5 = df_sorted.tail(5)

best_5.to_csv('results/best_samples.csv', index=False)
worst_5.to_csv('results/worst_samples.csv', index=False)
df_sorted.to_csv('results/wer_cer_table.csv', index=False)

print(f'Average WER: {round(avgwer * 100, 2)}%')
print(f'Average CER: {round(avgcer * 100, 2)}%')

print('Best 5')
print(best_5[['reference', 'prediction', 'wer', 'cer']].to_string(index=False))

print('Worst 5')
print(worst_5[['reference', 'prediction', 'wer', 'cer']].to_string(index=False))

metrics_df = pd.DataFrame({
    'Metric': ['WER', 'CER'],
    'Average': [f'{round(avgwer * 100, 2)}%',
                f'{round(avgcer * 100, 2)}%'
]})

metrics_df.to_csv('results/base_metrics.csv', index=False)


with open('results/base_metrics.txt', 'w') as f:
    f.write(f'Average WER: {round(avgwer * 100, 2)}%\n')
    f.write(f'Average CER: {round(avgcer * 100, 2)}%\n')


#vizualizations

plt.figure()
plt.plot(df_sorted['wer'].values, label='WER')
plt.plot(df_sorted['cer'].values, label='CER')
plt.xlabel('Sample')
plt.ylabel('Score')
plt.title('WER and CER per sample')
plt.legend()
plt.tight_layout()
plt.savefig('results/wer_cer_plot.png')
plt.show()


plt.figure()
plt.bar(['WER', 'CER'], [avgwer * 100, avgcer * 100])
plt.ylabel('%')
plt.title('Average WER and CER')
plt.tight_layout()
plt.savefig('results/avg_wer_cer_bar.png')
plt.show()