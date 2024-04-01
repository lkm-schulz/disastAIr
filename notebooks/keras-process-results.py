import pandas as pd
import numpy as np

RESULTS_FILE = "results-keras-test.txt"

df = pd.read_csv(RESULTS_FILE)

df_sd = df.groupby(['batch_size', 'epochs']).agg(np.std).sort_values(['epochs', 'batch_size'])
df_var = df.groupby(['batch_size', 'epochs']).agg(np.var).sort_values(['epochs', 'batch_size'])
df = df.groupby(['batch_size', 'epochs']).mean().sort_values(['epochs', 'batch_size'])

df_sd.to_csv("results-keras-test-sd.csv")
df_var.to_csv("results-keras-test-var.csv")
df.to_csv("results-keras-test-averaged.csv")

for epochs in [1,2,8,16,4]:
    df.loc[(slice(None), epochs), :].to_csv("deep-keras-test-" + str(epochs) + "epochs.csv")