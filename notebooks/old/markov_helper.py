from collections import defaultdict
from multiprocessing import Value
import time

def count_ngrams(sequences: list, n, progress) -> tuple[int, dict[tuple, int]]:
    total = 0
    frequencies = defaultdict(int)
    
    if (len(sequences) > 0):
        i = 0
        for seq in sequences:
            i += 1

            for pos in range(len(seq) - n):
                ngram = seq[pos:pos+n+1]
                key = (ngram[0],)
                for item in ngram[1:]:
                    key += (item,)
                frequencies[key] += 1
                total += 1

            if i % 10 == 0:
                progress.value = i

    progress.value = len(sequences)

    return (total, frequencies)

def progress_monitor(progresses, num_workers, num_sequences):
    while True:
        completed = sum(progresses[i].value for i in range(num_workers))
        progress = completed / (num_sequences * num_workers) * 100
        print(f'Counting NGrams for Markov Model: {progress:>5.2f}%', end='\r')

        if completed >= num_sequences * num_workers:
            print
            break
        time.sleep(0.25)
