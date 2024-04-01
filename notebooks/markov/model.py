from collections import defaultdict
import math
import time
import multiprocessing as mp
import decimal
import os
from .settings import DEF_SMOOTHING_WEIGHT, DEF_VOCAB_SIZE, MULTITHREAD_NUM_WORKERS
from .helpers import move_and_clear_line, print_over

class Model:
    order: int
    ngram_freqs: list[dict]
    totals: list[int]
    vocab_size: int
    __smoothing_weight__: float
    __smoothing_normalizer__: int
    __smoothing_weight_norm__: int
    __type__ = None

    def __init__(self, order: int, sequences: list = [], multithread: bool = True, verbose: bool = False, 
                 vocab_size: int = DEF_VOCAB_SIZE, smoothing_weight: float = DEF_SMOOTHING_WEIGHT):
        assert(order >= 0)

        self.order = order
        self.ngram_freqs = [defaultdict(int)] * (order + 1)
        self.totals = [0] * (order + 1)
        self.vocab_size = vocab_size
        self.set_smoothing_weight(smoothing_weight)

        if (len(sequences) > 0):
            if isinstance(sequences[0], str):
                self.__type__ = str
            elif isinstance(sequences[0], list):
                self.__type__ = list
            else:
                raise TypeError(f'sequences argument must be either a list of strings or list of lists (is list[{type(sequences[0])}]).')

            if verbose:
                print()

            start = time.time()

            if (multithread):
                if verbose:
                    print_over(f'Allocating resources for {MULTITHREAD_NUM_WORKERS} workers...')
                manager = mp.Manager()
                pool = mp.Pool(processes=MULTITHREAD_NUM_WORKERS)

                processes = [None] * (order + 1)  
                progresses = [None] * (order + 1)

                if verbose:
                    move_and_clear_line()
                    print(f'Launching tasks for worker processes...')

                try:
                    for n in range(order + 1):
                        progresses[n] = manager.Value('i', 0)
                        process = pool.apply_async(Model.__count_ngrams__, args=(sequences, n, progresses[n]))
                        processes[n] = process   

                    if verbose:
                        monitor = mp.Process(target=Model.__progress_monitor__, args=(progresses, order + 1, len(sequences)))
                        monitor.start()
                    
                    for n in range(len(processes)):
                        (total, frequencies) = processes[n].get()
                        self.totals[n] = total
                        self.ngram_freqs[n] = frequencies
                    
                finally:
                    pool.terminate()
                    if verbose:
                        monitor.terminate()
                    
            else:
                i = 0
                for seq in sequences:
                    i += 1
                    if verbose and i % 10 == 0:
                        print_over(f'Counting NGrams for Markov Model: {(i / len(sequences) * 100):>5.2f}%')
                    for n in range(0, order + 1):
                        for pos in range(len(seq) - n):
                            ngram = seq[pos:pos+n+1]
                            self.add_ngram_occurrence(ngram)

            duration = time.time() - start
            if verbose:
                print_over(f'Order {order} Markov Model of {len(sequences)} sequences built in {duration:.3f} seconds!')

    def __str__(self):
        return f'{self.order}-Order Markov NGram Object - Totals: {self.totals}'
    
    def __check_len__(self, l):
        if l > self.order + 1 or l == 0:
            raise ValueError(f'Invalid NGram - Length of ngram must be between 1 and {self.order + 1} for an order {self.order} NGrams object (is {l})')

    @classmethod
    def __tuple_from_list__(cls, l: list):
        t = (l[0],)
        for item in l[1:]:
            t += (item,)

        return t
    
    @classmethod
    def __count_ngrams__(cls, sequences: list, n, progress) -> tuple[int, dict[tuple, int]]:
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
    
    @classmethod
    def __progress_monitor__(cls, progresses, num_workers, num_sequences):
        while True:
            completed = sum(progresses[i].value for i in range(num_workers))
            progress = completed / (num_sequences * num_workers) * 100
            print_over(f'Counting NGrams for Markov Model: {progress:>5.2f}%')

            if completed >= num_sequences * num_workers:
                print
                break
            time.sleep(0.25)
    
    def __check_type__(self, a):
        if not isinstance(a, self.__type__):
            raise TypeError(f'Argument must be of type {self.__type__} (is {type(a)})')
        
    def set_smoothing_weight(self, weight):
        self.__smoothing_weight__ = weight
        self.__smoothing_normalizer__ = (10 ** (-1 * decimal.Decimal(str(self.__smoothing_weight__)).as_tuple().exponent))
        self.__smoothing_weight_norm__ = int(self.__smoothing_normalizer__ * self.__smoothing_weight__)
        # used to convert smoothing weight to int for large numbers
        assert(self.__smoothing_weight_norm__ == (self.__smoothing_normalizer__ * self.__smoothing_weight__))

    def add_ngram_occurrence(self, ngram):
        self.__check_len__(len(ngram))
        self.__check_type__(ngram)
        self.ngram_freqs[len(ngram) - 1][Model.__tuple_from_list__(ngram)] += 1
        self.totals[len(ngram) - 1] += 1

    def get_total_occurrences(self, ngram):
        self.__check_len__(len(ngram))
        self.__check_type__(ngram)
        return self.ngram_freqs[len(ngram) - 1][Model.__tuple_from_list__(ngram)]
    
    def get_log_cond_prob(self, ngram):
        self.__check_len__(len(ngram))
        self.__check_type__(ngram)

        top = math.log((self.__smoothing_weight__ + self.get_total_occurrences(ngram)))
        occurrences_bot = self.totals[0] if (len(ngram) == 1) else self.get_total_occurrences(ngram[:-1])
        smoothing_bot = self.__smoothing_weight_norm__ * (self.vocab_size ** (max(len(ngram) - 1, 1)))
        bot = math.log(self.__smoothing_normalizer__) - math.log(smoothing_bot + self.__smoothing_normalizer__ * occurrences_bot) 

        return top + bot
    
    def get_log_sum_prob(self, seq: str):
        # TODO: how to handle empty sequences (high or low value?)
        # if (len(seq) == 0):
        #     raise ValueError(f'Invalid Sequence - Cannot get probability for an empty list')

        start = 0
        end = 1
        sum = 0

        while (end <= len(seq)):
            sum += self.get_log_cond_prob(seq[start:end])
            end += 1

            # ngrams can be at most order + 1 characters long:
            if (end - start > self.order + 1):
                start += 1
        return sum

