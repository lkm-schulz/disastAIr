import multiprocessing as mp

DEF_SMOOTHING_WEIGHT = .1
DEF_VOCAB_SIZE = 95
MULTITHREAD_NUM_WORKERS = max(1, mp.cpu_count() - 2)