from markov import Classifier
import pandas as pd
from paths import PATH_DATA_TEST_LABELED, PATH_DATA_TRAIN, DIR_DATA
import time

NUM_RUNS = 5
ORDER = 6

def main():
    df_train = pd.read_csv(PATH_DATA_TRAIN)
    df_test = pd.read_csv(PATH_DATA_TEST_LABELED)

    times_train = [0] * NUM_RUNS
    times_classify = [0] * NUM_RUNS

    for i in range(NUM_RUNS):
        print(f'Train {i + 1}', end='')
        time = train(df_train["text"], df_train["target"], ORDER, False, False)
        print(f' - Time: {time / 1000 / 1000 / 1000:.3f}s')
        times_train[i] = time


    classifier = Classifier(ORDER, df_train["text"], df_train["target"], multithread=False, verbose=False)

    for i in range(NUM_RUNS):
        print(f'Classify {i + 1}', end='')
        time = classify(classifier, df_test["text"])
        print(f' - Time: {time / 1000 / 1000 / 1000:.3f}s')
        times_classify[i] = time

    return 0

def train(instances, labels, order, multithread, verbose):
    start = time.process_time_ns()
    classifier = Classifier(order, instances, labels, multithread=multithread, verbose=verbose)
    end = time.process_time_ns()
    return end - start

def classify(classifier, instances):
    start = time.process_time_ns()
    for seq in instances:
        classifier.classify(seq) 
    end = time.process_time_ns()
    return (end - start)

if __name__ == '__main__':
    print('Starting main')
    main()