from markov import Classifier
import pandas as pd
from paths import PATH_DATA_TEST_LABELED, PATH_DATA_TRAIN, DIR_DATA
import time
import regex as re

NUM_RUNS = 5

def main():
    df_train = pd.read_csv(PATH_DATA_TRAIN)
    df_test = pd.read_csv(PATH_DATA_TEST_LABELED)

    test_train(df_train, get_wordlist, 0, 'word-based')
    test_train(df_train, lambda x: x, 6, 'char-based')
    test_train(df_train, lambda x: x.lower(), 7, 'char-based (lowercased)')

    test_classify(df_train, df_test, get_wordlist, 0, 'word-based')
    test_classify(df_train, df_test, lambda x: x, 6, 'char-based')
    test_classify(df_train, df_test, lambda x: x.lower(), 7, 'char-based (lowercased)')

    return 0

def test_train(data, mapping, order, name):
    times_train = [0] * NUM_RUNS

    for i in range(NUM_RUNS):
        print(f'Train {i + 1}', end='')
        time = train(list(map(mapping, data['text'])), data['target'], order, False, False)
        print(f' - Time: {time / 1000 / 1000 / 1000:.3f}s')
        times_train[i] = time
    times_train_avg = sum(times_train) / len(times_train)
    print(f'[{name}, order: {order}] Avg training time (total): {times_train_avg / 1000 / 1000 / 1000:.3f}s')


def test_classify(train, test, mapping, order, name):

    times_classify = [0] * NUM_RUNS
    classifier = Classifier(order, list(map(mapping, train['text'])), train['target'], multithread=False, verbose=False)
    print(classifier.eval(list(map(mapping, test['text'])), test['target']))

    for i in range(NUM_RUNS):
        print(f'Classify {i + 1}', end='')
        time = classify(classifier, list(map(mapping, test['text'])))
        print(f' - Time: {time / 1000 / 1000 / 1000:.3f}s')
        times_classify[i] = time

    times_classify_avg = sum(times_classify) / len(times_classify)
    print(f'[{name}, order: {order}] Avg classification time (total): {times_classify_avg / 1000 / 1000 / 1000:.3f}s')
    print(f'[{name}, order: {order}] Avg classification time (per Instance): {times_classify_avg / len(test) / 1000:3f}μs')


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

def get_wordlist(text: str) -> list[str]:
    cleaned = text.lower()
    cleaned = re.sub(r'https?:\/\/.*[\r\n]*|[^\w\s]', ' ', cleaned)
    return cleaned.split()

if __name__ == '__main__':
    print('Starting main')
    main()