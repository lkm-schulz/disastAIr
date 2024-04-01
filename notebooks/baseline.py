from markov import ConfusionMatrix
import pandas as pd
from paths import PATH_DATA_TEST_LABELED, PATH_DATA_TRAIN, DIR_DATA
from functools import partial
import random

NUM_RUNS = 1000

def random_classify(instance):
    return random.choice([True, False])

def constant_classify(label, instance):
    return label

def eval(instances, labels, classify: callable) -> ConfusionMatrix:
    res = ConfusionMatrix()
    
    for i in range(len(instances)):
        if classify(instances[i]):
            if labels[i]:
                res.tp += 1
            else:
                res.fp += 1
        else:
            if labels[i]:
                res.fn += 1
            else:
                res.tn += 1

    return res


def main():
    df_test = pd.read_csv(PATH_DATA_TEST_LABELED)

    random.seed(10)
    cm_random = ConfusionMatrix()
    for i in range(NUM_RUNS):
        cm_random += eval(df_test["text"], df_test["target"], random_classify)
    cm_random /= NUM_RUNS
    cm_constant_true = eval(df_test["text"], df_test["target"], partial(constant_classify, True))
    cm_constant_false = eval(df_test["text"], df_test["target"], partial(constant_classify, False))

    print(cm_random)
    print(cm_constant_true)
    print(cm_constant_false)

if __name__ == '__main__':
    main()