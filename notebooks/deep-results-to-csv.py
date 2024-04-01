import os

KERAS_RESULTS_FILE = "results-keras"
KEYWORDS = ["batch_size", "epochs", "Total", "TP", "TN", "FP", "FN", "Prc", "Rec", "F1"]

with open(os.path.join(".", KERAS_RESULTS_FILE + ".csv"), "w") as fout:
    for i in range(len(KEYWORDS)):
        fout.write(KEYWORDS[i])
        if i != len(KEYWORDS) - 1:
            fout.write(",")
        else:
            fout.write('\n')
    
    with open(os.path.join(".", KERAS_RESULTS_FILE + ".txt"), "r") as fin:
        file = fin.readlines()
        for line in file:
            if len(line) < 10:
                continue
            results = line.split(",")
            for i in range(len(results)):
                result = results[i][results[i].find(":")+2:]                
                print_comma = True

                fout.write(result)

                if i != len(results) - 1:
                    fout.write(",")
                
                
