import sys
import re
import numpy as np
from graphcnn.helper import *

def run(filename):

    results = {}
    with open(filename, 'r') as file:
        for line in file:
            try:
                split = line[:-1].split('\t')
                
                no_folds = int(split[2][:-5])
                name = split[1]
                
                time = int(split[3][:-8])
                
                m = re.match('(.+) \\(\+\- (.+)\\)', split[4])
                acc = float(m.group(1))
                std = float(m.group(2))
                
                if (name, no_folds) not in results:
                    results[(name, no_folds)] = []
                results[(name, no_folds)].append((acc, std, time))
            except:
                pass
            
    for key, value in results.items():
        print_ext('Model name:', key[0])
        print_ext('Number of folds:', key[1])
        for i in range(len(value)):
            print_ext('Experiment', i+1, value[i][0], '+-', value[i][1], 'in', value[i][2], 'seconds')
            
        acc = np.mean(value, axis=0)
        print_ext('Summary:', acc[0], '+-', acc[1], 'in', acc[2], 'seconds\n')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_ext('Usage: %s RESULT_FILENAME' % sys.argv[0])
    else:
        run(sys.argv[1])