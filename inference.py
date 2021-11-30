import sys
import getopt

datapath = ''
out_csv_path = ''

argvs = sys.argv[1:]
try:
    opts, args = getopt.getopt(argvs, "", ["data=", "output="])
    for opt, arg in opts:
        if opt == '--data':
            datapath = arg
        elif opt == '--output':
            out_csv_path = arg
except getopt.GetoptError:
    print('train.py --data <data_path> ')
    sys.exit(2)
print(f'datapath{datapath} out_csv_path{out_csv_path}')
