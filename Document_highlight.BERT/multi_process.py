import os, sys

def highlight_score_split(label):
    for filename in os.listdir('./tmp_data'):
        if filename.find("_src") == -1:
            continue
        filename = filename.split('.')[0]
        flist = filename.split('_')
        if flist[2] == label:
            flabel = flist[2] + '_' + flist[3]
            os.system("nohup python quality_classifier.py " + flabel + " &")

if __name__ == '__main__':
    highlight_score_split(sys.argv[1])
