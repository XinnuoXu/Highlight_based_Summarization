#coding=utf8

if __name__ == '__main__':
    import sys, os
    ids = []
    for filename in os.listdir("../HROUGE_data/documents/"):
        ids.append(filename.split(".")[0])
    dir_path = "../Highlight_evaluation/50_trees/"
    fpout_src = open("./test_50.src", "w")
    fpout_tgt = open("./test_50.tgt", "w")
    for file_id in ids:
        src_path = dir_path + file_id + ".src"
        tgt_path = dir_path + file_id + ".tgt"
        with open(src_path, 'r') as file:
            line = file.read().strip()
            fpout_src.write(line + "\n")
        with open(tgt_path, 'r') as file:
            line = file.read().strip()
            fpout_tgt.write(line + "\n")
    fpout_src.close()
    fpout_tgt.close()
