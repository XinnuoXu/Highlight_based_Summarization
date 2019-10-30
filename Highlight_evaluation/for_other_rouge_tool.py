#coding=utf8

def get_file_rouge(ref_dir, hyp_dir):
    ref_map = {}; hyp_map = {}
    for filename in os.listdir(ref_dir):
        file_id = filename.split(".")[0]
        with open(ref_dir + filename, 'r') as file:
            ref_map[file_id] = file.read().strip()
    for filename in os.listdir(hyp_dir):
        file_id = filename.split(".")[0]
        with open(hyp_dir + filename, 'r') as file:
            hyp_map[file_id] = file.read().strip()
    fpout_ref = open("tmp.ref", "w")
    fpout_hyp = open("tmp.hyp", "w")
    for file_id in ref_map:
        fpout_ref.write(ref_map[file_id] + "\n")
        fpout_hyp.write(hyp_map[file_id] + "\n")
    fpout_ref.close()
    fpout_hyp.close()

if __name__ == '__main__':
    import sys, os
    ref_dir = "./50_docs/"
    if sys.argv[1] == "TConv":
        hyp_dir = '../HROUGE_data/summaries/system_tconvs2s/'
    elif sys.argv[1] == "PT":
        hyp_dir = '../HROUGE_data/summaries/system_ptgen/'
    elif  sys.argv[1] == "Ref":
        hyp_dir = '../HROUGE_data/summaries/ref_gold/'
    get_file_rouge(ref_dir, hyp_dir)

