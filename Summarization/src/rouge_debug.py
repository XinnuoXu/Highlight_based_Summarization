import argparse
import os
import time
# from multiprocess import Pool as Pool2
from multiprocessing import Pool

import shutil
import sys
import codecs
import json

# from onmt.utils.logging import init_logger, logger
from others import pyrouge


def process(data):
    candidates, references, pool_id = data
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = "rouge-tmp-{}-{}".format(current_time,pool_id)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155()
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def test_rouge_single(cand, ref, src, num_processes):
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]
    srcs = [line.strip() for line in src]
    assert len(candidates) == len(references)

    n_pool = num_processes
    pool = Pool(n_pool)

    fpout_log = open("../logs/rouge_in_details.json", "w")
    selected_tags = ["rouge_1_f_score", "rouge_2_f_score", "rouge_l_f_score"]
    final_results = {}
    for i in range(len(candidates)):
        arg = [([candidates[i]],[references[i]],i)]
        results = pool.map(process,arg)
        json_obj = {}
        json_obj["cand"] = candidates[i]
        json_obj["ref"] = references[i]
        json_obj["src"] = srcs[i]
        for j,r in enumerate(results):
            for k in r:
                if str(k) in selected_tags:
                    json_obj[str(k)] = r[k]
                if(k not in final_results):
                    final_results[k] = r[k]
                else:
                    final_results[k] += r[k]
            fpout_log.write(json.dumps(json_obj) + "\n")
    fpout_log.close()

    for k in final_results:
        final_results[k] = final_results[k]/ len(candidates)

    return final_results

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        # results_dict["rouge_3_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
    results_dict["rouge_1_recall"] * 100,
    results_dict["rouge_2_recall"] * 100,
    # results_dict["rouge_3_f_score"] * 100,
    results_dict["rouge_l_recall"] * 100

    # ,results_dict["rouge_su*_f_score"] * 100
    )


if __name__ == "__main__":
    model_id = sys.argv[1]
    threads_num = 1

    c_path = "../logs/abs_bert_cnndm." + model_id + ".candidate"
    r_path = "../logs/abs_bert_cnndm." + model_id + ".gold"
    s_path = "../logs/abs_bert_cnndm." + model_id + ".raw_src"

    candidates = codecs.open(c_path, encoding="utf-8")
    references = codecs.open(r_path, encoding="utf-8")
    src = codecs.open(s_path, encoding="utf-8")

    results_dict = test_rouge_single(candidates, references, src, threads_num)

    print(time.strftime('%H:%M:%S', time.localtime()))
    print(rouge_results_to_str(results_dict))
