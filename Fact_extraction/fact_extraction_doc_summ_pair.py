import sys
import os
import sys
import json
import os.path as path
import torch
from srl_tree import one_summary, get_batch_trees
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def get_srl(sentences, srl, fpout):
        batch_size = 200
        sentences = [{"sentence": line} for line in sentences]
        start_idx = 0
        while start_idx < len(sentences):
                batch_sentences = sentences[start_idx: min(start_idx + batch_size, len(sentences))]
                srl_res = srl.predict_batch_json(batch_sentences)
                start_idx += batch_size
                for s in srl_res:
                        fpout.write(json.dumps(s) + "\n")
        return srl_res

if __name__ == '__main__':
        if sys.argv[1] == "tree":
                output_dir = "doc_predictions/" + sys.argv[2] + ".tree"
                fpout = open(output_dir, "w")
                for line in open("doc_predictions/" + sys.argv[2] + ".srl"):
                        json.obj = json.loads(line.strip())
                        fpout.write(one_summary(json.obj) + "\n")
                fpout.close()

        if sys.argv[1] == "srl_summary":
                archive = load_archive("srl-model-2018.05.25.tar.gz", cuda_device=torch.cuda.current_device())
                srl = Predictor.from_archive(archive)
                print ("Loading Done")

                input_path = "doc_predictions/" + sys.argv[2] + "_pred.tgt"
                output_path = "doc_predictions/" + sys.argv[2] + "_tgt.srl"
                fpout = open(output_path, "w")
                summs = [line.strip() for line in open(input_path)]
                summs_srl = get_srl(summs, srl, fpout)

                fpout.close()

        if sys.argv[1] == "srl_gold":
                archive = load_archive("srl-model-2018.05.25.tar.gz", cuda_device=torch.cuda.current_device())
                srl = Predictor.from_archive(archive)
                print ("Loading Done")

                input_path = "doc_predictions/" + sys.argv[2] + "_pred.gold"
                output_path = "doc_predictions/" + sys.argv[2] + "_gold.srl"
                fpout = open(output_path, "w")
                summs = [line.strip() for line in open(input_path)]
                summs_srl = get_srl(summs, srl, fpout)

                fpout.close()
