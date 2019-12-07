import sys
import os
import sys
import json
import os.path as path
from srl_tree import one_summary
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

def get_srl(sentences, doc_ids, output_dir):
        archive = load_archive("srl-model-2018.05.25.tar.gz", cuda_device=0)
        srl = Predictor.from_archive(archive)
        batch_size = 5
        print ("Loading Done")

        fpout = open(output_dir, "w")
        sentences = [{"sentence": line} for line in sentences]
        srl_res = []; start_idx = 0
        while start_idx < len(sentences):
                batch_sentences = sentences[start_idx: min(start_idx + batch_size, len(sentences))]
                srl_res.extend(srl.predict_batch_json(batch_sentences))
                start_idx += batch_size
        for i, res in enumerate(srl_res):
                fpout.write(json.dumps({"srl": res, "doc_id": doc_ids[i]}) + "\n")
        fpout.close()

if __name__ == '__main__':
        if sys.argv[1] == "tree":
                output_dir = sys.argv[2] + ".tree/"
                if not path.exists(output_dir):
                        os.system("mkdir " + output_dir)
                for line in open(sys.argv[2] + ".srl"):
                        json_obj = json.loads(line.strip())
                        doc_id = json_obj["doc_id"]
                        fpout = open(output_dir + doc_id + ".tgt", "w")
                        fpout.write(one_summary(json_obj["srl"]))
                        fpout.close()
                        

        if sys.argv[1] == "srl":
                lines = []
                ids = []
                for filename in os.listdir("systems_output/" + sys.argv[2] + "/"):
                        path = "systems_output/" + sys.argv[2] + "/" + filename
                        doc_id = filename.split(".")[0]
                        with open(path, 'r') as file:
                                lines.append(file.read().strip())
                                ids.append(doc_id)
                get_srl(lines, ids, sys.argv[2] + ".srl")
