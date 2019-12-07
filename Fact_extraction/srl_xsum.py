import sys
import json
#from label import get_batch_trees
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor

archive = load_archive("srl-model-2018.05.25.tar.gz", cuda_device=0)
srl = Predictor.from_archive(archive)
batch_size = 30
print ("Loading Done")

def get_srl(sentences, one_file, output):
	one_file["dlen"] = len(one_file["document"])
	sentences = [{"sentence": line} for line in sentences]
	srl_res = []; start_idx = 0
	while start_idx < len(sentences):
		batch_sentences = sentences[start_idx: min(start_idx + batch_size, len(sentences))]
		srl_res.extend(srl.predict_batch_json(batch_sentences))
		start_idx += batch_size
	if len(srl_res) > 1:
		one_file["srl_summary"] = srl_res[0]
		one_file["srl_document"] = srl_res[1:]
		output.write(json.dumps(one_file) + "\n")

if __name__ == '__main__':
	one_file = {}; one_file_sentences = []
	#input_filename = "./XSum.txt." + sys.argv[1]
	#xsum_trees_file = "./XSum.srl." + sys.argv[1]; existed_files = []
	input_filename = sys.argv[1]
	xsum_trees_file = sys.argv[1].split(".")[0] + ".srl"; existed_files = []

	try:
		for line in open(xsum_trees_file):
			tree_json = json.loads(line)
			existed_files.append(tree_json["filename"])
	except:
		pass
			
	output = open(xsum_trees_file, "a")
	for line in open(input_filename):
		line = line.strip()
		if line == "":
			if "document" not in one_file \
				or "filename" not in one_file \
				or one_file["filename"] in existed_files:
					one_file.clear(); del one_file_sentences[:]
					continue
			get_srl(one_file_sentences, one_file, output)
			one_file.clear(); del one_file_sentences[:]
		elif line.startswith("[FILENAME]"):
			filename = line.split(" ")[1]
			one_file["filename"] = filename
		elif line.startswith("[XSUM]"):
			summary = " ".join(line.split(" ")[1:])
			one_file["summary"] = summary
			one_file_sentences.append(summary)
		else:
			if "document" not in one_file:
				one_file["document"] = []
				one_file["tree_document"] = []
			one_file["document"].append(line)
			one_file_sentences.append(line)
