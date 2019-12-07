import sys
import os

if __name__ == '__main__':
	if sys.argv[1] == "tree":
		command = "nohup srun --mem 100g -p amd-longq --nodes 1 python srl_tree.py "
		for filename in os.listdir():
			if filename.startswith("XSum.srl."):
				file_num = filename.split(".")[-1]
				os.system(command + file_num + " &")
	if sys.argv[1] == "srl":
		command = "nohup srun --mem 100g -p amd-longq --nodes 1 --gres=gpu python srl_xsum.py "
		for filename in os.listdir():
			if filename.startswith("XSum.txt."):
				file_num = filename.split(".")[-1]
				os.system(command + file_num + " &")
	if sys.argv[1] == "segment":
		one_file = {}; files = []
		input_filename = "../XSum/XSum/XSum-Dataset/extracts.txt"
		for line in open(input_filename):
			line = line.strip()
			if line == "":
				files.append(one_file)
				one_file = {}
			elif line.startswith("[FILENAME]"):
				filename = line.split(" ")[1]
				one_file["filename"] = filename
			elif line.startswith("[XSUM]"):
				summary = " ".join(line.split(" ")[1:])
				one_file["summary"] = summary
			else:
				if "document" not in one_file:
					one_file["document"] = []
				one_file["document"].append(line)
		file_idx = 0; files_num = 20000
		fpout = open("./XSum.txt.0", "w")
		for i in range(0, len(files)):
			if i > 0 and i % files_num == 0:
				fpout.close()
				fpout = open("./XSum.txt." + str(int(i/files_num)), "w")
			one_file = files[i]
			if "filename" not in one_file \
				or "summary" not in one_file \
				or "document" not in one_file:
					continue
			fpout.write("[FILENAME] " + one_file["filename"] + "\n")
			fpout.write("[XSUM] " + one_file["summary"] + "\n")
			for s in one_file["document"]:
				fpout.write(s + "\n")
			fpout.write("\n")
