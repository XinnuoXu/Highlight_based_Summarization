#coding=utf8

delete_sentences = ["These are external links and will open in a new window",
        "Share this with",
        "Email",
        "Facebook",
        "Messenger",
        "Twitter",
        "Pinterest",
        "WhatsApp",
        "LinkedIn",
        "Linkedin",
        "Copy this link"]

def _parse(toks):
    components = []
    for idx, token in enumerate(toks):
        aWord = token.lower()
        if aWord == '-lrb-':
            aWord = '('
        elif aWord == '-rrb-':
            aWord = ')'
        elif aWord == '``':
            aWord = '"'
        elif aWord == '\'\'':
            aWord = '"'
        elif aWord == '#':
            aWord = '£'
        #elif aWord == '€':
        #    aWord = '$'
        #if aWord.endswith("km") and aWord != "km":
        #    components.append(aWord.replace("km", ""))
        #    components.append("km")
        #elif aWord.endswith("cm") and aWord != "cm":
        #    components.append(aWord.replace("cm", ""))
        #    components.append("cm")
        #else:
        components.append(aWord)
    return components

def rephrase(line):
    flist = [item for item in line.strip().split("\t") if (item not in delete_sentences)]
    new_line = " ".join(flist)
    return " ".join(_parse(new_line.split()))

if __name__ == '__main__':
    import sys, json
    for line in open("highres/debug.data"):
        line = line.strip()
        json_obj = json.loads(line)
        doc_id = json_obj["doc_id"]
        doc = json_obj["document"]
        summary = json_obj["summary"]
        my_summ_path = "../../HROUGE_data/summaries/ref_gold/" + doc_id + ".data"
        my_doc_path = "../50_docs/" + doc_id + ".data"
        with open(my_doc_path, 'r') as file:
            my_doc = rephrase(file.read())
        with open(my_summ_path, 'r') as file:
            #my_summary = rephrase(file.read())
            my_summary = file.read().strip()
        if summary != my_summary:
            print ("[summary] ", summary)
            print ("[my_summary] ", my_summary)
        if doc != my_doc:
            doc = doc.split()
            my_doc = my_doc.split()
            for i, item in enumerate(doc):
                if item != my_doc[i]:
                    print (item, my_doc[i])
                    break
            print ("[doc] ", doc)
            print ("[my_doc] ", my_doc)
            print ("")

