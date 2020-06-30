# import spacy
import stanfordnlp
import json
import os
import numpy as np
# stanfordnlp.download('en')
file_name = ["dev_test.json","dev_dev.json","dev_train.json","train.json"]
# file_name = ["train.json"]
# spacy_parser = spacy.load("en")
stanford_parser = stanfordnlp.Pipeline(tokenize_pretokenized=True)
for file in file_name:
    with open(os.path.join("prepro_data", file)) as f:
        docs = json.load(f)
        doc_deps = []
        for i, doc in enumerate(docs):
            sents = doc["sents"]
            words_num = 0
            for s in sents:
                words_num+=len(s)
            deps = []
            nlp_sents = stanford_parser(sents)
            split_num = 0
            for sen in nlp_sents.sentences:
                dep = []
                root_num = 0
                roots = []
                split_num += len(sen.words)
                for word in sen.words:
                    dep.append(word.governor-1)
                    if word.governor-1 == -1:
                        root_num += 1
                assert root_num == 1
                deps.append(dep)

            assert split_num == words_num
            assert len(sents) == len(nlp_sents.sentences)
            print(i)
            doc_deps.append(deps)
        np.save(os.path.join("prepro_data",file.split(".j")[0]+"_dep.npy"), doc_deps)
