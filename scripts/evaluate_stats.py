"""
Evaluate the five NLP tasks of the RCT-ART system: NER, RE, JOINT NER + RE,
TABULATION (STRICT), TABULATION (RELAXED). Also generate confusion matrices.
"""
import spacy, operator, csv, os, json
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.scorer import Scorer,PRFScore
from spacy.vocab import Vocab
from itertools import zip_longest

# # make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def ner_evaluate(ner_model_loc, test_data, filename):
    """ Evaluates NER scores of model on test data, can output to console and file"""
    print("|| Loading model for NER task")
    nlp = spacy.load(ner_model_loc)
    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(nlp.vocab)
    examples = []
    for gold in docs:
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))
    print("|| Evaluating NER task performance")
    print(nlp.evaluate(examples))
    with open(f'../output_evaluations/{filename}_evaluation.txt', 'a') as file:
        file.write("|| NER Evaluation\n")
        file.write(f"{nlp.evaluate(examples)}\n")


# This function was extensively adapted from the spaCy relation component and RCT-ART
# template: https://github.com/explosion/projects/tree/v3/tutorials
# template: https://github.com/jetsunwhitton/RCT-ART
# it can be used to evaluate joint entity--relation extraction performance
def joint_ner_rel_evaluate(ner_model_loc, rel_model_loc, test_data, filename, print_details: bool):
    """Evaluates joint performance of ner and rel extraction model,
    as well as the rel model alone if only rel model provided"""
    if ner_model_loc != None:
        print("|| Loading models for joint task")
        ner = spacy.load(ner_model_loc)
        print("|| Evaluating joint task performance")
    else:
        print("|| Loading models for rel task")
        print("|| Evaluating rel task performance")
    rel = spacy.load(rel_model_loc)

    doc_bin = DocBin(store_user_data=True).from_disk(test_data)
    docs = doc_bin.get_docs(rel.vocab)
    examples = []
    for gold in docs:
        pred = Doc(
            rel.vocab,
            words=[t.text for t in gold],
            spaces=[t.whitespace_ for t in gold],
        )
        if ner_model_loc != None:
            pred.ents = ner(gold.text).ents
        else:
            pred.ents = gold.ents
        for name, proc in rel.pipeline:
            pred = proc(pred)
        examples.append(Example(pred, gold))

        # Print the gold and prediction, if gold label is not 0
        if print_details:
            gold_ents = [e.text for e in gold.ents]
            assessed_ents = []
            for value, rel_dict in pred._.rel.items():
                try:
                    gold_labels = [k for (k, v) in gold._.rel[value].items() if v == 1.0]
                    if gold_labels:
                        print(
                            f" pair: {value} --> gold labels: {gold_labels} --> predicted values: {rel_dict}"
                        )
                except KeyError:
                    pred_rel = max(rel_dict.items(),key=operator.itemgetter(1))
                    if pred_rel[1] > 0.5:
                        print("Relation mapped with wrong entity pair")
                    else:
                        parent_ent = list(filter(lambda x: x.start == value[0], pred.ents))[0].text
                        child_ent = list(filter(lambda x: x.start == value[1], pred.ents))[0].text
                        if parent_ent not in assessed_ents:
                            if parent_ent in gold_ents:
                                print(parent_ent," Correct entity and correctly didn't map relation")
                            else:
                                print(parent_ent," incorrect entity")
                            assessed_ents.append(parent_ent)
                        if child_ent not in assessed_ents:
                            if child_ent in gold_ents:
                                print(child_ent, "Correct entity and correctly didn't map relation")
                            else:
                                print(child_ent, "incorrect entity")
                            assessed_ents.append(child_ent)
            print()
    thresholds = [0.5]
    task = False
    if ner_model_loc != None:
        task = True
    _score_and_format(examples, thresholds, task, filename)


# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def _score_and_format(examples, thresholds, task, filename):
    """outputs rel and joint performance scores, to console and txt file"""
    with open(f'../output_evaluations/{filename}_evaluation.txt', 'a') as file:
        if task:
            file.write("|| NER + REL Evaluation\n")
        else:
            file.write("|| Rel Evaluation\n")
        for threshold in thresholds:
            score = score_relations(examples, threshold)
            results = {key: "{:.2f}".format(value) for key, value in score.items()}
            file.write(f"threshold {'{:.2f}'.format(threshold)} \t {results}\n")


# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def evaluate_result_tables(gold_loc, predicted_loc, filename, strict=True, text=False):
    """ Evaluates performance of model on tabulation task, compares prediction tables
    vs gold tables, can output to console and/or txt file"""
    print("|| Evaluating table task performance")
    prf = PRFScore()
    examples = []
    for gold_csv, pred_csv in zip(os.listdir(gold_loc), os.listdir(predicted_loc)):
        gold_open = open(os.path.join(gold_loc, gold_csv), encoding="utf-8", newline='')
        pred_open = open(os.path.join(predicted_loc, pred_csv), encoding="utf-8", newline='')
        gold_list = [d for d in csv.DictReader(gold_open)]
        pred_list = [d for d in csv.DictReader(pred_open)]
        for gold, pred in zip_longest(gold_list,pred_list,fillvalue=False):
            if gold != False: del gold[''] # remove extra CSV formatting
            if pred != False: del pred['']

            # set pred to false for false negative if intv row empty when gold isn't
            if pred == {'outcome': 'intervention', 'arm 1': '', 'arm 2': ''} \
                    and gold != {'outcome': 'intervention', 'arm 1': '', 'arm 2': ''}: pred = False

            examples.append({"gold":gold,"pred":pred})
        if gold_list == []:
            print("error")
            continue # empty lists in gold are error in data az
        if pred_list == []: # empty lists in pred are false negatives if not empty in gold
            for gold in gold_list:
                del gold['']
                examples.append({"gold": gold, "pred": {}})

    if strict: # assess table with exact entity matches
        for example in examples:
            if not example["pred"]:
                prf.fn += 1
                print("FN ----> ", example)
            elif not example["gold"]:
                prf.fp += 1
            else:
                if example["pred"] == example["gold"]:
                    prf.tp += 1
                    print("TP ----> ", example)
                else:
                    prf.fp += 1
                    print("FP ----> ", example) 

    else: # assess tables with less strict entity criteria -- gold/pred entity boundary overlap
        for example in examples:
            relaxed_match = True
            if not example["pred"]: prf.fn += 1 # empty prediction --> false negative
            elif not example["gold"]: prf.fp += 1 # prediction made when no gold tuple --> false postive
            else:
                for pred_val, gold_val in zip(example["pred"].values(), example["gold"].values()):
                    if gold_val not in pred_val and pred_val not in gold_val:
                        relaxed_match = False
                if relaxed_match: prf.tp += 1
                else: prf.fp += 1

    output = {"rel_micro_p": round(prf.precision, 2),
              "rel_micro_r": round(prf.recall, 2),
              "rel_micro_f": round(prf.fscore, 2),}
    with open(f'../output_evaluations/{filename}_evaluation.txt', 'a') as file:
        file.write("|| Table Evaluation")
        if text: file.write(" From Text")
        if strict: file.write(": Strict\n")
        else: file.write(": Relaxed\n")
        file.write(f"{output}\n")
    print(output)


# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def create_ner_confusion_matrix(model_path, test_path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    ner = spacy.load(model_path)
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    gold_docs = list(doc_bin.get_docs(ner.vocab))
    pred_docs = [ner(gold_doc.text) for gold_doc in gold_docs]
    gold_array = []
    pred_array = []
    for gold_doc, pred_doc in zip(gold_docs, pred_docs):
        for g_tok,p_tok in zip(gold_doc, pred_doc):
            if g_tok.ent_type_ == '':
                gold_array.append("NO_ENT")
            else:
                gold_array.append(g_tok.ent_type_)
            if p_tok.ent_type_ == '':
                pred_array.append("NO_ENT")
            else:
                pred_array.append(p_tok.ent_type_)
    cm = confusion_matrix(gold_array, pred_array,
                          labels=["OC","INTV","MEAS","COMP","CI","PVAL","NO_ENT"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["OC","INTV","MEAS","COMP","CI","PVAL","NO_ENT"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = disp.plot(include_values=True,
             cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()


# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def create_rel_confusion_matrix(model_path, test_path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from tabulate import relation_extraction
    vocab = Vocab()
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    for_pred = list(doc_bin.get_docs(vocab))
    pred_docs = relation_extraction(model_path, for_pred)
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    gold_docs = list(doc_bin.get_docs(vocab))
    pred_array, pred_keys, gold_keys, gold_array = [], [], [], []
    for pred_doc, gold_doc in zip(pred_docs,gold_docs):
        for pkey, p_rel_dict in pred_doc._.rel.items():
            pred_keys.append(pkey)
            if pkey in gold_doc._.rel.keys():
                gold_keys.append(pkey)
                gold_rel = gold_doc._.rel[pkey]  # get relation
                max_gold = max(gold_rel.items(),
                              key=operator.itemgetter(1))  # selects highest probability relation
                if max_gold[1] > 0.5:  # includes relation if above set threshold for probability
                    gold_array.append(max_gold[0])
                else:
                    gold_array.append("NO_RE")
                pred_rel = pred_doc._.rel[pkey]  # get relation
                max_pred = max(pred_rel.items(),
                              key=operator.itemgetter(1))  # selects highest probability relation
                if max_pred[1] > 0.5:  # includes relation if above set threshold for probability
                    pred_array.append(max_pred[0])
                else:
                    pred_array.append("NO_RE")

    cm = confusion_matrix(gold_array, pred_array, labels=["A1_RES", "A2_RES", "OC_RES", "COMP_RES", "CI_RES", "PVAL_RES", "NO_RE"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A1_RES", "A2_RES", "OC_RES", "COMP_RES", "CI_RES", "PVAL_RES", "NO_RE"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()


def update_spacy_pmid_from_jsonl(doc_loc, json_loc, rel_model, filename, output_dir):
    nlp = spacy.load(rel_model)
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)
    docs = doc_bin.get_docs(nlp.vocab)
    docs_pmid_updated = []
    pmid_update_count = 0

    for doc in docs:
        doc.user_data["pmid"] = ""
        doc_text = "".join(doc.text.split())
        with open(json_loc, "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                example_text = "".join(example["text"].split())
                if doc_text == example_text:   
                    try:
                        doc.user_data["pmid"] = example["user_data"]["pmid"]
                        pmid_update_count += 1
                    except KeyError:
                        print(example)
                    break   
        docs_pmid_updated.append(doc)
    doc_bin_pmid_updated = DocBin(docs=docs_pmid_updated, store_user_data=True)
    doc_bin_pmid_updated.to_disk(f"{output_dir}/{filename}.spacy")
    print("Total examples:", len(docs_pmid_updated))
    print("PMID updated:", pmid_update_count)



if __name__ == "__main__":
    
    # # Evaluate corpus_stats_2 with test_pmid_updated dataset
    # doc_loc = "../datasets/1_stats/2_preprocessed/corpus_stats_2_pmid_updated/test_pmid_updated.spacy"
    # ner_model = "../trained_models/stats/ner/model-best"
    # rel_model = "../trained_models/stats/rel/model-best"
    # gold_table = "../datasets/1_stats/3_gold_tables/corpus_stats_2_pmid_updated/test_pmid_updated"
    # pred_table = "../output_tables/stats/pred_tables/corpus_stats_2_pmid_updated/test_pmid_updated"
    # gold_table_from_text = "../datasets/1_stats/4_gold_tables_from_text/corpus_stats_2_pmid_updated/test_pmid_updated"
    # pred_table_from_text = "../output_tables/stats/pred_tables_from_text/corpus_stats_2_pmid_updated/test_pmid_updated"
    # ner_evaluate(ner_model, doc_loc, "corpus_stats_2_pmid_updated_test")
    # joint_ner_rel_evaluate(None, rel_model, doc_loc, "corpus_stats_2_pmid_updated_test", False)
    # joint_ner_rel_evaluate(ner_model, rel_model, doc_loc, "corpus_stats_2_pmid_updated_test", False)
    # evaluate_result_tables(gold_table, pred_table, "corpus_stats_2_pmid_updated_test", strict=True)
    # evaluate_result_tables(gold_table, pred_table, "corpus_stats_2_pmid_updated_test", strict=False)
    # evaluate_result_tables(gold_table_from_text, pred_table_from_text, "corpus_stats_2_pmid_updated_test", strict=True, text=True)
    # evaluate_result_tables(gold_table_from_text, pred_table_from_text, "corpus_stats_2_pmid_updated_test", strict=False, text=True)

    # Update PMIDs in corpus_stats_2
    doc_loc = "../datasets/1_stats/2_preprocessed/corpus_stats_2/train.spacy"
    json_loc = "../datasets/1_stats/1_annotation_sets/corpus_stats_2_pmid_updated.jsonl"
    rel_model = "../trained_models/stats/rel/model-best"
    output_dir = "../datasets/1_stats/2_preprocessed/corpus_stats_2"
    update_spacy_pmid_from_jsonl(doc_loc, json_loc, rel_model, "train_pmid_updated", output_dir)

