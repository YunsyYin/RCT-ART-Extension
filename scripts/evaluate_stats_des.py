"""
Evaluate the five NLP tasks of the RCT-ART system: NER, RE, JOINT NER + RE,
TABULATION (STRICT), TABULATION (RELAXED). Also generate confusion matrices.
"""
import spacy, operator, csv, os
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
from spacy.scorer import Scorer,PRFScore
from spacy.vocab import Vocab
from itertools import zip_longest

# # make the factory work
from rel_pipe_des import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


# This function is adapted from RCT-ART
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


# This function is adapted from the spaCy relation component and RCT-ART
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


# This function is adapted from RCT-ART
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


# This function is adapted from RCT-ART
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
            if pred == {'outcome': 'intervention', 'arm 1': '', 'arm 2': '', 'comparative statistic': '', '95% CI': '', 'p-value': '', 'note': ''} \
                    and gold != {'outcome': 'intervention', 'arm 1': '', 'arm 2': '', 'comparative statistic': '', '95% CI': '', 'p-value': '', 'note': ''}: pred = False

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
            print(example)
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


# This function is adapted from RCT-ART
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
                          labels=["OC","INTV","MEAS","COMP","CI","PVAL","DES","NO_ENT"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["OC","INTV","MEAS","COMP","CI","PVAL","DES","NO_ENT"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = disp.plot(include_values=True,
             cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()


# This function is adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def create_rel_confusion_matrix(model_path, test_path):
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    from tabulate import relation_extraction
    vocab = Vocab()
    doc_bin = DocBin(store_user_data=True).from_disk(test_path)
    for_pred = list(doc_bin.get_docs(vocab))
    pred_docs = relation_extraction(model_path, for_pred, descriptive=True)
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

    cm = confusion_matrix(gold_array, pred_array, labels=["A1_RES", "A2_RES", "OC_RES", "COMP_RES", "CI_RES", "PVAL_RES", "DES_RES", "NO_RE"],
                          sample_weight=None, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["A1_RES", "A2_RES", "OC_RES", "COMP_RES", "CI_RES", "PVAL_RES", "DES_RES", "NO_RE"])
    font = {'family': 'normal',
            'weight': 'normal',
            'size': 12}
    plt.rc('font', **font)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')

    plt.show()



if __name__ == "__main__":

    # Evaluate corpus_stats_descriptive with test dataset
    # replicates = ["1_combined_before_splitted", "2_splitted_before_combined"]
    # datasets = ["test", "test_numerical", "test_descriptive"]
    # for replicate in replicates:
    #     ner_model = f"../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/{replicate}/ner/model-best"
    #     rel_model = f"../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/{replicate}/rel/model-best"
    #     gold_table = f"../datasets/3_stats_descriptive/3_gold_tables/{replicate}"
    #     gold_table_text = f"../datasets/3_stats_descriptive/4_gold_tables_from_text/{replicate}"
    #     pred_table = f"../output_tables/stats_descriptive/pred_tables/{replicate}"
    #     pred_table_text = f"../output_tables/stats_descriptive/pred_tables_from_text/{replicate}"
    #     for dataset in datasets:
    #         doc_loc = f"../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive/{replicate}/{dataset}.spacy"
    #         ner_evaluate(ner_model, doc_loc, f"corpus_all_{replicate}_{dataset}")
    #         joint_ner_rel_evaluate(None, rel_model, doc_loc, f"corpus_all_{replicate}_{dataset}", False)
    #         joint_ner_rel_evaluate(ner_model, rel_model, doc_loc, f"corpus_all_{replicate}_{dataset}", False)
    #         evaluate_result_tables(f"{gold_table}/{dataset}", f"{pred_table}/{dataset}", f"corpus_all_{replicate}_{dataset}", strict=True)
    #         evaluate_result_tables(f"{gold_table}/{dataset}", f"{pred_table}/{dataset}", f"corpus_all_{replicate}_{dataset}", strict=False)
    #         evaluate_result_tables(f"{gold_table_text}/{dataset}", f"{pred_table_text}/{dataset}", f"corpus_all_{replicate}_{dataset}", strict=True, text=True)
    #         evaluate_result_tables(f"{gold_table_text}/{dataset}", f"{pred_table_text}/{dataset}", f"corpus_all_{replicate}_{dataset}", strict=False, text=True)

    # Create confusion matrix
    # replicates = ["1_combined_before_splitted", "2_splitted_before_combined"]
    # datasets = ["test", "test_numerical", "test_descriptive"]
    # for replicate in replicates:
    #     ner_model = f"../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/{replicate}/ner/model-best"
    #     rel_model = f"../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/{replicate}/rel/model-best"
    #     for dataset in datasets:
    #         doc_loc = f"../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive/{replicate}/{dataset}.spacy"
    #         create_ner_confusion_matrix(ner_model, doc_loc)
    #         create_rel_confusion_matrix(rel_model, doc_loc)

    # Sample comparison
    # models = ["ner", "rel"]
    # numbers = ["200_1", "200_2", "300_1", "300_2", "400_1", "400_2", "500_1", "500_2"]
    # for model in models:
    #     for number in numbers:
    #         model_loc = f"../trained_model_history/4_sample_comparison/{model}/{number}/{model}/model-best"
    #         doc_loc = f"../trained_model_history/4_sample_comparison/{model}/{number}/datasets/test.spacy"
    #         if model == "ner":
    #             ner_evaluate(model_loc, doc_loc, f"{model}_{number}")
    #         else:
    #             joint_ner_rel_evaluate(None, model_loc, doc_loc, f"{model}_{number}", False)

    # Unseen disease domains
    replicates = ["1_splitted_before_combined", "2_combined_before_splitted"]
    domains = ["autism_as_test", "cardiovascular_disease_as_test", "solid_tumour_cancer_as_test"]
    for replicate in replicates:
        for domain in domains:
            doc_loc = f"../datasets/3_stats_descriptive/2_preprocessed/out_of_domain/{replicate}/{domain}/test.spacy"
            ner_model = f"../trained_model_history/3_stats_descriptive/out_of_domain/{replicate}/{domain}/ner/model-best"
            rel_model = f"../trained_model_history/3_stats_descriptive/out_of_domain/{replicate}/{domain}/rel/model-best"
            gold_table = f"../datasets/3_stats_descriptive/3_gold_tables/out_of_domain/{replicate}/{domain}"
            gold_table_text = f"../datasets/3_stats_descriptive/4_gold_tables_from_text/out_of_domain/{replicate}/{domain}"
            pred_table = f"../output_tables/stats_descriptive/pred_tables/out_of_domain/{replicate}/{domain}"
            pred_table_text = f"../output_tables/stats_descriptive/pred_tables_from_text/out_of_domain/{replicate}/{domain}"
            ner_evaluate(ner_model, doc_loc, f"{domain}_{replicate}")
            joint_ner_rel_evaluate(None, rel_model, doc_loc, f"{domain}_{replicate}", False)
            joint_ner_rel_evaluate(ner_model, rel_model, doc_loc, f"{domain}_{replicate}", False)
            evaluate_result_tables(gold_table, pred_table, f"{domain}_{replicate}", strict=True)
            evaluate_result_tables(gold_table, pred_table, f"{domain}_{replicate}", strict=False)
            evaluate_result_tables(gold_table_text, pred_table_text, f"{domain}_{replicate}", strict=True, text=True)
            evaluate_result_tables(gold_table_text, pred_table_text, f"{domain}_{replicate}", strict=False, text=True)

    

