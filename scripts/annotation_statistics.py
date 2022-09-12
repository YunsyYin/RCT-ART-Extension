import json
import spacy, os
from spacy.tokens import DocBin

# make the factory work
from rel_pipe import make_relation_extractor, score_relations

# make the config work
from rel_model import create_relation_model, create_classification_layer, create_instances, create_tensors


def calculate_ner_labels_in_json(json_loc):
    intv = oc = meas = comp = ci = pval = des = 0
    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":
                for span in example['spans']:
                    if span['label'] == 'INTV': intv += 1
                    elif span['label'] == 'OC': oc += 1
                    elif span['label'] == 'MEAS': meas += 1
                    elif span['label'] == 'COMP': comp += 1
                    elif span['label'] == 'CI': ci += 1
                    elif span['label'] == 'PVAL': pval += 1
                    elif span['label'] == 'DES': des += 1
    return {
        'INTV': intv,
        'OC': oc,
        'MEAS': meas,
        'COMP': comp,
        'CI': ci,
        'PVAL': pval,
        'DES': des,
    }


def calculate_rel_labels_in_json(json_loc):
    a1_res = a2_res = oc_res = comp_res = ci_res = pval_res = des_res = 0
    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            if example["answer"] == "accept":
                for relation in example['relations']:
                    if relation['label'] == 'A1_RES': a1_res += 1
                    elif relation['label'] == 'A2_RES': a2_res += 1
                    elif relation['label'] == 'OC_RES': oc_res += 1
                    elif relation['label'] == 'COMP_RES': comp_res += 1
                    elif relation['label'] == 'CI_RES': ci_res += 1
                    elif relation['label'] == 'PVAL_RES': pval_res += 1
                    elif relation['label'] == 'DES_RES': des_res += 1
    return {
        'A1_RES': a1_res,
        'A2_RES': a2_res,
        'OC_RES': oc_res,
        'COMP_RES': comp_res,
        'CI_RES': ci_res,
        'PVAL_RES': pval_res,
        'DES_RES': des_res,
    }


def calculate_examples_in_doc(doc_loc):
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)
    return doc_bin.__len__()


def calculate_ner_labels_in_doc(doc_loc, ner_model):
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)
    nlp = spacy.load(ner_model)
    docs = doc_bin.get_docs(nlp.vocab)
    intv = oc = meas = comp = ci = pval = des = 0
    for doc in docs:
        for ent in doc.ents:
            if ent.label_ == 'INTV': intv += 1
            elif ent.label_ == 'OC': oc += 1
            elif ent.label_ == 'MEAS': meas += 1  
            elif ent.label_ == 'COMP': comp += 1
            elif ent.label_ == 'CI': ci += 1
            elif ent.label_ == 'PVAL': pval += 1
            elif ent.label_ == 'DES': des += 1
    return {
        'INTV': intv,
        'OC': oc,
        'MEAS': meas,
        'COMP': comp,
        'CI': ci,
        'PVAL': pval,
        'DES': des,
    }


def calculate_rel_labels_in_doc(doc_loc, rel_model):
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)
    nlp = spacy.load(rel_model)
    docs = doc_bin.get_docs(nlp.vocab)
    a1_res = a2_res = oc_res = comp_res = ci_res = pval_res = des_res = 0
    for doc in docs:
        keyError = False
        for key, value in doc._.rel.items():
            try:
                if value['A1_RES'] == 1.0: a1_res += 1
                elif value['A2_RES'] == 1.0: a2_res += 1
                elif value['OC_RES'] == 1.0: oc_res += 1
                elif value['COMP_RES'] == 1.0: comp_res += 1
                elif value['CI_RES'] == 1.0: ci_res += 1  
                elif value['PVAL_RES'] == 1.0: pval_res += 1  
                elif value['DES_RES'] == 1.0: des_res += 1  
            except KeyError as e: 
                keyError = True
                pass
    return {
        'A1_RES': a1_res,
        'A2_RES': a2_res,
        'OC_RES': oc_res,
        'COMP_RES': comp_res,
        'CI_RES': ci_res,
        'PVAL_RES': pval_res,
        'DES_RES': des_res,
    }


def output_text_with_ner_label(doc_loc, doc_name, ner_label):
    doc_bin = DocBin(store_user_data=True).from_disk(os.path.join(doc_loc, doc_name))
    nlp = spacy.load('../trained_models/stats_descriptive/ner/model-best')
    docs = doc_bin.get_docs(nlp.vocab)
    with open(f'../output_examples/{doc_name}_{ner_label}.txt', 'a') as file:
        file.write(f'|| {ner_label}\n')
        for doc in docs:
            for ent in doc.ents:
                if ent.label_ == ner_label:
                    file.write(f'{doc.text}\n')
                    break
                    

def output_text_with_rel_label(doc_loc, doc_name, rel_label):
    doc_bin = DocBin(store_user_data=True).from_disk(os.path.join(doc_loc, doc_name))
    nlp = spacy.load('../trained_models/stats_descriptive/rel/model-best')
    docs = doc_bin.get_docs(nlp.vocab)
    with open(f'../output_examples/{doc_name}_{rel_label}.txt', 'a') as file:
        file.write(f'|| {rel_label}\n')
        for doc in docs:
            for key, value in doc._.rel.items():
                if value[rel_label] == 1.0:
                    file.write(f'{doc.text}\n')
                    break



if __name__ == "__main__":

    stats_anno_loc = "../datasets/4_sample_comparison/ner/1_annotation_sets"
    stats_anno_path = os.listdir(stats_anno_loc)
    # preprocessed_loc = "../trained_models_history/1_stats/rel_annotation_2/5/corpus_stats_2"
    # preprocessed_loc = "../datasets/2_stats_descriptive/2_preprocessed/out_of_domain/cardiovascular_disease_as_test"
    # preprocessed_loc = "../datasets/3_descriptive/2_preprocessed/out_of_domain/2_splitted_before_combined/solid_tumour_cancer_as_test"
    # preprocessed_loc = "../datasets/1_stats/2_preprocessed/corpus_stats_2_pmid_updated"
    # preprocessed_loc = "../trained_models_history/3_descriptive/3/cardiovascular_disease_as_test"
    preprocessed_loc = "../datasets/4_sample_comparison/rel/2_preprocessed/rel_optimised_500_1"
    # preprocessed_loc = "../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive_all/2_splitted_before_combined"
    preprocessed_path = os.listdir(preprocessed_loc)
    # ner_model = "../trained_models_history/3_descriptive/1/ner/model-best"
    # rel_model = "../trained_models_history/3_descriptive/1/rel/model-best"
    ner_model = "../trained_models/stats_descriptive/ner/model-best"
    rel_model = "../trained_models/stats_descriptive/rel/model-best"

    # # Calculate the total number of each ner label in a json file
    # for file in stats_anno_path:
    #     print('||', file, calculate_ner_labels_in_json(os.path.join(stats_anno_loc, file)))

    # # Calculate the total number of each rel label in a json file
    # for file in stats_anno_path:
    #     print('||', file, calculate_rel_labels_in_json(os.path.join(stats_anno_loc, file)))

    # # Calculate the total number of examples in a doc file
    # for file in preprocessed_path:
    #     print('||', file, calculate_examples_in_doc(os.path.join(preprocessed_loc, file)))

    # # Calculate the total number of each ner and rel label in a spacy file
    # for file in preprocessed_path:
    #     print('||', file, calculate_ner_labels_in_doc(os.path.join(preprocessed_loc, file), ner_model))

    # for file in preprocessed_path:
    #     print('||', file, calculate_rel_labels_in_doc(os.path.join(preprocessed_loc, file), rel_model))

    # Output text with a specific ner label
    ner_labels = ['DES']
    preprocessed_loc = "../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/1_combined_before_splitted/datasets"
    for ner_label in ner_labels:
        output_text_with_ner_label(preprocessed_loc, 'test.spacy', ner_label)   

    # # Output text with a specific rel label
    # rel_labels = ['PVAL_RES']
    # preprocessed_loc = "../trained_model_history/3_stats_descriptive/corpus_stats_descriptive/2_splitted_before_combined/datasets"
    # for rel_label in rel_labels:
    #     output_text_with_rel_label(preprocessed_loc, 'test.spacy', rel_label)  
