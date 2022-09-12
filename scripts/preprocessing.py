import json, random, ast, spacy, os
from spacy.tokens import DocBin, Doc
from spacy.vocab import Vocab
from wasabi import Printer


# This function is adapted from the spaCy relation component and RCT-ART
# template: https://github.com/explosion/projects/tree/v3/tutorials
# template: https://github.com/jetsunwhitton/RCT-ART
def annotations_to_spacy(json_loc):
    """Converts Prodigy annotations into doc object with custom rel attribute."""
    msg = Printer()
    MAP_LABELS = {
        "A1_RES": "A1_RES",
        "A2_RES": "A2_RES",
        "OC_RES": "OC_RES",
        "COMP_RES": "COMP_RES",
        "PVAL_RES": "PVAL_RES",
        "CI_RES": "CI_RES", 
        "DES_RES": "DES_RES",       
    }
    try:
        Doc.set_extension("rel", default={})
    except ValueError:
        print("Rel extension already set on doc")
    vocab = Vocab()
    docs = []

    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            span_starts = set()
            if example["answer"] == "accept":
                try:
                    # Parse the tokens
                    words = [t["text"] for t in example["tokens"]]
                    spaces = [t["ws"] for t in example["tokens"]]
                    doc = Doc(vocab, words=words, spaces=spaces)

                    # Parse the PICO entities
                    spans = example["spans"]
                    entities = []
                    span_end_to_start = {}
                    for span in spans:
                        entity = doc.char_span(
                            span["start"], span["end"], label=span["label"]
                        )
                        span_end_to_start[span["token_end"]] = span["token_start"]
                        entities.append(entity)
                        span_starts.add(span["token_start"])
                    doc.ents = entities

                    # Parse the PICO relations
                    rels = {}
                    for x1 in span_starts:
                        for x2 in span_starts:
                            rels[(x1, x2)] = {}
                    relations = example["relations"]
                    for relation in relations:
                        # swaps tokens to correct relation positions
                        start = span_end_to_start[relation["head"]]
                        end = span_end_to_start[relation["child"]]
                        label = relation["label"]
                        label = MAP_LABELS[label]
                        if label not in rels[(start, end)]:
                            rels[(start, end)][label] = 1.0


                    # The annotation is complete, so fill in zero's where the data is missing
                    for x1 in span_starts:
                        for x2 in span_starts:
                            for label in MAP_LABELS.values():
                                if label not in rels[(x1, x2)]:
                                    rels[(x1, x2)][label] = 0.0
                    doc._.rel = rels
                    try:
                        if type(example["user_data"]) == dict:
                            doc.user_data["pmid"] = example["user_data"]["pmid"]
                        else:
                            pmid = ast.literal_eval(example["user_data"])
                            doc.user_data["pmid"] = pmid["pmid"]
                    except KeyError:
                        pass # pmids have not been added to glaucoma dataset
                    docs.append(doc)
                except KeyError as e:
                    msg.fail(f"Skipping doc because of key error")
                    print(doc)
    random.shuffle(docs)
    print(json_loc," --> #", len(docs))
    return docs


# This function is adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def train_dev_test_split(docs, output_dir):
    """
    Splits spaCy docs collection into train test and dev datasets
    :param docs: list
    :return: train dev and test spacy files for model training and testing
    """
    random.shuffle(docs)  # randomise pmids before train, dev, test split
    l = len(docs)
    train = docs[0:int(l * 0.7)]
    dev = docs[int(l * 0.7):int(l * 0.8)]
    test = docs[int(l * 0.8):]

    doc_bin = DocBin(docs=train, store_user_data=True)
    doc_bin.to_disk(f"{output_dir}/train.spacy")
    print(f"{len(train)} training sentences")

    doc_bin = DocBin(docs=dev, store_user_data=True)
    doc_bin.to_disk(f"{output_dir}/dev.spacy")
    print(f"{len(dev)} dev sentences")

    doc_bin = DocBin(docs=test, store_user_data=True)
    doc_bin.to_disk(f"{output_dir}/test.spacy")
    print(f"{len(test)} test sentences")


def combine_docs(filename, output_dir, *doc_locs):
    """
    Combines docs together
    :return: a spacy file with datasets
    """
    doc_bin = DocBin(store_user_data=True)
    for doc_loc in doc_locs:
        doc_bin.merge(DocBin(store_user_data=True).from_disk(doc_loc))
    doc_bin.to_disk(f"{output_dir}/{filename}.spacy")
    print(len(doc_bin), filename, "sentences")


# This function is adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def out_of_domain_split(doc_dict, exclude):
    """excludes one domain from full domain train and dev sets for use as test set,
    input dictionary of docs with domains as keys"""
    merged_docs = list()

    for key in doc_dict:
        if key == exclude:
            continue
        merged_docs.extend(doc_dict[key])

    random.shuffle(merged_docs)
    l = len(merged_docs)
    train = merged_docs[0:int(l * 0.9)]
    dev = merged_docs[int(l * 0.9):]

    test = doc_dict[exclude]
    random.shuffle(test)

    docbin = DocBin(docs=train, store_user_data=True)
    docbin.to_disk(f"../datasets/3_descriptive/2_preprocessed/out_of_domain/{exclude}_as_test/train.spacy")
    print(f"{len(train)} training sentences")

    docbin = DocBin(docs=dev, store_user_data=True)
    docbin.to_disk(f"../datasets/3_descriptive/2_preprocessed/out_of_domain/{exclude}_as_test/dev.spacy")
    print(f"{len(dev)} dev sentences")

    docbin = DocBin(docs=test, store_user_data=True)
    docbin.to_disk(f"../datasets/3_descriptive/2_preprocessed/out_of_domain/{exclude}_as_test/test.spacy")
    print(f"{len(test)} test sentences")


def separate_numerical_and_descriptive_docs(doc_loc, output_dir):
    numerical_docs = []
    descriptvie_docs = []
    doc_bin = DocBin(store_user_data=True).from_disk(os.path.join(doc_loc))
    nlp = spacy.load('../trained_models/stats_descriptive/ner/model-best')
    docs = doc_bin.get_docs(nlp.vocab)
    for doc in docs:
        descriptive = False
        for ent in doc.ents:
            if ent.label_ == "DES":
                descriptive = True
                break
        if descriptive: descriptvie_docs.append(doc)
        else: numerical_docs.append(doc)

    docbin = DocBin(docs=numerical_docs, store_user_data=True)
    docbin.to_disk(f"{output_dir}/test_numerical.spacy")
    print(f"{len(numerical_docs)} numerical test sentences")

    docbin = DocBin(docs=descriptvie_docs, store_user_data=True)
    docbin.to_disk(f"{output_dir}/test_descriptive.spacy")
    print(f"{len(descriptvie_docs)} descriptive test sentences")


def optimise_ner_dataset(stats_json, descriptive_json, output_dir, example_number):
    """Optimise ner label proportion for ner model training"""
    accept_list = []
    candidate_list = []

    # Include all examples from the descriptive corpus
    with open(descriptive_json, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            accept_list.append(example)
            example_number -= 1

    # Include examples with COMP and CI labels from the numerical corpus
    with open(stats_json, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            accept = False
            for span in example['spans']:
                if span['label'] == 'COMP' or span['label'] == 'CI':
                    accept = True
                    break
            if accept:                     
                accept_list.append(example)
                example_number -= 1
            else:
                candidate_list.append(example)

    # Fill in samples from candidates
    random.shuffle(candidate_list)
    accept_list = accept_list + candidate_list[0:example_number]
    random.shuffle(accept_list)

    # Open json writer and dump data
    with open(f"{output_dir}/1_annotation_sets/optimised_ner.jsonl", 'w', encoding='utf-8') as jsonfile:
        for row in accept_list:
            jsonfile.write(json.dumps(row) + "\n")

    # Split datasets
    optimised_doc = annotations_to_spacy(f"{output_dir}/1_annotation_sets/optimised_ner.jsonl")
    train_dev_test_split(optimised_doc, f"{output_dir}/2_preprocessed")


def optimise_rel_dataset(stats_json, descriptive_json, output_dir, example_number):
    """Optimise rel label proportion for rel model training"""
    accept_list = []
    candidate_list = []

    # Include examples with COMP_RES and CI_RES labels from the numerical corpus
    with open(stats_json, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            accept = False
            candidate = False
            for relation in example['relations']:
                if relation['label'] == 'COMP_RES' or relation['label'] == 'CI_RES':
                    accept = True
                    break
                elif relation['label'] == 'PVAL_RES':
                    candidate = True
            if accept:                     
                accept_list.append(example)
                example_number -= 1
            elif candidate:
                candidate_list.append(example)

    # Include examples with COMP_RES and CI_RES labels from the descriptive corpus
    with open(descriptive_json, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            accept = False
            candidate = False
            for relation in example['relations']:
                if relation['label'] == 'COMP_RES' or relation['label'] == 'CI_RES':
                    accept = True
                    break
                elif relation['label'] == 'PVAL_RES' or relation['label'] == 'DES_RES':
                    candidate = True
            if accept:                     
                accept_list.append(example)
                example_number -= 1
            elif candidate:
                candidate_list.append(example)

    # Fill in samples from candidates
    random.shuffle(candidate_list)
    accept_list = accept_list + candidate_list[0:example_number]
    random.shuffle(accept_list)

    # Open json writer and dump data
    with open(f"{output_dir}/1_annotation_sets/optimised_rel.jsonl", 'w', encoding='utf-8') as jsonfile:
        for row in accept_list:
            jsonfile.write(json.dumps(row) + "\n")

    # Split datasets
    optimised_doc = annotations_to_spacy(f"{output_dir}/1_annotation_sets/optimised_rel.jsonl")
    train_dev_test_split(optimised_doc, f"{output_dir}/2_preprocessed")


    
if __name__ == "__main__":

    # corpus_docs = annotations_to_spacy("../datasets/4_gold_tables_from_text/pmid_updated/corpus_stats_2_pmid_updated.jsonl")
    # doc_bin = DocBin(docs=corpus_docs, store_user_data=True)
    # doc_bin.to_disk("../datasets/4_gold_tables_from_text/pmid_updated/corpus_stats_2_pmid_updated.spacy")

    # corpus_docs = annotations_to_spacy("../datasets/7_des_annotation/corpus_des.jsonl")
    # train_dev_test_split(corpus_docs,"../datasets/8_preprocessed_des/corpus_des")

    # corpus_docs = annotations_to_spacy("../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2/corpus_des_2.jsonl")
    # train_dev_test_split(corpus_docs,"../datasets/2_stats_descriptive/2_preprocessed/corpus_des_2")

    # corpus_docs = annotations_to_spacy("../datasets/3_descriptive/1_annotation_sets/corpus_descriptive_2/corpus_descriptive_2.jsonl")
    # train_dev_test_split(corpus_docs,"../datasets/3_descriptive/2_preprocessed/corpus_descriptive_2")

    # # Create descriptive datasets 
    # output_dir = "../datasets/2_stats_descriptive/2_preprocessed/corpus_descriptive"
    # datasets = ["train", "dev", "test"]
    # for dataset in datasets:
    #     autism = f"../datasets/2_stats_descriptive/2_preprocessed/domains/autism/{dataset}.spacy"
    #     cardiovasvular = f"../datasets/2_stats_descriptive/2_preprocessed/domains/cardiovascular_disease/{dataset}.spacy"
    #     tumour = f"../datasets/2_stats_descriptive/2_preprocessed/domains/solid_tumour_cancer/{dataset}.spacy"
    #     combine_docs(dataset, output_dir, autism, cardiovasvular, tumour)


    # # Include 3 descriptive datasets into the statistical dataset 
    # output_dir = "../datasets/2_stats_descriptive/2_preprocessed/corpus_stats_des"
    # datasets = ["train", "dev", "test"]
    # for dataset in datasets:
    #     corpus_stats_2 = f"../datasets/1_stats/2_preprocessed/corpus_stats_2_pmid_updated/{dataset}_pmid_updated.spacy"
    #     autism = f"../datasets/2_stats_descriptive/2_preprocessed/domains/autism/{dataset}.spacy"
    #     cardiovasvular = f"../datasets/2_stats_descriptive/2_preprocessed/domains/cardiovascular_disease/{dataset}.spacy"
    #     tumour = f"../datasets/2_stats_descriptive/2_preprocessed/domains/solid_tumour_cancer/{dataset}.spacy"
    #     combine_docs(dataset, output_dir, corpus_stats_2, autism, cardiovasvular, tumour)
    
    # Out of domain datasets
    output_dir = "../datasets/2_stats_descriptive/2_preprocessed/out_of_domain/"
    corpus_stats = "../datasets/1_stats/2_preprocessed/corpus_stats_2_pmid_updated"
    autism = "../datasets/2_stats_descriptive/2_preprocessed/domains/autism"
    cardiovasvular = "../datasets/2_stats_descriptive/2_preprocessed/domains/cardiovascular_disease"
    tumour = "../datasets/2_stats_descriptive/2_preprocessed/domains/solid_tumour_cancer"
    autism_jsonl = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2/domains/autism.jsonl"
    cardiovascular_jsonl = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2/domains/cardiovascular_disease.jsonl"
    tumour_jsonl = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2/domains/solid_tumour_cancer.jsonl"

    # # Autism
    # combine_docs("train", f"{output_dir}/autism_as_test", f"{corpus_stats}/train_pmid_updated.spacy", f"{corpus_stats}/test_pmid_updated.spacy", f"{cardiovasvular}/train.spacy", f"{cardiovasvular}/test.spacy", f"{tumour}/train.spacy", f"{tumour}/test.spacy")
    # combine_docs("dev", f"{output_dir}/autism_as_test", f"{corpus_stats}/dev_pmid_updated.spacy", f"{cardiovasvular}/dev.spacy", f"{tumour}/dev.spacy")
    # DocBin(docs=annotations_to_spacy(autism_jsonl), store_user_data=True).to_disk(f"{output_dir}/autism_as_test/test.spacy")

    # # Cardiovascular disease
    # combine_docs("train", f"{output_dir}/cardiovascular_disease_as_test", f"{corpus_stats}/train_pmid_updated.spacy", f"{corpus_stats}/test_pmid_updated.spacy", f"{autism}/train.spacy", f"{autism}/test.spacy", f"{tumour}/train.spacy", f"{tumour}/test.spacy")
    # combine_docs("dev", f"{output_dir}/cardiovascular_disease_as_test", f"{corpus_stats}/dev_pmid_updated.spacy", f"{autism}/dev.spacy", f"{tumour}/dev.spacy")
    # DocBin(docs=annotations_to_spacy(cardiovascular_jsonl), store_user_data=True).to_disk(f"{output_dir}/cardiovascular_disease_as_test/test.spacy")

    # # Solid tumour cancer
    # combine_docs("train", f"{output_dir}/solid_tumour_cancer_as_test", f"{corpus_stats}/train_pmid_updated.spacy", f"{corpus_stats}/test_pmid_updated.spacy", f"{autism}/train.spacy", f"{autism}/test.spacy", f"{cardiovasvular}/train.spacy", f"{cardiovasvular}/test.spacy")
    # combine_docs("dev", f"{output_dir}/solid_tumour_cancer_as_test", f"{corpus_stats}/dev_pmid_updated.spacy", f"{autism}/dev.spacy", f"{cardiovasvular}/dev.spacy")
    # DocBin(docs=annotations_to_spacy(tumour_jsonl), store_user_data=True).to_disk(f"{output_dir}/solid_tumour_cancer_as_test/test.spacy")

    # # Out of domain processing
    # domain_path = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2/domains"
    # doc_dict = {}
    # exclude_list = ["autism", "cardiovascular_disease", "solid_tumour_cancer"]
    # for domain in os.listdir(domain_path):
    #    doc_dict[domain.replace(".jsonl","")] = annotations_to_spacy(os.path.join(domain_path, domain))
    # for exclude in exclude_list:
    #    out_of_domain_split(doc_dict,exclude)

    # # Out of domain processing
    # domain_path = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_descriptive/domains"
    # doc_dict = {}
    # exclude_list = ["autism", "cardiovascular_disease", "solid_tumour_cancer"]
    # for domain in os.listdir(domain_path):
    #    doc_dict[domain.replace(".jsonl","")] = annotations_to_spacy(os.path.join(domain_path, domain))
    # for exclude in exclude_list:
    #    out_of_domain_split(doc_dict,exclude)


    output_dir = "../datasets/3_descriptive/2_preprocessed/out_of_domain/2_splitted_before_combined"
    domain_loc = "../datasets/3_descriptive/2_preprocessed/corpus_descriptive/domains"
    autism = f"{domain_loc}/autism"
    cardiovasvular = f"{domain_loc}/cardiovascular_disease"
    tumour = f"{domain_loc}/solid_tumour_cancer"

    # # Autism
    # combine_docs("train", f"{output_dir}/autism_as_test", f"{cardiovasvular}/train.spacy", f"{cardiovasvular}/test.spacy", f"{tumour}/train.spacy", f"{tumour}/test.spacy")
    # combine_docs("dev", f"{output_dir}/autism_as_test", f"{cardiovasvular}/dev.spacy", f"{tumour}/dev.spacy")
    # combine_docs("test", f"{output_dir}/autism_as_test", f"{autism}/train.spacy", f"{autism}/dev.spacy", f"{autism}/test.spacy")

    # # Cardiovascular disease
    # combine_docs("train", f"{output_dir}/cardiovascular_disease_as_test", f"{autism}/train.spacy", f"{autism}/test.spacy", f"{tumour}/train.spacy", f"{tumour}/test.spacy")
    # combine_docs("dev", f"{output_dir}/cardiovascular_disease_as_test", f"{autism}/dev.spacy", f"{tumour}/dev.spacy")
    # combine_docs("test", f"{output_dir}/cardiovascular_disease_as_test", f"{cardiovasvular}/train.spacy", f"{cardiovasvular}/dev.spacy", f"{cardiovasvular}/test.spacy")

    # # Solid tumour cancer
    # combine_docs("train", f"{output_dir}/solid_tumour_cancer_as_test", f"{autism}/train.spacy", f"{autism}/test.spacy", f"{cardiovasvular}/train.spacy", f"{cardiovasvular}/test.spacy")
    # combine_docs("dev", f"{output_dir}/solid_tumour_cancer_as_test", f"{autism}/dev.spacy", f"{cardiovasvular}/dev.spacy")
    # combine_docs("test", f"{output_dir}/solid_tumour_cancer_as_test", f"{tumour}/train.spacy", f"{tumour}/dev.spacy", f"{tumour}/test.spacy")

    stats_json = "../datasets/1_stats/1_annotation_sets/corpus_stats_2_pmid_updated.jsonl"
    descriptvie_json = "../datasets/2_descriptive/1_annotation_sets/corpus_descriptive_all.jsonl"
    optimise_ner_dataset(stats_json, descriptvie_json, "../datasets/4_optimised/ner", 500)
    optimise_rel_dataset(stats_json, descriptvie_json, "../datasets/4_optimised/rel", 500)

    # corpus_docs = annotations_to_spacy("../datasets/4_optimised/rel/1_annotation_sets/optimised_rel.jsonl")
    # train_dev_test_split(corpus_docs,"../datasets/4_optimised/rel/2_preprocessed")

    # output_dir = "../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive_all/1_combined_before_splitted"
    # exclude_numerical_docs(f"{output_dir}/test.spacy", output_dir)
    # output_dir = "../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive_all/2_splitted_before_combined"
    # exclude_numerical_docs(f"{output_dir}/test.spacy", output_dir)


    output_dir = "../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive_all/1_combined_before_splitted"
    # separate_numerical_and_descriptive_docs(f"{output_dir}/test.spacy", output_dir)
    output_dir = "../datasets/3_stats_descriptive/2_preprocessed/corpus_stats_descriptive_all/2_splitted_before_combined"
    # separate_numerical_and_descriptive_docs(f"{output_dir}/test.spacy", output_dir)



