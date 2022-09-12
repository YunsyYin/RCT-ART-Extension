"""
Tabulates RCT abstract result sentences.
"""
import spacy
from spacy.tokens import DocBin, Doc
from spacy.training.example import Example
import operator, csv, os, io
from collections import defaultdict
import pandas as pd
from difflib import SequenceMatcher


# This function is adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def named_entity_recognition(ner_model, input_docs):
    """
    Takes list of docs and extracts named entities in doc text and adds them to each doc
    """
    print("|| Extracting entities")
    nlp = spacy.load(ner_model)  # the ner model is loaded here
    ent_processed_docs = []
    for doc in input_docs:
        input_text = doc.text  # doc.text extracts only text from doc file as input for model prediction
        try:
            doc.ents = nlp(input_text).ents  # model predicts entities and replaces the gold ents in doc
        except ValueError:
            print("Error assigning occurred in example of PMID", doc.user_data["pmid"])
        ent_processed_docs.append(doc)
    return ent_processed_docs


# This function is adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def relation_extraction(rel_model, input_docs, descriptive=False):
    """
    Takes list of docs and predicts entity pair relations, adding them to each doc
    """
    print("|| Extracting relations")
    if not descriptive:
        from evaluate_stats import joint_ner_rel_evaluate
    else:
        from evaluate_stats_des import joint_ner_rel_evaluate
    rel_nlp = spacy.load(rel_model)
    rel_processed_docs = []
    for doc in input_docs:
        doc._.rel = {}  # clear pre-exsiting rels in gold data
        for name, proc in rel_nlp.pipeline:  # take rel component from pipeline
            rel_preds = proc(doc)  # makes predicts probability of relation type between every entity pair
        rel_processed_docs.append(rel_preds)
    return rel_processed_docs


def tabulate_pico_entities_text(docs, descriptive=False):
    """
    Tabulates the predicted result text entities using the extracted relations
    """
    print("|| Tabulating multiple docs")

    intv_sum_dict = {"arm 1": set(), "arm 2": set()}
    oc_frames = []

    for doc in docs:

        # Create dictionaries for sorting entities into
        intv_dict = {"arm 1": set(), "arm 2": set()}
        meas_dict = defaultdict(lambda: defaultdict(set))
        comp_dict = defaultdict(lambda: defaultdict(set))
        ci_dict = defaultdict(lambda: defaultdict(set))    
        pval_dict = defaultdict(lambda: defaultdict(set))
        des_dict = defaultdict(lambda: defaultdict(set))
        arm_switch = False

        # Extract sorting infromation from relations into dictionaries
        for key in doc._.rel:
            rel = doc._.rel[key]  # get relation
            pred_rel = max(rel.items(), key=operator.itemgetter(1))  # select relation type with highest probability
            if pred_rel[1] > 0.5:  # include relation if above set threshold for probability
                head = [ent for ent in doc.ents if ent.start == key[0]][0] # get parent entity (intv or outcome)
                child = [ent for ent in doc.ents if ent.start == key[1]][0] # get child entity (meas or description)
                if pred_rel[0] == "A1_RES":
                    intv_dict["arm 1"].add(head.text)
                    if child.label_ == "MEAS":
                        meas_dict[key[1]]["arm"].add(1)
                    elif child.label_ == "DES":
                        des_dict[key[1]]["arm"].add(1)
                elif pred_rel[0] == "A2_RES":
                    intv_dict["arm 2"].add(head.text)
                    if child.label_ == "MEAS":
                        meas_dict[key[1]]["arm"].add(2)
                    elif child.label_ == "DES":
                        des_dict[key[1]]["arm"].add(2)
                elif pred_rel[0] == "OC_RES":
                    meas_dict[key[1]]["outcomes"].add(head.text)
                elif pred_rel[0] == "COMP_RES":
                    comp_dict[key[1]]["outcomes"].add(head.text)
                elif pred_rel[0] == "CI_RES":
                    ci_dict[key[1]]["outcomes"].add(head.text)
                elif pred_rel[0] == "PVAL_RES":
                    pval_dict[key[1]]["outcomes"].add(head.text)
                elif pred_rel[0] == "DES_RES":
                    des_dict[key[1]]["outcomes"].add(head.text)            

        # Assign intv into respective arms by similarity comparison
        arm_1 = ', '.join(sorted(str(x) for x in intv_dict["arm 1"])) # convert set to string
        arm_2 = ', '.join(sorted(str(x) for x in intv_dict["arm 2"])) # convert set to string

        if len(intv_sum_dict["arm 1"]) == 0 and len(intv_sum_dict["arm 2"]) == 0:
            intv_sum_dict["arm 1"].add(arm_1)
            intv_sum_dict["arm 2"].add(arm_2)

        elif len(intv_dict["arm 1"]) != 0 or len(intv_dict["arm 2"]) != 0:
            
            def similarity(str_1, str_2):
                stopwords = ["of", "the", "treated", "group", "groups", "subjects", "patients", "administration"]
                str_1_list = [word.replace("-treated", "") for word in str_1.lower().split() if not word.lower() in stopwords]
                str_2_list = [word.replace("-treated", "") for word in str_2.lower().split() if not word.lower() in stopwords]
                score = []
                if len(str_1) != 0 and len(str_2) != 0:
                    for word_1 in str_1_list:
                        for word_2 in str_2_list:
                            score.append(SequenceMatcher(None, word_1, word_2).ratio())
                else:
                    return 0
                return sum(score) / len(score)

            arm_similarity = {}
            arm_1_sum = ' '.join(sorted(str(x) for x in intv_sum_dict["arm 1"]))
            arm_2_sum = ' '.join(sorted(str(x) for x in intv_sum_dict["arm 2"]))
            arm_1_for_comp = ' '.join(sorted(str(x) for x in intv_dict["arm 1"]))
            arm_2_for_comp = ' '.join(sorted(str(x) for x in intv_dict["arm 2"]))
            arm_similarity["arm_1_arm_1_sum"] = similarity(arm_1_for_comp, arm_1_sum)
            arm_similarity["arm_1_arm_2_sum"] = similarity(arm_1_for_comp, arm_2_sum)
            arm_similarity["arm_2_arm_1_sum"] = similarity(arm_2_for_comp, arm_1_sum)
            arm_similarity["arm_2_arm_2_sum"] = similarity(arm_2_for_comp, arm_2_sum)
            max_similarity = max(arm_similarity, key=arm_similarity.get)
            max_score = arm_similarity[max_similarity]

            if len(intv_dict["arm 1"]) != 0 and len(intv_dict["arm 2"]) != 0:
                if max_similarity == "arm_1_arm_1_sum" or max_similarity == "arm_2_arm_2_sum":
                    intv_sum_dict["arm 1"].add(arm_1)
                    intv_sum_dict["arm 2"].add(arm_2)
                else: 
                    intv_sum_dict["arm 1"].add(arm_2)
                    intv_sum_dict["arm 2"].add(arm_1)
                    arm_switch = True
            else:
                if (max_similarity == "arm_1_arm_1_sum" or max_similarity == "arm_2_arm_2_sum") and max_score >= 0.5:
                    intv_sum_dict["arm 1"].add(arm_1)
                    intv_sum_dict["arm 2"].add(arm_2)
                elif (max_similarity == "arm_1_arm_2_sum" or max_similarity == "arm_2_arm_1_sum") and max_score < 0.5:
                    intv_sum_dict["arm 1"].add(arm_1)
                    intv_sum_dict["arm 2"].add(arm_2)
                else:              
                    intv_sum_dict["arm 1"].add(arm_2)
                    intv_sum_dict["arm 2"].add(arm_1)
                    arm_switch = True  

        # Seperate measures into a dictionary of respective outcomes and arms
        oc_dict = defaultdict(lambda: defaultdict(set))
        for k, v in meas_dict.copy().items():
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
            if v["arm"] == {1,2}:
                oc_dict[outcome]["arm_1"].add(meas)
                oc_dict[outcome]["arm_2"].add(meas)
                meas_dict.pop(k)
            elif v["arm"] == {1}:
                if not arm_switch:
                    oc_dict[outcome]["arm_1"].add(meas)
                else:
                    oc_dict[outcome]["arm_2"].add(meas)
                meas_dict.pop(k)
            elif v["arm"] == {2}:
                if not arm_switch:
                    oc_dict[outcome]["arm_2"].add(meas)
                else:
                    oc_dict[outcome]["arm_1"].add(meas)
                meas_dict.pop(k)

        # Seperate comparative statistics into a dictionary of respective outcomes
        for k, v in comp_dict.copy().items():
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            comp = [ent.text for ent in doc.ents if ent.start == k][0]  # get full comparative statistic entity
            oc_dict[outcome]["comp"].add(comp)
            comp_dict.pop(k)

        # Seperate 95% CI into a dictionary of respective comparative statistics
        for k, v in ci_dict.copy().items():
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            ci = [ent.text for ent in doc.ents if ent.start == k][0]  # get full 95% CI entity
            oc_dict[outcome]["ci"].add(ci)
            ci_dict.pop(k)

        # Seperate p-values into a dictionary of respective outcomes
        for k, v in pval_dict.copy().items():
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            pval = [ent.text for ent in doc.ents if ent.start == k][0]  # get full p-value entity
            oc_dict[outcome]["pval"].add(pval)
            pval_dict.pop(k)

        # Seperate descriptions into a dictionary of respective outcomes and arms
        for k, v in des_dict.copy().items():
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            description = [ent.text for ent in doc.ents if ent.start == k][0]  # get full description entity
            if v["arm"] == {1,2}:
                oc_dict[outcome]["arm_1"].add(description)
                oc_dict[outcome]["arm_2"].add(description)
                des_dict.pop(k)
            elif v["arm"] == {1}:
                if not arm_switch:
                    oc_dict[outcome]["arm_1"].add(description)
                else:
                    oc_dict[outcome]["arm_2"].add(description)
                des_dict.pop(k)
            elif v["arm"] == {2}:
                if not arm_switch:
                    oc_dict[outcome]["arm_2"].add(description)
                else:
                    oc_dict[outcome]["arm_1"].add(description)
                des_dict.pop(k)
            else:
                oc_dict[outcome]["des"].add(description)

        # Sort measures with no associated intv in sentences with intvs
        for k, v in meas_dict.copy().items(): 
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
            if "arm_1" and "arm_2" in oc_dict.copy()[outcome]:
                oc_dict[outcome + ", total study group"]["arm_1"].add(meas)
                oc_dict[outcome + ", total study group"]["arm_2"].add(meas)
                meas_dict.pop(k)
            elif "arm_1" in oc_dict.copy()[outcome]:  # add measures to opposite arm if intv name missing
                oc_dict[outcome]["arm_2"].add(meas)
                meas_dict.pop(k)
            elif "arm_2" in oc_dict.copy()[outcome]:
                oc_dict[outcome]["arm_1"].add(meas)
                meas_dict.pop(k)

        # Sort measures in sentences with no intv by first mention
        for k, v in sorted(meas_dict.copy().items()): 
            outcome = ', '.join(sorted(str(x) for x in v["outcomes"]))
            meas = [ent.text for ent in doc.ents if ent.start == k][0]  # get full measure entity
            if len(intv_dict) == 2:
                oc_dict[outcome + ", total study group"]["arm_1"].add(meas)
                oc_dict[outcome + ", total study group"]["arm_2"].add(meas)
                meas_dict.pop(k)
            elif outcome in oc_dict.copy():
                oc_dict[outcome]["arm_2"].add(meas)
                meas_dict.pop(k)
            else:
                oc_dict[outcome]["arm_1"].add(meas)
                meas_dict.pop(k)

        # Mark empty cells as not reported (NR) and create outcome rows of table 
        for oc in oc_dict:
            if "arm_1" not in oc_dict[oc]:
                oc_dict[oc]["arm_1"].add("NR") # if meas missing, then included as not reported (NR)
            if "arm_2" not in oc_dict[oc]:
                oc_dict[oc]["arm_2"].add("NR")
            if "comp" not in oc_dict[oc]:
                oc_dict[oc]["comp"].add("NR")   
            if "ci" not in oc_dict[oc]:
                oc_dict[oc]["ci"].add("NR") 
            if "pval" not in oc_dict[oc]:
                oc_dict[oc]["pval"].add("NR") 
            if "des" not in oc_dict[oc]:
                oc_dict[oc]["des"].add("NR")

            m_arm_1 = ', '.join(sorted(str(x) for x in oc_dict[oc]["arm_1"]))  # convert set to string
            m_arm_2 = ', '.join(sorted(str(x) for x in oc_dict[oc]["arm_2"]))  # convert set to string
            comp_str = ', '.join(sorted(str(x) for x in oc_dict[oc]["comp"]))  # convert set to string
            ci_str = ', '.join(sorted(str(x) for x in oc_dict[oc]["ci"]))  # convert set to string
            pval_str = ', '.join(sorted(str(x) for x in oc_dict[oc]["pval"]))  # convert set to string
            des_str = ', '.join(sorted(str(x) for x in oc_dict[oc]["des"]))  # convert set to string

            if not descriptive:
                oc_row = pd.DataFrame([[oc, m_arm_1, m_arm_2, comp_str, ci_str, pval_str]], columns=["outcome", "arm 1", "arm 2", "comparative statistic", "95% CI", "p-value"])
            else:
                oc_row = pd.DataFrame([[oc, m_arm_1, m_arm_2, comp_str, ci_str, pval_str, des_str]], columns=["outcome", "arm 1", "arm 2", "comparative statistic", "95% CI", "p-value", "note"])
            oc_frames.append(oc_row)

    # Create intv entity row of table
    arm_1_sum = ', '.join(sorted(str(x) for x in intv_sum_dict["arm 1"] if len(x) != 0))
    arm_2_sum = ', '.join(sorted(str(x) for x in intv_sum_dict["arm 2"] if len(x) != 0))
    if not descriptive:
        intv_frame = pd.DataFrame([["intervention", arm_1_sum, arm_2_sum, "", "", ""]], columns=["outcome", "arm 1", "arm 2", "comparative statistic", "95% CI", "p-value"])     
    else:
        intv_frame = pd.DataFrame([["intervention", arm_1_sum, arm_2_sum, "", "", "", ""]], columns=["outcome", "arm 1", "arm 2", "comparative statistic", "95% CI", "p-value", "note"])     
    if oc_frames:
        oc_frame = pd.concat(oc_frames)
        table = pd.concat([intv_frame, oc_frame])
    else:
        table = intv_frame
    return table
    

# This function was extensively adapted from RCT-ART
# template: https://github.com/jetsunwhitton/RCT-ART
def output_csvs(dataframes, output_loc):
    """
    Outputs list of dataframes as csv files
    """
    num = 0
    for df in dataframes:
        with io.open(f"{output_loc}/result_tab_{num}.csv", 'w', encoding='utf-8') as output:
            df.to_csv(output)
            num += 1


def tabulate_gold_and_pred_tables(doc_loc, ner_model, rel_model, gold_output_loc, pred_output_loc, descriptive=False):
    """
    Outputs gold and pred tables
    """
    # Used for custom models
    if not descriptive:
        from evaluate_stats import joint_ner_rel_evaluate
    else:
        from evaluate_stats_des import joint_ner_rel_evaluate
    nlp = spacy.load(rel_model)
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)

    # Output gold tables
    docs_gold = doc_bin.get_docs(nlp.vocab)
    gold_num = 0
    for doc in docs_gold:
        gold_num += 1
        try:
            pmid = doc.user_data["pmid"]
        except KeyError:
            pmid = "woPMID" 
        with io.open(f"{gold_output_loc}/result_tab_{gold_num}_{pmid}.csv", 'w', encoding='utf-8') as output:
            tabulate_pico_entities(doc, descriptive=descriptive).to_csv(output)   

    # Output pred tables
    docs_pred = doc_bin.get_docs(spacy.blank("en").vocab)
    ner_preds = named_entity_recognition(ner_model, docs_pred)
    rel_preds = relation_extraction(rel_model, ner_preds, descriptive=descriptive)
    pred_num = 0
    for doc in rel_preds:
        pred_num += 1
        try:
            pmid = doc.user_data["pmid"]
        except KeyError:
            pmid = "woPMID" 
        with io.open(f"{pred_output_loc}/result_tab_{pred_num}_{pmid}.csv", 'w', encoding='utf-8') as output:
            tabulate_pico_entities(doc, descriptive=descriptive).to_csv(output)   


def tabulate_gold_and_pred_tables_from_text(doc_loc, ner_model, rel_model, gold_output_loc, pred_output_loc, descriptive=False):
    """
    Outputs gold and pred tables from integrated sentences with the same PMID
    """
    if not descriptive:
        from evaluate_stats import joint_ner_rel_evaluate
    else:
        from evaluate_stats_des import joint_ner_rel_evaluate
    nlp = spacy.load(rel_model)
    doc_bin = DocBin(store_user_data=True).from_disk(doc_loc)
    docs_gold = doc_bin.get_docs(nlp.vocab)
    docs_pred = doc_bin.get_docs(spacy.blank("en").vocab)
    pmid_dict = {}

    # Create a PMID dict with the corresponding number of sentence examples
    for doc in docs_gold:
        try:
            if doc.user_data["pmid"] not in pmid_dict:
                pmid_dict[doc.user_data["pmid"]] = 1
            else:
                pmid_dict[doc.user_data["pmid"]] += 1
        except KeyError:
            pass

    # Keep PMIDs with more than one example sentence
    for k, v in pmid_dict.copy().items():
        if v <= 1:
            del pmid_dict[k]    
        
    # Output gold tables
    for k, v in pmid_dict.items():
        docs_text = []
        for doc in doc_bin.get_docs(nlp.vocab):
            try:
                if doc.user_data["pmid"] == k:
                    docs_text.append(doc)
            except KeyError:
                pass 
        with io.open(f"{gold_output_loc}/result_tab_{k}.csv", 'w', encoding='utf-8') as output:
            tabulate_pico_entities_text(docs_text, descriptive=descriptive).to_csv(output)   

    # Output pred tables
    ner_preds = named_entity_recognition(ner_model, docs_pred)
    rel_preds = relation_extraction(rel_model, ner_preds, descriptive=descriptive)
    for k, v in pmid_dict.items():
        docs_text = []
        for doc in rel_preds:
            try:
                if doc.user_data["pmid"] == k:
                    docs_text.append(doc)
            except KeyError:
                pass     
        with io.open(f"{pred_output_loc}/result_tab_{k}.csv", 'w', encoding='utf-8') as output:
            tabulate_pico_entities_text(docs_text, descriptive=descriptive).to_csv(output)   



if __name__ == "__main__":

    # Tabulate gold and pred tables in test dataset for corpus_stats_descriptive
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
    #         tabulate_gold_and_pred_tables(doc_loc, ner_model, rel_model, f"{gold_table}/{dataset}", f"{pred_table}/{dataset}", descriptive=True)
    #         tabulate_gold_and_pred_tables_from_text(doc_loc, ner_model, rel_model, f"{gold_table_text}/{dataset}", f"{pred_table_text}/{dataset}", descriptive=True)

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
            tabulate_gold_and_pred_tables(doc_loc, ner_model, rel_model, gold_table, pred_table, descriptive=True)
            tabulate_gold_and_pred_tables_from_text(doc_loc, ner_model, rel_model, gold_table_text, pred_table_text, descriptive=True)

            