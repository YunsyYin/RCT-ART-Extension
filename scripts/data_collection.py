import csv, json, codecs, os
from spacy.lang.en import English
from Bio import Entrez


def csv_to_jsonl(csv_loc, filename, output_dir):
    """
    Convert the csv data format (with single spreadsheet) into the jsonl format used by the prodigy annotation software.
    :csv_loc: the location of csv file
    :filename: the filename of the output jsonl file 
    :output_dir: the output path of the jsonl file
    :return: output a jsonl file
    """
    text_list = []
    sent_list = []

    # Open csv reader and convert each row into a list 
    with codecs.open(csv_loc, encoding='utf-8', errors='ignore') as csvfile:
        csvReader = csv.DictReader(csvfile)
        for row in csvReader:
            text_list.append(row)

    # Split the text into sentences using sentencizer
    nlp_sent = English()
    nlp_sent.add_pipe("sentencizer")

    for i in range(len(text_list)):
        pmid_dict = {"pmid":str}
        pmid_dict["pmid"] = text_list[i]["pmid"]
        sent_doc = nlp_sent(text_list[i]["text"])
        for sent in sent_doc.sents:
            sent_dict = {"text":str, "user_data":dict}
            sent_dict["text"] = sent.text
            sent_dict["user_data"] = pmid_dict
            sent_list.append(sent_dict)
        
    # Open json writer and dump data
    with open(f"{output_dir}/{filename}.jsonl", 'w', encoding='utf-8') as jsonfile:
        for row in sent_list:
            jsonfile.write(json.dumps(row) + "\n")


def update_pmid(json_loc, filename, output_dir):
    """
    Query pubmed database for pmids and update the jsonl file
    :json_loc: the location of jsonl file
    :filename: the filename of the output jsonl file 
    :output_dir: the output path of the jsonl file
    :return: output a jsonl file
    """
    stopwords = ["a", "about", "again", "all", "almost", "also", "although", "always", "among", "an", "and", "another", "any", "are", "as", "at", 
                 "be", "because", "been", "before", "being", "between", "both", "but", "by", 
                 "can", "could", "did", "do", "does", "done", "due", "during", 
                 "each", "either", "enough", "especially", "etc", "for", "found", "from", "further", 
                 "had", "has", "have", "having", "here", "how", "however", "i", "if", "in", "into", "is", "it", "its", "itself", "just", 
                 "kg", "km", "made", "mainly", "make", "may", "mg", "might", "ml", "mm", "most", "mostly", "must", "nearly", "neither", "no", "nor", 
                 "obtained", "of", "often", "on", "our", "overall", "perhaps", "pmid", "quite", "rather", "really", "regarding", 
                 "seem", "seen", "several", "should", "show", "showed", "shown", "shows", "significantly", "since", "so", "some", "such", 
                 "than", "that", "the", "their", "theirs", "them", "then", "there", "therefore", "these", "they", "this", "those", "through", "thus", "to",
                 "upon", "various", "very", "was", "we", "were", "what", "when", "which", "while", "with", "within", "without", "would",
                 "%", ",", ".", ";", ":", "(", ")", "[", "]", "/", "=", ">", "<", "+", "-", "±", "/-"]
    count = 0
    new_examples = []
    pmid_not_found = []

    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            count += 1
            example = json.loads(line)
            query = example["text"]
            Entrez.email = 'yun-tzu.yin.21@ucl.ac.uk'
            handle = Entrez.esearch(db="pubmed", term=query, retmax="2")
            record = Entrez.read(handle)
                
            if len(record["IdList"]) != 1:  
                sent_list = []
                sent_list = ['"'+word["text"]+'"[All Fields]' for word in example["tokens"] if not word["text"].lower() in stopwords]
                query = " AND ".join(sent_list)
                handle = Entrez.esearch(db="pubmed", term=query, retmax="2")
                record = Entrez.read(handle)    

                if len(record["IdList"]) != 1:      
                    sent_list = []
                    sent_list = [word["text"] for word in example["tokens"] if not word["text"].lower() in stopwords]
                    query = " ".join(sent_list)
                    handle = Entrez.esearch(db="pubmed", term=query, retmax="2")
                    record = Entrez.read(handle)  
            
            if len(record["IdList"]) == 1:
                pmid_dict = {}
                pmid_dict["pmid"] = record["IdList"][0]
                example["user_data"] = pmid_dict             
            else:
                pmid_not_found.append(count)

            try:
                del example["versions"]
            except KeyError:
                pass

            new_examples.append(example)

    # Open json writer and dump data
    with open(f"{output_dir}/{filename}.jsonl", 'w', encoding='utf-8') as jsonfile:
        for row in new_examples:
            jsonfile.write(json.dumps(row) + "\n")
                
    pmid_not_found_str = ", ".join(str(count) for count in pmid_not_found)
    print("PMID not found: ", pmid_not_found_str)
         

def retrieve_result_text_from_pubmed(json_loc, output_dir):
    """
    Query pubmed database by pmids in the jsonl file and return the result text of RCT abstracts as a jsonl file
    :json_loc: the location of jsonl file
    :output_dir: the output path of the jsonl file
    :return: output a jsonl file
    """
    domain_sent_dict = {"autism": set(),
                        "blood_cancer": set(),
                        "cardiovascular_disease": set(),
                        "diabetes": set(),
                        "glaucoma": set(),
                        "solid_tumour_cancer": set()
                        }
    domain_pmid_dict = {"autism": set(),
                        "blood_cancer": set(),
                        "cardiovascular_disease": set(),
                        "diabetes": set(),
                        "glaucoma": set(),
                        "solid_tumour_cancer": set()
                        }
    domain_result_dict = {"autism": list(),
                          "blood_cancer": list(),
                          "cardiovascular_disease": list(),
                          "diabetes": list(),
                          "glaucoma": list(),
                          "solid_tumour_cancer": list()
                          }

    # Withdraw sentences from each domain in the gold corpus                    
    for domain in domain_sent_dict.keys():
        with open(f"../datasets/0_gold_corpus/domains/{domain}.jsonl", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)    
                domain_sent_dict[domain].add(example["text"])

    # Categorise the updated pmids based on the sentences in each domain  
    with open(json_loc, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            for domain in domain_sent_dict.keys():
                if example["text"] in domain_sent_dict[domain]:
                    try:
                        if example["user_data"]["pmid"] != "":
                            domain_pmid_dict[domain].add(example["user_data"]["pmid"])
                    except KeyError:
                        pass
    
    # Retrieve result text of each doamin 
    for domain in domain_pmid_dict.keys():
        pmids = ','.join(sorted(str(x) for x in domain_pmid_dict[domain]))
        Entrez.email = 'yun-tzu.yin.21@ucl.ac.uk'
        handle = Entrez.efetch(db="pubmed", id=pmids, retmode="xml")
        record = Entrez.read(handle)
        for article in record["PubmedArticle"]:
            # pmid_dict = {"pmid":str}
            text_dict = {"text":str, "user_data": {"pmid":str}}
            text_dict["user_data"]["pmid"] = str(article["MedlineCitation"]["PMID"])
            for text in article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"]:
                try:
                    if text.attributes["NlmCategory"] == "RESULTS":
                        text_dict["text"] = str(text)
                        domain_result_dict[domain].append(text_dict)
                        break
                except KeyError:
                    pass
        print(domain,"PMIDs sent:", len(record["PubmedArticle"]), "Texts retrieved", len(domain_result_dict[domain]))
        
        # Open json writer and dump data
        with open(f"{output_dir}/{domain}.jsonl", 'w', encoding='utf-8') as jsonfile:
            for row in domain_result_dict[domain]:
                jsonfile.write(json.dumps(row) + "\n")


def split_text_to_sentence(json_loc, output_dir):
    nlp_sent = English()
    nlp_sent.add_pipe("sentencizer")
    domain_sent_dict = {"autism": list(),
                        "blood_cancer": list(),
                        "cardiovascular_disease": list(),
                        "diabetes": list(),
                        "glaucoma": list(),
                        "solid_tumour_cancer": list()
                        }

    # Split the text into sentences using sentencizer
    for domain in domain_sent_dict.keys():
        with open(f"{json_loc}/{domain}.jsonl", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                pmid_dict = {"pmid":str}
                pmid_dict["pmid"] = example["user_data"]["pmid"]
                sent_doc = nlp_sent(example["text"])
                for sent in sent_doc.sents:
                    sent_dict = {"text":str, "user_data":pmid_dict}
                    sent_dict["text"] = sent.text
                    domain_sent_dict[domain].append(sent_dict)
        print(domain, len(domain_sent_dict[domain]))

        # Open json writer and dump data
        with open(f"{output_dir}/{domain}.jsonl", 'w', encoding='utf-8') as jsonfile:
            for row in domain_sent_dict[domain]:
                jsonfile.write(json.dumps(row) + "\n")


def exclude_annotated_sentences(json_loc, output_dir):
    annotated_dict = {"autism": list(),
                      "blood_cancer": list(),
                      "cardiovascular_disease": list(),
                      "diabetes": list(),
                      "glaucoma": list(),
                      "solid_tumour_cancer": list()
                      }
    unannotated_dict = {"autism": list(),
                        "blood_cancer": list(),
                        "cardiovascular_disease": list(),
                        "diabetes": list(),
                        "glaucoma": list(),
                        "solid_tumour_cancer": list()
                        }
      
    for domain in annotated_dict.keys():

        # Withdraw sentences from each domain in the gold corpus 
        with open(f"../datasets/0_gold_corpus/domains/{domain}.jsonl", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                annotated_sent = "".join(example["text"].split())
                if annotated_sent[-1:] != ".":
                    annotated_sent = annotated_sent + "."
                if annotated_sent[:7] == "RESULTS":
                    annotated_sent = annotated_sent[7:]
                annotated_dict[domain].append(annotated_sent)
      
        # Exclude sentences existing in gold corpus
        with open(f"{json_loc}/{domain}.jsonl", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                sent = "".join(example["text"].split())
                if sent not in annotated_dict[domain]:
                    sent_dict = {"text":str, "user_data": {"pmid":str}}
                    sent_dict["text"] = example["text"]
                    sent_dict["user_data"]["pmid"] = example["user_data"]["pmid"]
                    unannotated_dict[domain].append(sent_dict)

        # Open json writer and dump data
        with open(f"{output_dir}/{domain}.jsonl", 'w', encoding='utf-8') as jsonfile:
            for row in unannotated_dict[domain]:
                jsonfile.write(json.dumps(row) + "\n")


def remove_rejected_text(json_loc, output_dir):
    for domain in os.listdir(json_loc):
        accept_list = []
        with open(f"{json_loc}/{domain}", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                if example["answer"] == "accept":
                    accept_list.append(example)
        
        # Open json writer and dump data
        with open(f"{output_dir}/{domain}", 'w', encoding='utf-8') as jsonfile:
            for row in accept_list:
                jsonfile.write(json.dumps(row) + "\n")


def combine_jsonl_files(json_loc, output_dir):
    accept_list = []
    for domain in os.listdir(json_loc):
        with open(f"{json_loc}/{domain}", "r", encoding="utf8") as jsonfile:
            for line in jsonfile:
                example = json.loads(line)
                accept_list.append(example)
        
    # Open json writer and dump data
    with open(f"{output_dir}/combined.jsonl", 'w', encoding='utf-8') as jsonfile:
        for row in accept_list:
            jsonfile.write(json.dumps(row) + "\n")    



if __name__ == "__main__":
    domains = ["cardiology","cardiomyopathy","infections","nephrology","oncology"]
    csv_loc = "../datasets/5_paediatric/0_data_collection/1_result_texts_csv"
    output_dir = "../datasets/5_paediatric/0_data_collection/2_result_sentences"
    for domain in domains:
        csv_to_jsonl(f"{csv_loc}/{domain}.csv", domain, output_dir)

    # json_loc = "../datasets/4_gold_tables_from_text/pmid_updated"
    # filename = "corpus_stats_2"
    # output_dir = "../datasets/4_gold_tables_from_text/pmid_updated"
    # update_pmid(f"{json_loc}/{filename}.jsonl", f"{filename}_pmid_updated", output_dir)

    # json_loc = "../datasets/4_gold_tables_from_text/pmid_updated/corpus_stats_2_pmid_updated.jsonl"
    # output_dir = "../datasets/6_data_collection/text/domains"
    # retrieve_result_text_from_pubmed(json_loc, output_dir)

    # json_loc = "../datasets/6_data_collection/text/domains"
    # output_dir = "../datasets/6_data_collection/sentence/all/domains"
    # split_text_to_sentence(json_loc, output_dir)

    # json_loc = "../datasets/6_data_collection/sentence/all/domains"
    # output_dir = "../datasets/6_data_collection/sentence/filtered/domains"
    # exclude_annotated_sentences(json_loc, output_dir)

    # json_loc = "../datasets/6_data_collection/sentence/4_rel_annotated"
    # output_dir = "../datasets/6_data_collection/sentence/5_rel_annotated_accept"
    # remove_rejected_text(json_loc, output_dir)

    # json_loc = "../datasets/6_data_collection/sentence/5_rel_annotated_accept/domains"
    # output_dir = "../datasets/6_data_collection/sentence/6_corpus_des"
    # combine_jsonl_files(json_loc, output_dir)

    # json_loc = "../datasets/2_stats_descriptive/1_annotation_sets/domains"
    # output_dir = "../datasets/2_stats_descriptive/1_annotation_sets"
    # combine_jsonl_files(json_loc, output_dir)

    # json_loc = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2_cardio_tumour/domains"
    # output_dir = "../datasets/2_stats_descriptive/1_annotation_sets/corpus_des_2_cardio_tumour"
    # combine_jsonl_files(json_loc, output_dir)

    # json_loc = "../datasets/3_descriptive/1_annotation_sets/corpus_descriptive/domains"
    # output_dir = "../datasets/3_descriptive/1_annotation_sets/corpus_descriptive"
    # combine_jsonl_files(json_loc, output_dir)
