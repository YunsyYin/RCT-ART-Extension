import spacy_streamlit, spacy, operator
import streamlit as st
from spacy import displacy
from spacy.tokens import DocBin
from spacy.pipeline import merge_entities
from spacy.lang.en import English
from tabulate import relation_extraction, tabulate_pico_entities_text
import base64


# set page config   
st.set_page_config(
	page_title="RCT-ART Extension",
    page_icon="logo_icon.gif"
)

st.sidebar.image("logo.jpg")
st.sidebar.markdown("RCT-ART Extension is an NLP pipeline built with spaCy for converting clinical trial result text into tables through jointly extracting intervention, outcome, outcome measure and comparative statistic entities and their relations. ")
st.sidebar.subheader("Current constraints:")
st.sidebar.markdown("""
                    - Only abstracts from studies with 2 trial arms
                    - Must be text with study results
                    - Must contain at least least one intervention (e.g. drug name), outcome description (e.g. blood pressure) and non-comparative outcome measure)
                    """)
st.title("RCT-ART Extension Demo")
st.header("Randomised Controlled Trial Abstract Result Tabulator")

# Define the model and default text
ner_model = "trained_models/stats/ner/model-best"
rel_model = "trained_models/stats/rel/model-best"

default_text = "The primary endpoint was reached by 51 of 73 children taking ivabradine (70%) versus 5 of 41 taking placebo (12%) at varying doses (odds ratio: 17.24; p < 0.0001). Between baseline and 12 months, there was a greater increase in left ventricular ejection fraction in patients taking ivabradine than placebo (13.5% vs. 6.9%; p = 0.024). "
default_text = "The percentage of patients who had a hematologic complete response was significantly higher in the daratumumab group than in the control group (53.3% vs. 18.1%) (relative risk ratio, 2.9; 95% confidence interval [CI], 2.1 to 4.1; P<0.001). Systemic administration-related reactions to daratumumab occurred in 7.3% of the patients. "
default_text = "Systemic administration-related reactions to daratumumab occurred in 7.3% of the patients. The percentage of patients who had a hematologic complete response was significantly higher in the daratumumab group than in the control group (53.3% vs. 18.1%) (relative risk ratio, 2.9; 95% confidence interval [CI], 2.1 to 4.1; P<0.001).  "

# Enter result section
st.subheader("Enter result text for analysis")
text = st.text_area("Input should follow constraints outlined in sidebar", default_text, height=200).strip()

nlp_sent = English()
nlp_sent.add_pipe("sentencizer")
sent_doc = nlp_sent(text)

ent_doc_bin = DocBin()
rel_doc_bin = DocBin()
    
nlp_ner = spacy.load(ner_model)
print('Pipelines: ', nlp_ner.pipe_names)  # Pipelines: ['transformer', 'ner']

for sent in sent_doc.sents:
    print(type(sent.text))
    ent_doc = nlp_ner(sent.text)
    ent_doc_bin.add(ent_doc)  

ent_docs = ent_doc_bin.get_docs(nlp_ner.vocab)

# NER analysis section
st.subheader("NER analysis")

for ent_doc in ent_docs:

    spacy_streamlit.visualize_ner(
        ent_doc,
        labels=["INTV", "OC", "MEAS", "COMP", "PVAL", "CI"],
        show_table=False,
        title=False,
        key=str(ent_doc.text)
    )

    # RE process
    rel_doc = relation_extraction(rel_model,[ent_doc])[0]  # type: spacy.tokens.doc.Doc
    rel_doc_bin.add(rel_doc)

    deps = {"words": [],"arcs": []}  # store the relation in dict

    for tok in rel_doc:
        deps["words"].append({"text": tok.text, "tag": tok.ent_type_})

    for key in rel_doc._.rel:
        rel = rel_doc._.rel[key]  # get relation
        pred_rel = max(rel.items(), key=operator.itemgetter(1))  # selects relation type with highest probability
        if pred_rel[1] > 0.5:  # includes relation if above set threshold for probability
            if key[0] > key[1] and rel_doc[key[1]].ent_type_ != "MEAS":
                deps["arcs"].append({"start": key[1], "end": key[0], "label":  pred_rel[0], "dir": "right"})
            elif key[0] > key[1]:
                deps["arcs"].append({"start": key[1], "end": key[0], "label": pred_rel[0], "dir": "left"})
            elif rel_doc[key[1]].ent_type_ != "MEAS":
                deps["arcs"].append({"start": key[0], "end": key[1], "label": pred_rel[0], "dir": "right"})
            else:
                deps["arcs"].append({"start": key[0], "end": key[1], "label": pred_rel[0], "dir": "right"})

    html = displacy.render(deps, style="dep", manual=True, options={'distance':80}) # Squre shaped -> options={'compact': True}

    # RE analysis section
    st.subheader("RE analysis")
    st.write(spacy_streamlit.util.get_svg(html), unsafe_allow_html=True)

nlp_rel = spacy.load(rel_model)
rel_docs = rel_doc_bin.get_docs(nlp_rel.vocab)
df = tabulate_pico_entities_text(rel_docs)
# print('Relation: ', rel_doc._.rel)

# Define tabulation format
heading_properties = [('font-size', '12px')]
cell_properties = [('font-size', '12px')]
dfstyle = [dict(selector="th", props=heading_properties),dict(selector="td", props=cell_properties)]

#df.style.set_table_styles([cell_hover, index_names, headers])

#Tabulation section
st.subheader("Tabulation")
st.table(df.style.set_table_styles(dfstyle))

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="result_sentence.csv">Download csv file</a>'
    return href

st.markdown(get_table_download_link(df), unsafe_allow_html=True)
