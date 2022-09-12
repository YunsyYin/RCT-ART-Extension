"""
This script defines the relation extractor component for use as part of a spaCy pipeline.
It includes functions for training the relation extraction model, as well as for evaluating
its performance both alone and jointly after named entity recognition
"""
from itertools import islice
from typing import Tuple, List, Iterable, Optional, Dict, Callable, Any

from spacy.scorer import PRFScore
from thinc.types import Floats2d
import numpy, operator
from spacy.training.example import Example
from thinc.api import Model, Optimizer
from spacy.tokens.doc import Doc
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.vocab import Vocab
from spacy import Language
from thinc.model import set_dropout_rate
from wasabi import Printer


Doc.set_extension("rel", default={}, force=True)
msg = Printer()


# This object is adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
@Language.factory(
    "relation_extractor",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
        "a1_res_p": None,
        "a1_res_r": None,
        "a1_res_f": None,
        "a2_res_p": None,
        "a2_res_r": None,
        "a2_res_f": None,
        "oc_res_p": None,
        "oc_res_r": None,
        "oc_res_f": None,
        "comp_res_p": None,
        "comp_res_r": None,
        "comp_res_f": None,
        "ci_res_p": None,
        "ci_res_r": None,
        "ci_res_f": None, 
        "pval_res_p": None,
        "pval_res_r": None,
        "pval_res_f": None,
        "des_res_p": None,
        "des_res_r": None,
        "des_res_f": None, 
    },
)


# This object is sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
def make_relation_extractor(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a RelationExtractor component."""
    return RelationExtractor(nlp.vocab, model, name, threshold=threshold)


# This class is sourced from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
class RelationExtractor(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize a relation extractor."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {"labels": [], "threshold": threshold}

    @property
    def labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["labels"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_label(self, label: str) -> int:
        """Add a new label to the pipe."""
        if not isinstance(label, str):
            raise ValueError("Only strings can be added as labels to the RelationExtractor")
        if label in self.labels:
            return 0
        self.cfg["labels"] = list(self.labels) + [label]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        """Apply the pipe to a Doc."""
        # check that there are actually any candidate instances in this batch of examples
        total_instances = len(self.model.attrs["get_instances"](doc))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc - returning doc as is.")
            return doc

        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def predict(self, docs: Iterable[Doc]) -> Floats2d:
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        get_instances = self.model.attrs["get_instances"]
        total_instances = sum([len(get_instances(doc)) for doc in docs])
        if total_instances == 0:
            msg.info("Could not determine any instances in any docs - can not make any predictions.")
        scores = self.model.predict(docs)
        return self.model.ops.asarray(scores)

    def set_annotations(self, docs: Iterable[Doc], scores: Floats2d) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        c = 0
        get_instances = self.model.attrs["get_instances"]
        for doc in docs:
            for (e1, e2) in get_instances(doc):
                offset = (e1.start, e2.start)
                if offset not in doc._.rel:
                    doc._.rel[offset] = {}
                for j, label in enumerate(self.labels):
                    doc._.rel[offset][label] = scores[c, j]
                c += 1

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # check that there are actually any candidate instances in this batch of examples
        total_instances = 0
        for eg in examples:
            total_instances += len(self.model.attrs["get_instances"](eg.predicted))
        if total_instances == 0:
            msg.info("Could not determine any instances in doc.")
            return losses

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient ** 2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                relations = example.reference._.rel
                for indices, label_dict in relations.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError("Call begin_training with relevant entities and relations "
                             "annotated in at least a few reference examples!")
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return score_relations(examples, self.threshold)


# This function is adapted from the spaCy relation component
# template: https://github.com/explosion/projects/tree/v3/tutorials
def score_relations(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score rel in a batch of examples."""
    micro_prf = PRFScore()
    a1_res_prf = PRFScore()
    a2_res_prf = PRFScore()
    oc_res_prf = PRFScore()
    comp_res_prf = PRFScore()
    ci_res_prf = PRFScore()
    pval_res_prf = PRFScore()
    des_res_prf = PRFScore()
    for example in examples:
        gold = example.reference._.rel
        pred = example.predicted._.rel
        for key, pred_dict in pred.items():
            gold_labels = [k for (k, v) in gold.get(key, {}).items() if v == 1.0]
            for k, v in pred_dict.items():
                if v >= threshold:
                    if k in gold_labels:
                        micro_prf.tp += 1
                        if k == "A1_RES": a1_res_prf.tp += 1
                        elif k == "A2_RES": a2_res_prf.tp += 1
                        elif k == "OC_RES": oc_res_prf.tp += 1
                        elif k == "COMP_RES": comp_res_prf.tp += 1
                        elif k == "CI_RES": ci_res_prf.tp += 1   
                        elif k == "PVAL_RES": pval_res_prf.tp += 1
                        elif k == "DES_RES": des_res_prf.tp += 1                          
                    else:
                        micro_prf.fp += 1
                        if k == "A1_RES": a1_res_prf.fp += 1
                        elif k == "A2_RES": a2_res_prf.fp += 1
                        elif k == "OC_RES": oc_res_prf.fp += 1
                        elif k == "COMP_RES": comp_res_prf.fp += 1                         
                        elif k == "CI_RES": ci_res_prf.fp += 1 
                        elif k == "PVAL_RES": pval_res_prf.fp += 1
                        elif k == "DES_RES": des_res_prf.fp += 1                       
                else:
                    if k in gold_labels:
                        micro_prf.fn += 1
                        if k == "A1_RES": a1_res_prf.fn += 1
                        elif k == "A2_RES": a2_res_prf.fn += 1
                        elif k == "OC_RES": oc_res_prf.fn += 1
                        elif k == "COMP_RES": comp_res_prf.fn += 1
                        elif k == "CI_RES": ci_res_prf.fn += 1  
                        elif k == "PVAL_RES": pval_res_prf.fn += 1
                        elif k == "DES_RES": des_res_prf.fn += 1              
    return {
        "rel_micro_p": micro_prf.precision,
        "rel_micro_r": micro_prf.recall,
        "rel_micro_f": micro_prf.fscore,
        "a1_res_p": a1_res_prf.precision,
        "a1_res_r":a1_res_prf.recall,
        "a1_res_f": a1_res_prf.fscore,
        "a2_res_p": a2_res_prf.precision,
        "a2_res_r": a2_res_prf.recall,
        "a2_res_f": a2_res_prf.fscore,
        "oc_res_p": oc_res_prf.precision,
        "oc_res_r": oc_res_prf.recall,
        "oc_res_f": oc_res_prf.fscore,
        "comp_res_p": comp_res_prf.precision,
        "comp_res_r": comp_res_prf.recall,
        "comp_res_f": comp_res_prf.fscore,
        "ci_res_p": ci_res_prf.precision,
        "ci_res_r": ci_res_prf.recall,
        "ci_res_f": ci_res_prf.fscore,
        "pval_res_p": pval_res_prf.precision,
        "pval_res_r": pval_res_prf.recall,
        "pval_res_f": pval_res_prf.fscore,
        "des_res_p": des_res_prf.precision,
        "des_res_r": des_res_prf.recall,
        "des_res_f": des_res_prf.fscore,
    }
