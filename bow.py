class BowDocument:
    def __init__(self, doc_id):
        self.terms = {}
        self.doc_id = doc_id
        self.doc_len = 0
        self.dataset_id = 0

    def add_term(self, term):
        if term not in self.terms.keys():
            self.terms[term] = 1
        else:
            self.terms[term] += 1
    
    def get_terms(self):
        return self.terms

    def get_term_counts(self, term):
        try:
            return self.terms[term]
        except:
            return 0

    def set_doc_len(self, len):
        self.len = len

    def get_doc_len(self) -> int:
        return self.len

    def get_doc_id(self) -> str:
        return self.doc_id
    
    def set_dataset_id(self, dataset_id):
        self.dataset_id = dataset_id

    def get_dataset_id(self) ->str:
        return self.dataset_id

class BowCollection:
    def __init__(self):
        self.docs = {}

    def add_doc(self, doc_id, doc):
        self.docs[doc_id] = doc

    def get_doc(self, doc_id) -> BowDocument:
        return self.docs[doc_id]

    def get_docs(self):
        return self.docs

    def get_total_length(self):
        return len(self.docs)

    def get_dataset_docs(self, dataset_id):
        return {k: v for k,v in self.docs.items() if v.get_dataset_id() == dataset_id}

class DatasetBowColl:
    def __init__(self):
        self.colls = {}

    def add_coll(self, dataset_id, coll):
        self.colls[dataset_id] = coll

    def get_colls(self, dataset_id) -> BowCollection:
        return self.colls[dataset_id]

    def get_dataset_ids(self):
        return self.colls.keys()

class TopicDocument:
    def __init__(self, topic_id):
        self.terms = {}
        self.topic_id = topic_id

    def add_term(self, term):
        if term not in self.terms.keys():
            self.terms[term] = 1
        else:
            self.terms[term] += 1

    def get_terms(self):
        return self.terms
    
    def get_topic_id(self):
        return self.topic_id

class TopicCollection:
    def __init__(self):
        self.topic_docs = {}

    def add_topic_doc(self, topic_id, topic_doc):
        self.topic_docs[topic_id] = topic_doc

    def get_topic_doc(self, topic_id):
        return self.topic_docs[topic_id]

    def get_topic_docs(self):
        return self.topic_docs

class FeedBackCollection:
    def __init__(self, dataset_id):
        self.relevance = []
        self.irrelevance = []
        self.dataset_id = dataset_id

    def add_relevance_doc(self, document_id):
        self.relevance.append(document_id)

    def add_irrelevance_doc(self, document_id):
        self.irrelevance.append(document_id)

    def get_relevance_docs(self) -> list:
        return self.relevance

    def get_irrelevance_docs(self) -> list:
        return self.irrelevance

    def get_total_length(self):
        return len(self.relevance) + len(self.irrelevance)

class DatasetFeedBackColl:
    def __init__(self):
        self.colls = {}

    def add_feedback_coll(self, dataset_id, coll):
        self.colls[dataset_id] = coll

    def get_feedback_coll(self, dataset_id):
        return self.colls[dataset_id]

    def get_dataset_ids(self):
        return self.colls.keys()
