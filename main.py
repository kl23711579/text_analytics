import re
import html
import string
from math import log, ceil
from stemming.porter2 import stem
from statistics import mean, stdev
from scipy import stats

def calc_stdev(relevance):
    std = stdev(relevance.values())
    if not std:
        return relevance
    r_ = [value for value in relevance.values()]
    mean_value = mean(r_)
    standard_value = { doc_id: (value-mean_value)/std for doc_id, value in relevance.items() }
    return standard_value

# get documents/terms in top?%
def get_top_relevant(d, percentage) -> list:
    if len(d) == 0:
        return []

    result = []
    needed_len = int(ceil(percentage*len(d)))
    for i, k in enumerate(d):
        if i >= needed_len:
            break
        else:
            result.append(k)

    return result

# adopt regular expression to get itemid
def get_item_id(content) -> int:
    itemid_text = re.search(r"itemid=\"(\d+)\"", content)
    return itemid_text.group(1)

# tokenise content, return list
def tokenisation(content, stop_word_list) -> list:
    # html unescape. For example: &quot; -> " $amp; -> &  
    content = html.unescape(content)

    # extract content in <p>...</p> and put in words
    text = re.finditer(r"<p>(.*?)</p>", content)
    words = [ i.group(1) for i in text ]
    
    # remove digitals and punctuation
    words = [ i.translate(str.maketrans("", "", string.digits)).translate(str.maketrans(string.punctuation, " "*len(string.punctuation))) for i in words ]

    # split " " 
    words = [ i.lower() for sub in words for i in sub.split() ]

    # len
    words = [ stem(i) for i in words if len(i) > 1 and i not in stop_word_list ]

    return words

# used in calculate idf
# return {term: freq}}
def calc_nk(bowCollection) -> dict:
    # store tf base on dataset_id
    result = {}
    BowDocuments = BowCollection.get_docs()
    for BowDocument in BowDocuments.values():
        terms = BowDocument.get_terms()
        for term in terms:
            if term in result:
                result[term] += 1
            else:
                result[term] = 1
        
    # sort 
    result = { k: v for k, v in sorted(result.items(), key=lambda i: i[1], reverse=True) }
    
    return result

# the result is {term: idf}
def calc_idf_normal(BowCollection, term_nk) -> dict:
    result = {}
    BowDocuments = BowCollection.get_docs()

    # calculate total length
    doc_lens = [ d.get_doc_len() for d in BowDocuments.values() ]
    total_length = sum(doc_lens)

    result = { term: log(total_length/nk) for term, nk in term_nk.items()}
    
    return result

# return {doc_id: {term: tf_idf}}}
def calc_tf_idf_normal(BowCollection, idf):
    result = {}
    BowDocuments = BowCollection.get_docs()
    for BowDocument in BowDocuments.values():
        terms = BowDocument.get_terms()
        a_tf_idf = { term: (1+log(v))*idf[term] for term, v in terms.items() }
        value = [ v**2 for v in a_tf_idf.values() ]
        Denominator = sum(value) ** 0.5
        tf_idf = { term: v/Denominator for term, v in a_tf_idf.items()}
        result[BowDocument.get_doc_id()] = { k: v for k, v in sorted(tf_idf.items(), key=lambda i: i[1], reverse=True)}

    return result

# original -> {docid {term: freq}}
# return {term: {docid: freq}}
def invert_list(original):
    result = {}
    for doc_id, terms in original.items():
        for term, value in terms.items():
            if term in result:
                result[term][doc_id] = value
            else:
                result[term] = {}
                result[term][doc_id] = value

    return result

# return {doc_id: relevance}
def calc_relevance_tf_idf_normal(tf_idf, TopicDocument):
    query_terms = TopicDocument.get_terms()

    # invert 
    invert_tf_idf = invert_list(tf_idf)

    # {doc_id, relevance}
    R = {k: 0 for k in tf_idf.keys()}

    # select invert list base on query
    query_term_invert = {}
    for query_term in query_terms.keys():
        if query_term in invert_tf_idf:
            query_term_invert[query_term] = invert_tf_idf[query_term]

    for query_term, doc_tf_idf in query_term_invert.items():
        for doc, tf_idf in doc_tf_idf.items():
            R[doc] = R[doc] + tf_idf

    R = { k: v for k, v in sorted(R.items(), key=lambda i: i[1], reverse=True) }
    return R

def calc_bm25_normal(BowCollection, TopicDocument, nk, relevant=[]):
    query_terms = TopicDocument.get_terms()
    BowDocuments = BowCollection.get_docs()
    origin = { BowDocument.get_doc_id(): BowDocument.get_terms() for BowDocument in BowDocuments.values() }
    invert_lists = invert_list(origin)

    # calculate total length
    doc_lens = { BowDocument.get_doc_id(): BowDocument.get_doc_len() for BowDocument in BowDocuments.values() }
    total_length = sum(doc_lens.values())
    avgdl = total_length / len(doc_lens.values())

    N = len(BowDocuments)
    R = len(relevant)

    result = {k: 0 for k in BowDocuments.keys()}

    query_term_invert = {}
    for query_term in query_terms.keys():
        if query_term in invert_lists:
            query_term_invert[query_term] = invert_lists[query_term]

    b = 0.75
    k1 = 0
    k2 = 100
    doc_K = { docid: k1 * ( (1-b) + (b * (dl/avgdl) ) ) for docid, dl in doc_lens.items() }

    # calculate term frequence in relevant documents
    r = { k: 0 for k in query_term_invert.keys()}
    for doc_id in relevant:
        terms = BowCollection.get_doc(doc_id).get_terms()
        for term in r.keys():
            if term in terms:
                r[term] += 1

    for query_term, invert in query_term_invert.items():
        for doc, freq in invert.items():
            n_ = nk[query_term]
            r_ = r[query_term]
            K = doc_K[doc]
            result[doc] = result[doc] + ( log(((r_+0.5)/(R-r_+0.5))/((n_-r_+0.5)/(N-n_-R+r_+0.5)), 2) * (((k1+1)*freq)/(K+freq)) * (((k2+1)*query_terms[query_term])/(k2+query_terms[query_term])) )

    # sort
    result = { k: v for k, v in sorted(result.items(), key=lambda i: i[1], reverse=True) }
    return result

# if document inculdes query_term, add 1
def calc_binary(BowCollection, TopicDocument):
    query_terms = TopicDocument.get_terms() # term: freq
    BowDocuments = BowCollection.get_docs()
    origin = { BowDocument.get_doc_id(): BowDocument.get_terms() for BowDocument in BowDocuments.values() }
    invert_lists = invert_list(origin)

    result = {BowDocument.get_doc_id():0 for BowDocument in BowDocuments.values()}
    # if no query term, all documents are irrelvant
    if len(query_terms) == 0:
        return result

    # invert_list => {term: {docid: freq}}
    for term in query_terms.keys():
        if term in invert_lists.keys():
            for doc_id in invert_lists[term].keys():
                result[doc_id] = result[doc_id] + 1


    result = { k: v for k, v in sorted(result.items(), key=lambda i: i[1], reverse=True) }

    return result

def calc_baseline(BowCollection, BaselineTopic, nk):
    query_terms = BaselineTopic.get_terms()
    BowDocuments = BowCollection.get_docs()
    
    doc_lens = { BowDocument.get_doc_id(): BowDocument.get_doc_len() for BowDocument in BowDocuments.values() }
    total_length = sum(doc_lens.values())
    avgdl = total_length / len(doc_lens.values())

    N = len(BowDocuments)

    bm25s = {}
    for doc_id, doc in BowDocuments.items():
        k = 1.2 * ((1 - 0.75) + 0.75 * (doc_lens[doc_id] / float(avgdl)))
        bm25_ = 0.0
        for qt in query_terms.keys():
            n = 0
            if qt in nk.keys():
                n = nk[qt]
                f = doc.get_term_counts(qt)
                qf = query_terms[qt]
                bm = log(1.0 / ((n + 0.5) / (N - n + 0.5)), 2) * (((1.2 + 1) * f) / (k + f)) * ( ((100 + 1) * qf) / float(100 + qf))
                bm25_ += bm
        bm25s[doc.get_doc_id()] = bm25_

    bm25s = { k: v for k, v in sorted(bm25s.items(), key=lambda i: i[1], reverse=True) }
    return bm25s

# Employ tfidf algorithm to calculate term weighting 
def calc_tfidf_term_weighting(BowCollection, relevance_feedback, nk):
    # terms in relevance document
    T = []
    F_re = 0
    F_irre = 0
    beta = 16
    gamma = 4

    relevance = relevance_feedback.get_relevance_docs()
    if len(relevance) == 0:
        return [], {}

    # step 1
    for doc_id in relevance:
        BowDocument = BowCollection.get_doc(doc_id)
        terms = BowDocument.get_terms()
        for term in terms.keys():
            if term not in T:
                T.append(term)
    
    # step 2
    tf_re = { k: 0 for k in T }
    tf_irre = { k: 0 for k in T }
    df = { k: 0 for k in T }

    # step 3
    for doc_id in relevance:
        BowDocument = BowCollection.get_doc(doc_id)
        terms = BowDocument.get_terms()
        for term in terms.keys():
            try:
                tf_re[term] += 1
                F_re += 1
            except:
                continue

    # step 4
    irrelevance = relevance_feedback.get_irrelevance_docs()
    for doc_id in irrelevance:
        BowDocument = BowCollection.get_doc(doc_id)
        terms = BowDocument.get_terms()
        for term in terms.keys():
            try:
                tf_irre[term] += 1
                F_irre += 1
            except:
                continue

    # step 5
    for term in T:
        df[term] = nk[term]

    # step 6
    N = BowCollection.get_total_length()
    R = len(relevance)
    idf = { term: log(N/(value+0.5), 2) for term, value in df.items() }
    tf_re = { term: (value/(F_re+0.5)) for term, value in tf_re.items() }
    tf_irre = { term: (value/(F_irre+0.5)) for term, value in tf_irre.items() }
    w = { term: ( ((beta/(R+0.5))*tf_re[term]) - ((gamma/(N-R+0.5))*tf_irre[term]) ) * value for term, value in idf.items() }

    # mean
    x = [ v for v in w.values() ]
    mean_value = mean(x)
    w = { k: v for k,v in w.items() if v >= mean_value }
    T = w.keys()
    
    # sort
    w = { k: v for k, v in sorted(w.items(), key=lambda i: i[1], reverse=True) }
    # print(w)

    return T, w

# Employ bm25 algorithm to calculate term weighting 
def calc_bm25_term_weighting(BowCollection, relevance_feedback, nk_):
    T = []
    relevance = relevance_feedback.get_relevance_docs()
    if len(relevance) == 0:
        return [], {}

    # step 1
    for doc_id in relevance:
        BowDocument = BowCollection.get_doc(doc_id)
        terms = BowDocument.get_terms()
        for term in terms.keys():
            if term not in T:
                T.append(term)

    # step 2
    nk = { k: 0 for k in T }
    rk = { k: 0 for k in T }

    # step 3
    for term in T:
        nk[term] = nk_[term]

    # step 4
    for doc_id in relevance:
        BowDocument = BowCollection.get_doc(doc_id)
        terms = BowDocument.get_terms()
        for term in terms.keys():
            try:
                rk[term] += 1
            except:
                continue

    # step 5
    N = BowCollection.get_total_length()
    R = len(relevance)
    w = { term: ((value+0.5)/(R-value+0.5))/((nk[term]-value+0.5)/( (N-nk[term])-(R-value)+0.5 ))for term, value in rk.items() }

    # sort
    w = { k: v for k, v in sorted(w.items(), key=lambda i: i[1], reverse=True) }

    # selection top50 terms
    top = 50
    if len(w) > top:
        w_ = {}
        for i, k in enumerate(w):
            if i < top:
                w_[k] = w[k]
            else:
                break
        w = w_
        T = w.keys()
    else:
        T = w.keys()

    return T, w

# test algorithm for tfidf
def calc_ranking(BowCollection, T, w):
    BowDocuments = BowCollection.get_docs()
    ranking_score = { doc_id: 0 for doc_id in BowDocuments.keys()}
    if len(T) == 0:
        return ranking_score

    origin = { BowDocument.get_doc_id(): BowDocument.get_terms() for BowDocument in BowDocuments.values() }
    invert_lists = invert_list(origin)

    for term in T:
        for doc_id in invert_lists[term].keys():
            ranking_score[doc_id] = ranking_score[doc_id] + w[term]

    ranking_score = { k: v for k, v in sorted(ranking_score.items(), key=lambda i: i[1], reverse=True) }

    return ranking_score

def measure_result(file, score, real_feedback, dataset_id=0):
    relevant = real_feedback.get_relevance_docs()
    correct = []
    total_precision = 0
    top10 = score.keys()
    retrieve = list(top10)[0:10]

    for i in range(0, len(retrieve)):
        if retrieve[i] in relevant:
            correct.append(retrieve[i])
            total_precision += (len(correct) / (i+1))
    
    recall = len(correct) / len(relevant)
    precision = len(correct) / 10

    try:
        f_m = (2 * recall * precision) / (recall + precision)
    except:
        f_m = 0

    try:
        average_precision = total_precision/len(correct)
    except:
        average_precision = 0

    with open(file, "w") as f:
        print(f"{dataset_id}", file=f)
        print(f"relevant documents = {str(len(relevant))}", file=f)
        print(f"retrieve documents = {str(len(retrieve))}", file=f)
        print(f"retrieve documents are relevant = {str(len(correct))}", file=f)
        print(f"Recall = {str(recall)}", file=f)
        print(f"Precision = {str(precision)}", file=f)
        print(f"F-Measure = {str(f_m)}", file=f)
        print(f"Average Precision = {str(average_precision)}", file=f)
        print("", file=f)

    return {'Topic': dataset_id, 'Recall': recall, 'Precision': precision, 'F1': f_m, 'Average_precision':average_precision}

# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
def write_file(file_name, relevance_list, dataset_id, threshold=0) -> None:
    with open(file_name, "w") as f:
        for k, v in relevance_list.items():
            r = 0
            if v >= threshold:
                r = 1
            print(f"R{dataset_id} {k} {r}", file=f)

# write origin number, do not need to translate to binary result
def write_file2(file_name, relevance_list) -> None:
    with open(file_name, "w") as f:
        for k, v in relevance_list.items():
            print(f"{k} {v}", file=f)

def write_file3(file_name, relevance_documents, all_documents, dataset_id) -> None:
    with open(file_name, "w") as f:
        for k, v in all_documents.items():
            if k in relevance_documents:
                print(f"R{dataset_id} {k} 1", file=f)
            else:
                print(f"R{dataset_id} {k} 0", file=f)

def make_folder():
    dirs = ['baseline', 'baseline_measure', 'tfidf_measure', 'bm25_measure', 'tfidf_normal', 'bm25_normal', 'tfidf_feedback', 'bm25_feedback']
    for i in dirs:
        try:
            os.mkdir(i)
        except:
            pass

# read file
if __name__ == '__main__':
    import os
    import sys
    import bow
    import readtopic
    import readRelevance
    import csv

    myPath = 'Data'
    # myPath = 'TestData'

    dirs = os.listdir(myPath)
    dirs = sorted(dirs)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    DatasetBowCollection = bow.DatasetBowColl()

    # genarate the query for different algorithm
    TopicCollection_tfidf = readtopic.get_topic_document()
    TopicCollection_bm25 = readtopic.get_topic_document()
    BaselineTopicColl = readtopic.get_baseline_topic()

    # create necessary folder
    make_folder()

    stop_word_list = []
    with open("common-english-words.txt", "r") as file:
        content = file.read()
        stop_word_list += content.split(",")

    # tokenisation XML data
    for _dir in dirs:
        filenames = os.listdir(myPath + "/" + _dir)
        filenames = sorted(filenames)
        if ".DS_Store" in filenames:
            filenames.remove(".DS_Store")
        filenames = [  myPath + "/" + _dir + "/" + i for i in filenames if ".xml" in i] # get filenames
        BowCollection = bow.BowCollection()
        dataset_id = _dir[-3:]

        for filename in filenames:
            with open(filename, "r") as file:
                content = file.read()
                doc_id = get_item_id(content)

                b = bow.BowDocument(doc_id)
                BowCollection.add_doc(b.get_doc_id(), b)

                tokens = tokenisation(content, stop_word_list)
                for token in tokens:
                    b.add_term(token)
                
                b.set_doc_len(len(b.terms))
                b.set_dataset_id(dataset_id)
        
        DatasetBowCollection.add_coll(dataset_id, BowCollection)

    # discovery train set, two algorithm
    # 1. tfidf
    # 2. Boolean
    term_nk = {}
    relevance_baseline = {}
    file_tf_idf_normal = "tfidf_normal/Training"
    file_bm25_normal = "bm25_normal/Training"
    file_baseline = "baseline/BaselineResult"
    for dataset_id in DatasetBowCollection.get_dataset_ids():
        BowCollection = DatasetBowCollection.get_colls(dataset_id)
        term_nk[dataset_id] = calc_nk(BowCollection)

        # tfidf
        idf = calc_idf_normal(BowCollection, term_nk[dataset_id])
        tfidf = calc_tf_idf_normal(BowCollection, idf)
        relevance_tf_idf_normal = calc_relevance_tf_idf_normal(tfidf, TopicCollection_tfidf.get_topic_doc(dataset_id))
        standard_value = calc_stdev(relevance_tf_idf_normal)
        write_file(file_tf_idf_normal + dataset_id + ".txt", standard_value, dataset_id, 1)

        # apply binary method to caiculate relevant document
        relevance_bm_25 = calc_binary(BowCollection, TopicCollection_bm25.get_topic_doc(dataset_id))
        relevant = get_top_relevant(relevance_bm_25, 0.2)
        write_file3(file_bm25_normal + dataset_id + ".txt", relevant, relevance_bm_25, dataset_id)

        # baseline
        num = str(int(dataset_id) - 100)
        relevance_baseline[dataset_id] = calc_baseline(BowCollection, BaselineTopicColl.get_topic_doc(dataset_id), term_nk[dataset_id])
        write_file2(file_baseline + num + ".dat", relevance_baseline[dataset_id]) 

    # refine the relevant result which provide by boolean retrieve
    for i in range(0,3):
        # read feedback file store in DatasetFeedBackColl
        dataset_bm25_fb_normal = readRelevance.read_feedback_dir("bm25_normal")

        for dataset_id in DatasetBowCollection.get_dataset_ids():
            BowCollection = DatasetBowCollection.get_colls(dataset_id)
            relevant = dataset_bm25_fb_normal.get_feedback_coll(dataset_id).get_relevance_docs()
            relevance_bm_25 = calc_bm25_normal(BowCollection, TopicCollection_bm25.get_topic_doc(dataset_id), term_nk[dataset_id], relevant)
            relevant = get_top_relevant(relevance_bm_25, 0.1)

            write_file3(file_bm25_normal + dataset_id + ".txt", relevant, relevance_bm_25, dataset_id)
    
    # information flitering system using tfidf and bm25,
    # read feedback file and store in DatasetFeedBackColl
    dataset_tfidf_fb_normal = readRelevance.read_feedback_dir("tfidf_normal")
    dataset_bm25_fb_normal = readRelevance.read_feedback_dir("bm25_normal")
    tfidf_ranking_score = {}
    bm25_ranking_score = {}
    for dataset_id in DatasetBowCollection.get_dataset_ids():
            BowCollection = DatasetBowCollection.get_colls(dataset_id)

            # tfidf
            tfidf_fb_normal = dataset_tfidf_fb_normal.get_feedback_coll(dataset_id)
            tfidf_T, tfidf_term_weighting = calc_tfidf_term_weighting(BowCollection, tfidf_fb_normal, term_nk[dataset_id])
            tfidf_ranking_score[dataset_id] = calc_ranking(BowCollection, tfidf_T, tfidf_term_weighting)

            # bm25
            bm25_fb_normal = dataset_bm25_fb_normal.get_feedback_coll(dataset_id)
            bm25_T, bm25_term_weighting = calc_bm25_term_weighting(BowCollection, bm25_fb_normal, term_nk[dataset_id])
            bm25_ranking_score[dataset_id] = calc_ranking(BowCollection, bm25_T, bm25_term_weighting)
    
    # write feedback of IF
    file_tf_idf = "tfidf_feedback/result"
    file_bm25 = "bm25_feedback/result"
    for dataset_id in DatasetBowCollection.get_dataset_ids():
        num = str(int(dataset_id) - 100)
        write_file2(file_tf_idf + num + ".dat", tfidf_ranking_score[dataset_id])
        write_file2(file_bm25 + num + ".dat", bm25_ranking_score[dataset_id])

    # read feedback of IF and baseline
    tfidf_result = readRelevance.read_feedback_dir2("tfidf_feedback")
    bm25_result = readRelevance.read_feedback_dir2("bm25_feedback")
    baseline_result = readRelevance.read_feedback_dir2("baseline")

    # measurement
    dataset_real_feedback = readRelevance.read_feedback_dir("topicassignment101-150")
    measure_tf_idf = "tfidf_measure/Training"
    measure_bm25 = "bm25_measure/Training"
    measure_baseline = "baseline_measure/Training"
    tfidf_measure = []
    bm25_measure = []
    baseline_measure = []
    for dataset_id in DatasetBowCollection.get_dataset_ids():
        real_feedback = dataset_real_feedback.get_feedback_coll(dataset_id)
        tfidf_measure.append( measure_result(measure_tf_idf + dataset_id + ".txt", tfidf_result[dataset_id], real_feedback, dataset_id) )
        bm25_measure.append(measure_result(measure_bm25 + dataset_id + ".txt", bm25_result[dataset_id], real_feedback, dataset_id) )
        baseline_measure.append(measure_result(measure_baseline + dataset_id + '.txt', baseline_result[dataset_id], real_feedback, dataset_id) )
    
    # store measure result in csv
    header = ['Topic', 'Recall', 'Precision', 'F1', 'Average_precision']
    with open("EvaluationResult_tfidf.dat", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(tfidf_measure)
    
    with open("EvaluationResult_BM25.dat", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(bm25_measure)
    
    with open("EvaluationResult_baseline.dat", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(baseline_measure)
    
    # t-test
    measure_items = ['Recall', 'Precision', 'F1', 'Average_precision']
    for item in measure_items:
        tfidf_m_result = [ value[item] for value in tfidf_measure ]
        bm25_m_result = [ value[item] for value in bm25_measure ]
        baseline_m_result = [ value[item] for value in baseline_measure ]
        tfidf_t_stat, tfidf_p_value = stats.ttest_ind(tfidf_m_result, baseline_m_result, equal_var=False)
        bm25_t_stat, bm25_p_value = stats.ttest_ind(bm25_m_result,baseline_m_result, equal_var=False)
        print(f"Mean value")
        print(f"tfidf = {mean(tfidf_m_result)}")
        print(f"bm25 = {mean(bm25_m_result)}")
        print(f"baseline = {mean(baseline_m_result)}")
        print(f"t-test for {item}")
        print(f"TF_IDF")
        print(f"t = {tfidf_t_stat}  p = {tfidf_p_value}")
        print(f"BM25")
        print(f"t = {bm25_t_stat}  p = {bm25_p_value}")
        print("")
