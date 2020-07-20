import re
import html
import string
import bow
from math import log
from stemming.porter2 import stem

def tokenisation(content, stop_word_list) -> list:
    # html unescape. For example: &quot; -> " $amp; -> &  
    content = html.unescape(content)

    words = [ content.replace("Description:", "").replace("Documents", "").replace("documents", "").replace("Relevant", "").replace("relevant", "") ]
    
    # remove digitals and punctuation
    words = [ i.translate(str.maketrans("", "", string.digits)).translate(str.maketrans(string.punctuation, " "*len(string.punctuation))) for i in words ]

    # split " " 
    words = [ i.lower() for sub in words for i in sub.split() ]

    # len
    words = [ stem(i) for i in words if len(i) > 1 and i not in stop_word_list ]

    return words

def get_topic_document():
    file_name = "TopicStatements101-150.txt"
    TopicCollection = bow.TopicCollection()
    with open(file_name, "r") as f:
        content = f.read()
        content = content.replace("\n", "")
        topic_ids = re.findall(r"<num>.*?(\d*)\s?<title>",content)
        titles = re.findall(r"<title>(.*?)<desc>",content)
        # descs = re.findall(r"<desc>(.*?)<narr>",content)

        stop_word_list = []
        with open("common-english-words.txt", "r") as file:
            content = file.read()
            stop_word_list += content.split(",")

        for index, topic_id in enumerate(topic_ids):
            topic = bow.TopicDocument(topic_id)
            TopicCollection.add_topic_doc(topic_id, topic)

            title_token = tokenisation(titles[index], stop_word_list)

            for token in title_token:
                topic.add_term(token)
                
    return TopicCollection

def get_baseline_topic():
    file_name = "TopicStatements101-150.txt"
    TopicCollection = bow.TopicCollection()
    with open(file_name, "r") as f:
        content = f.read()
        content = content.replace("\n", "")
        topic_ids = re.findall(r"<num>.*?(\d*)\s?<title>",content)
        titles = re.findall(r"<title>(.*?)<desc>",content)

        stop_word_list = []
        with open("common-english-words.txt", "r") as file:
            content = file.read()
            stop_word_list += content.split(",")

        for index, topic_id in enumerate(topic_ids):
            topic = bow.TopicDocument(topic_id)
            TopicCollection.add_topic_doc(topic_id, topic)

            title_token = tokenisation(titles[index], stop_word_list)

            for token in title_token:
                topic.add_term(token)
                
    return TopicCollection

if __name__ == "__main__":
    file_name = "TopicStatements101-150.txt"
    with open(file_name, "r") as f:
        content = f.read()
        content = content.replace("\n", "")
        print(content)
        topic_ids = re.findall(r"<num>.*?(\d*)\s?<title>",content)
        titles = re.findall(r"<title>(.*?)<desc>",content)
        descs = re.findall(r"<desc>(.*?)<narr>",content)
        narr = re.findall(r"Narrative:(.*?)\.",content)

    stop_word_list = []
    with open("common-english-words.txt", "r") as file:
        content = file.read()
        stop_word_list += content.split(",")

    TopicCollection = bow.TopicCollection()

    for index, topic_id in enumerate(topic_ids):
        topic = bow.TopicDocument(topic_id)
        TopicCollection.add_topic_doc(topic_id, topic)

        title_token = tokenisation(titles[index], stop_word_list)
        desc_token = tokenisation(descs[index], stop_word_list)

        for token in title_token:
            topic.add_term(token)
        for token in desc_token:
            topic.add_term(token)

    docs = TopicCollection.get_topic_docs()
    with open("TopicResult.txt", "w") as f:
        for topic_id, doc in docs.items():
            # print(doc.terms)
            print(f"Topid: {topic_id}", file=f)
            for k, v in doc.terms.items():
                print(f"{k} : {v}", file=f)
            print("",file=f)

# with open("TopicResult.txt", "w") as f:
#     for i, v in enumerate(top_ids):
#         top_id = v
#         title = titles[i]
#         desc = descs[i]
#         print(title)
#         print(f'ID:{top_id} Title:{title} Description:{desc}', file=f)
