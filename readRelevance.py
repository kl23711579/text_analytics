
import bow
import os
import re

# read binary feedback
def read_feedback_dir(dir_name):
    os.chdir(dir_name)
    filenames = os.listdir()
    filenames = sorted(filenames)
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")

    DatasetFBColl = bow.DatasetFeedBackColl()

    for filename in filenames:
        with open(filename, "r") as f:
            contents = f.readlines()
            dataset_id = contents[0].split()[0][-3:]
            FeedBackCollection = bow.FeedBackCollection(dataset_id)
            for content in contents:
                content = content.split()
                if int(content[2]) == 1:
                    FeedBackCollection.add_relevance_doc(content[1])
                elif int(content[2]) == 0:
                    FeedBackCollection.add_irrelevance_doc(content[1])

            DatasetFBColl.add_feedback_coll(dataset_id, FeedBackCollection)

    os.chdir("..")
    return DatasetFBColl

# read float number feedback
def read_feedback_dir2(dir_name):
    os.chdir(dir_name)
    filenames = os.listdir()
    filenames = sorted(filenames)
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")

    fb = {}
    for filename in filenames:
        with open(filename, "r") as f:
            contents = f.readlines()
            result_ = re.search(r"\d{1,2}", filename)
            num_ = result_.group()
            dataset_id = str(int(num_) + 100)

            fb[dataset_id] = {}
            for content in contents:
                content = content.split()
                fb[dataset_id][content[0]] = float(content[1])

    os.chdir("..")
    return fb

if __name__ == "__main__":
    os.chdir("topicassignment101-150")
    filenames = os.listdir()
    filenames = sorted(filenames)
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")

    DatasetFBColl = bow.DatasetFeedBackColl()

    for filename in filenames:
        with open(filename, "r") as f:
            contents = f.readlines()
            dataset_id = contents[0].split()[0][-3:]
            FeedBackCollection = bow.FeedBackCollection(dataset_id)
            for content in contents:
                content = content.split()
                if int(content[2]) == 1:
                    FeedBackCollection.add_relevance_doc(int(content[1]))
                elif int(content[2]) == 0:
                    FeedBackCollection.add_irrelevance_doc(int(content[1]))

            DatasetFBColl.add_feedback_coll(dataset_id, FeedBackCollection)

    with open("../test.txt", "w") as f:
        for dataset_id in DatasetFBColl.get_dataset_ids():
            fb = DatasetFBColl.get_feedback_coll(dataset_id)
            re = fb.get_relevance_docs()
            irre = fb.get_irrelevance_docs()
            print(f"Dataset: {dataset_id}", file=f)
            print(f"relevance = {re}", file=f)
            print(f"irrelvance = {irre}", file=f)
            print("", file=f)
