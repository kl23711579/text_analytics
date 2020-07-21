# 1. Install dependencies

> pip3 install -r requirement.txt

# 2. Prepare Data

- Put all XML data into `Data` folder
- Pleasure ensure stem function in `stemming` folder
- Needed files
    -  folder - `topicassignment101-105`
    - `TopicStatements101-105.txt`
    - `common-english-words.txt`

# 3. Execute

> python3 main.py

# 4. Result

The baseline of this project is BM25 algorithm, the `baseline_feedback` is the result of baseline.

In this project, two algorithms are employed:
    
    1. tfidf, the result is store in `tfidf_feedback`

    2. bm25 with binary approach, the result is store in `bm25_feedback`

The statistical result si shown on terminal.

# Team member

Jen-I Wen
