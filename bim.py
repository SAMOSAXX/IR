#BIM
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
from collections import defaultdict

ps = PorterStemmer()

# 1. Preprocessing
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stemmed_tokens = []
    for t in tokens:
        stemmed_tokens.append(ps.stem(t))
    return stemmed_tokens

def preprocess_docs(doc_dict):
    processed = {}
    for doc_id, content in doc_dict.items():
        processed[doc_id] = preprocess(content)
    return processed

# 2. BIM Functions
def compute_df(processed_docs):
    df_scores = defaultdict(int)
    for tokens in processed_docs.values():
        for token in set(tokens):
            df_scores[token] += 1
    return df_scores

def compute_phase1_weights(df, N):
    weights = {}
    for term, dk in df.items():
        weights[term] = math.log((N - dk + 0.5) / (dk + 0.5))
    return weights

def compute_phase2_weights(df, processed_docs, relevant_docs, N):
    Nr = len(relevant_docs)
    rk_dict = defaultdict(int)

    for doc_id in relevant_docs:
        tokens = processed_docs[doc_id]
        for token in set(tokens):  # use set directly
            rk_dict[token] += 1

    weights = {}
    for term, dk in df.items():
        rk = rk_dict.get(term, 0)
        numerator = (rk + 0.5) / (Nr - rk + 0.5)
        denominator = (dk - rk + 0.5) / (N - Nr - dk + rk + 0.5)
        weights[term] = math.log(numerator / denominator)
    return weights

def score_docs(processed_docs, query_tokens, weights):
    scores = {}
    for doc_id, tokens in processed_docs.items():
        score = 0.0
        for t in query_tokens:
            if t in tokens and t in weights:
                score += weights[t]
        scores[doc_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# 3. Visualization
def plot_rsv_scores(ranked_docs, phase, query_id):
    docs = []
    scores = []
    for doc_id, score in ranked_docs:
        docs.append(doc_id)
        scores.append(score)

    plt.figure(figsize=(8,5))
    plt.bar(docs, scores, color='lightcoral')
    plt.axhline(y=0, color='b', linestyle='--', label='Threshold (0)')
    plt.xlabel('Documents')
    plt.ylabel('RSV Score')
    plt.title('BIM RSV Scores (' + phase + ') for ' + query_id)
    plt.legend()
    plt.show()

def plot_incidence_matrix(processed_docs):
    terms = []
    for tokens in processed_docs.values():
        for t in tokens:
            if t not in terms:
                terms.append(t)
    terms.sort()

    N = len(processed_docs)
    incidence_matrix = np.zeros((len(terms), N), dtype=int)

    j = 0
    for doc_id, tokens in processed_docs.items():
        i = 0
        for term in terms:
            if term in tokens:
                incidence_matrix[i, j] = 1
            i += 1
        j += 1

    plt.figure(figsize=(8,6))
    sns.heatmap(incidence_matrix,
                xticklabels=list(processed_docs.keys()),
                yticklabels=terms,
                cmap='Blues', annot=True)
    plt.title('Incidence Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

# --- Example data ---
database_docs = {
    "d1": "The Boolean retrieval model is a classic model in information retrieval. It answers queries based on Boolean logic. A document is a set of words.",
    "d2": "The vector space model represents documents as vectors. The weight of a term is calculated using term frequency. This model can rank documents.",
    "d3": "Probabilistic information retrieval models are based on probability theory. The Okapi BM25 model is a popular ranking function used by a search engine."
}

query_docs = {
    "d4": "In the vector space model, term frequency and inverse document frequency are used to calculate term weight. This allows the model to rank search results.",
    "d5": "The vector space model can rank documents. It represents documents as vectors, and the weight of a term is computed using term frequency."
}

# --- Processing ---
processed_db_docs = preprocess_docs(database_docs)
processed_query_docs = preprocess_docs(query_docs)

N = len(processed_db_docs)
df = compute_df(processed_db_docs)
weights_phase1 = compute_phase1_weights(df, N)

# Example relevance info (for feedback)
relevance = {
    'd4': ['d2'],
    'd5': ['d2']
}

# Plot incidence matrix once for DB
plot_incidence_matrix(processed_db_docs)

for doc_id, query_tokens in processed_query_docs.items():
    print("\nQuery from '" + doc_id + "'...")

    # --- Phase I retrieval (no feedback) ---
    ranked_docs_phase1 = score_docs(processed_db_docs, query_tokens, weights_phase1)
    print("Phase I (no relevance feedback) Ranking:")
    for db_id, score in ranked_docs_phase1:
        print("  " + db_id + ": " + str(round(score,4)))
    plot_rsv_scores(ranked_docs_phase1, "Phase I", doc_id)

    # --- Phase II retrieval (with relevance feedback) ---
    relevant_docs = relevance.get(doc_id, [])
    if len(relevant_docs) > 0:
        weights_phase2 = compute_phase2_weights(df, processed_db_docs, relevant_docs, N)
        ranked_docs_phase2 = score_docs(processed_db_docs, query_tokens, weights_phase2)
        print("Phase II (with relevance feedback) Ranking:")
        for db_id, score in ranked_docs_phase2:
            print("  " + db_id + ": " + str(round(score,4)))
        plot_rsv_scores(ranked_docs_phase2, "Phase II", doc_id)
        
        
#BM25 MODEL BIM
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter

ps = PorterStemmer()

# 1. Preprocessing
def preprocess(text):
    text = text.lower()
    tokens = re.findall(r'\b\w+\b', text)
    stemmed_tokens = []
    for t in tokens:
        stemmed_tokens.append(ps.stem(t))
    return stemmed_tokens

def preprocess_docs(doc_dict):
    processed = {}
    for doc_id, content in doc_dict.items():
        processed[doc_id] = preprocess(content)
    return processed

# 2. BM25 Functions
def compute_doc_lengths(processed_docs):
    lengths = {}
    total_len = 0
    for doc_id, tokens in processed_docs.items():
        lengths[doc_id] = len(tokens)
        total_len += len(tokens)
    if len(processed_docs) > 0:
        avgdl = total_len / len(processed_docs)
    else:
        avgdl = 0
    return lengths, avgdl

def compute_df(processed_docs):
    df = defaultdict(int)
    for tokens in processed_docs.values():
        unique_tokens = set(tokens)   # simplified
        for token in unique_tokens:
            df[token] += 1
    return df

def compute_idf(df, N):
    idf = {}
    for term, dk in df.items():
        idf[term] = math.log((N - dk + 0.5) / (dk + 0.5) + 1)
    return idf

def score_docs_bm25(processed_docs, query_tokens, idf, doc_lengths, avgdl, k1=1.5, b=0.75):
    scores = {}
    for doc_id, tokens in processed_docs.items():
        tf_scores = Counter(tokens)   # simplified

        score = 0.0
        for t in query_tokens:
            if t in idf:
                tf = tf_scores.get(t, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_lengths[doc_id] / avgdl))
                if denominator != 0:
                    score += idf[t] * (numerator / denominator)
        scores[doc_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# 3. Visualization
def plot_rsv_scores(ranked_docs, query_id):
    docs = []
    scores = []
    for pair in ranked_docs:
        docs.append(pair[0])
        scores.append(pair[1])

    plt.figure(figsize=(8,5))
    plt.bar(docs, scores, color='lightcoral')
    plt.axhline(y=0, color='b', linestyle='--')
    plt.xlabel('Documents')
    plt.ylabel('BM25 Score')
    plt.title('BM25 Scores for ' + query_id)
    plt.show()

def plot_incidence_matrix(processed_docs):
    terms = []
    for tokens in processed_docs.values():
        for t in tokens:
            if t not in terms:
                terms.append(t)
    terms.sort()

    incidence_matrix = np.zeros((len(terms), len(processed_docs)), dtype=int)

    j = 0
    for doc_id, tokens in processed_docs.items():
        i = 0
        for term in terms:
            if term in tokens:
                incidence_matrix[i, j] = 1
            i += 1
        j += 1

    plt.figure(figsize=(8,6))
    sns.heatmap(incidence_matrix,
                xticklabels=list(processed_docs.keys()),
                yticklabels=terms,
                cmap='Blues', annot=True)
    plt.title('Incidence Matrix Heatmap')
    plt.xlabel('Documents')
    plt.ylabel('Terms')
    plt.show()

# --- Example data ---
database_docs = {
    "d1": "The Boolean retrieval model is a classic model in information retrieval. It answers queries based on Boolean logic. A document is a set of words.",
    "d2": "The vector space model represents documents as vectors. The weight of a term is calculated using term frequency. This model can rank documents.",
    "d3": "Probabilistic information retrieval models are based on probability theory. The Okapi BM25 model is a popular ranking function used by a search engine."
}

query_docs = {
    "d4": "In the vector space model, term frequency and inverse document frequency are used to calculate term weight. This allows the model to rank search results.",
    "d5": "The vector space model can rank documents. It represents documents as vectors, and the weight of a term is computed using term frequency."
}

# --- Processing ---
processed_db_docs = preprocess_docs(database_docs)
processed_query_docs = preprocess_docs(query_docs)

N = len(processed_db_docs)
doc_lengths, avgdl = compute_doc_lengths(processed_db_docs)
df = compute_df(processed_db_docs)
idf = compute_idf(df, N)

# Plot incidence matrix once for DB
plot_incidence_matrix(processed_db_docs)

for doc_id, query_tokens in processed_query_docs.items():
    print("\nQuery from '" + doc_id + "'...")

    ranked_docs = score_docs_bm25(processed_db_docs, query_tokens, idf, doc_lengths, avgdl)

    print("BM25 Ranking:")
    for pair in ranked_docs:
        db_id = pair[0]
        score = pair[1]
        print("  " + db_id + ": " + str(round(score,4)))

    plot_rsv_scores(ranked_docs, doc_id)
