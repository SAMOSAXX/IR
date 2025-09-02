#VSM
import re
import math
from nltk.stem import PorterStemmer
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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

# 2. Vector Space Model Functions
def compute_vsm_vectors(processed_docs):
    N = len(processed_docs)
    df_scores = defaultdict(int)
    for tokens in processed_docs.values():
        for token in set(tokens):
            df_scores[token] += 1

    vectors = defaultdict(dict)
    for doc_id, tokens in processed_docs.items():
        len_i = len(tokens)
        if len_i == 0:
            continue

        tf_scores = Counter(tokens)
        for token, tf in tf_scores.items():
            df = df_scores[token]
            weight = (tf / len_i) * math.log((N + 1) / (0.5 + df))
            vectors[doc_id][token] = weight
    return vectors, df_scores

def create_single_vector(tokens, N, df_scores):
    vector = {}
    len_i = len(tokens)
    if len_i == 0:
        return vector

    tf_scores = Counter(tokens)
    for token, tf in tf_scores.items():
        df = df_scores.get(token, 0)
        weight = (tf / len_i) * math.log((N + 1) / (0.5 + df))
        vector[token] = weight
    return vector

def cosine_similarity(vec1, vec2):
    # Compute dot product
    dot_product = 0.0
    for k in vec1:
        if k in vec2:
            dot_product += vec1[k] * vec2[k]
    # Compute sum of squares for vec1
    sum_sq_vec1 = 0.0
    for val in vec1.values():
        sum_sq_vec1 += val * val
    # Compute sum of squares for vec2
    sum_sq_vec2 = 0.0
    for val in vec2.values():
        sum_sq_vec2 += val * val
    # Compute magnitude
    magnitude = math.sqrt(sum_sq_vec1) * math.sqrt(sum_sq_vec2)
    # Retun cosine similarity
    if magnitude == 0:
        return 0.0
    else:
        return dot_product / magnitude


# --- Additional Similarity Functions ---
def jaccard_similarity(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def dice_similarity(tokens1, tokens2):
    set1, set2 = set(tokens1), set(tokens2)
    if not set1 and not set2:
        return 0.0
    return 2 * len(set1 & set2) / (len(set1) + len(set2))

def plot_incidence_matrix(processed_docs):
    # Collect all unique terms
    terms = sorted(set(t for tokens in processed_docs.values() for t in tokens))
    doc_ids = list(processed_docs.keys())

    # Initialize matrix
    incidence_matrix = np.zeros((len(terms), len(doc_ids)), dtype=int)

    # Fill matrix
    for j, doc_id in enumerate(doc_ids):
        for i, term in enumerate(terms):
            if term in processed_docs[doc_id]:
                incidence_matrix[i, j] = 1

    # Plot heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(incidence_matrix,
                xticklabels=doc_ids,
                yticklabels=terms,
                cmap="Blues", cbar=False, annot=True, fmt="d")
    plt.title("Incidence Matrix (Terms × Documents)")
    plt.xlabel("Documents")
    plt.ylabel("Terms")
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

db_vectors, df_scores = compute_vsm_vectors(processed_db_docs)
N = len(processed_db_docs)

print(f"Database built successfully from {N} documents.\n")

threshold = 0.40
results = {}

for doc_id, query_tokens in processed_query_docs.items():
    query_vector = create_single_vector(query_tokens, N, df_scores)
    similarities = {}
    print(f"Checking '{doc_id}'...")

    for db_id, db_vector in db_vectors.items():
        sim_score = cosine_similarity(query_vector, db_vector)
        jaccard = jaccard_similarity(query_tokens, processed_db_docs[db_id])
        dice = dice_similarity(query_tokens, processed_db_docs[db_id])

        print(f"  - With '{db_id}': Cosine={sim_score:.4f}, Jaccard={jaccard:.4f}, Dice={dice:.4f}")
        similarities[db_id] = (sim_score, jaccard, dice)

    # Sort by cosine similarity
    similarities_sorted = sorted(similarities.items(), key=lambda x: x[1][0], reverse=True)
    for db_id, (cos_val, _, _) in similarities_sorted:
        print(f"  - Cosine Similarity with '{db_id}': {cos_val:.4f}")

    if similarities_sorted[0][1][0] > threshold:
        print(f"  -> Verdict: DUPLICATE FOUND (most similar: {similarities_sorted[0][0]})\n")
    else:
        print(f"  -> Verdict: NOT a duplicate (all cosine similarity scores <= {threshold})\n")

    results[doc_id] = similarities

# --- Plotting ---

# Bar plots for each query
for query_id, sims in results.items():
    labels = list(sims.keys())
    cosine_vals = [sims[db][0] for db in labels]
    jaccard_vals = [sims[db][1] for db in labels]
    dice_vals = [sims[db][2] for db in labels]

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, cosine_vals, width, label="Cosine")
    plt.bar(x, jaccard_vals, width, label="Jaccard")
    plt.bar(x + width, dice_vals, width, label="Dice")

    plt.ylabel("Similarity")
    plt.title(f"Similarity of {query_id} with database documents")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

plot_incidence_matrix(processed_db_docs)