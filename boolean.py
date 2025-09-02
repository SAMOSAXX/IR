#BOOLEAN
import re
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

# 2. Index Construction
def build_inverted_index(processed_docs):
    inverted_index = defaultdict(set)
    for doc_id, tokens in processed_docs.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return inverted_index

# 3. Boolean Query Processing
def process_query(query, index, all_doc_ids):
    or_parts = query.lower().split(' or ')
    result_sets_for_or = []
    for part in or_parts:
        and_parts = part.split(' and ')
        sets_for_and = []
        for term_part in and_parts:
            term_part = term_part.strip()
            if term_part.startswith('not '):
                term = term_part[4:]
                stem = ps.stem(term)
                term_docs = index.get(stem, set())
                sets_for_and.append(all_doc_ids - term_docs)
            else:
                stem = ps.stem(term_part)
                sets_for_and.append(index.get(stem, set()))

        if len(sets_for_and) > 0:
            intersected_set = sets_for_and[0].copy()
            for s in sets_for_and[1:]:
                intersected_set.intersection_update(s)
        else:
            intersected_set = set()

        result_sets_for_or.append(intersected_set)

    final_result = set()
    for s in result_sets_for_or:
        final_result.update(s)
    return final_result


# --- Example data (instead of file paths) ---
documents = {
    "d1": "The Boolean retrieval model is a classic model in information retrieval. It answers queries based on Boolean logic.",
    "d2": "The vector space model represents documents as vectors. The weight of a term is calculated using term frequency.",
    "d3": "Probabilistic information retrieval models are based on probability theory. The Okapi BM25 model is a popular ranking function.",
    "d4": "In the vector space model, term frequency and inverse document frequency are used to calculate term weight.",
    "d5": "The vector space model can rank documents. It represents documents as vectors, and the weight of a term is computed using term frequency."
}

# --- Processing ---
processed_docs = preprocess_docs(documents)
all_doc_ids = set(processed_docs.keys())
inverted_index = build_inverted_index(processed_docs)

print("Inverted index built successfully.\n")

# --- Run example queries ---
queries = ["retrieval AND model", "boolean OR vector", "model AND NOT classic"]
for q in queries:
    result_docs = process_query(q, inverted_index, all_doc_ids)
    print("Query:", q, "→ Result:", sorted(list(result_docs)))
    
    
#BOOLEAN 2
import re
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

# 2. Index Construction
def build_inverted_index(processed_docs):
    inverted_index = defaultdict(set)
    for doc_id, tokens in processed_docs.items():
        for token in tokens:
            inverted_index[token].add(doc_id)
    return inverted_index

def translate_query(query, index, all_doc_ids):
    tokens = query.replace("(", " ( ").replace(")", " ) ").split()
    translated = []
    for t in tokens:
        upper_t = t.upper()
        if upper_t == "AND":
            translated.append("&")
        elif upper_t == "OR":
            translated.append("|")
        elif upper_t == "NOT":
            translated.append("all_doc_ids -")
        elif t in ("(", ")"):
            translated.append(t)
        else:
            stem = ps.stem(t.lower())
            translated.append(f"index.get('{stem}', set())")
    return " ".join(translated)


def process_query_advanced(query, index, all_doc_ids):
    q = translate_query(query, index, all_doc_ids)
    return eval(q, {"index": index, "all_doc_ids": all_doc_ids})


# --- Example data (instead of file paths) ---
documents = {
    "d1": "The Boolean retrieval model is a classic model in information retrieval. It answers queries based on Boolean logic.",
    "d2": "The vector space model represents documents as vectors. The weight of a term is calculated using term frequency.",
    "d3": "Probabilistic information retrieval models are based on probability theory. The Okapi BM25 model is a popular ranking function.",
    "d4": "In the vector space model, term frequency and inverse document frequency are used to calculate term weight.",
    "d5": "The vector space model can rank documents. It represents documents as vectors, and the weight of a term is computed using term frequency."
}

# --- Processing ---
processed_docs = preprocess_docs(documents)
all_doc_ids = set(processed_docs.keys())
inverted_index = build_inverted_index(processed_docs)

print("Inverted index built successfully.\n")

# --- Run example queries ---
queries = [
    # simple
    "retrieval AND model",
    "boolean OR vector",
    "model AND NOT classic",

    # with parentheses
    "(retrieval AND model) OR (boolean AND NOT classic)",
    "(retrieval OR probabilistic) or (model OR networks)",
    "retrieval AND (model OR bm25)",
    "(retrieval AND model) AND (NOT networks)",
    "(probabilistic OR bm25) AND NOT (boolean OR classic)",

    # nested parentheses
    "((retrieval AND model) OR (probabilistic AND bm25)) AND NOT (networks OR classic)"
]

for q in queries:
    result_docs = process_query_advanced(q, inverted_index, all_doc_ids)
    print("Query:", q, "→ Result:", sorted(list(result_docs)))
