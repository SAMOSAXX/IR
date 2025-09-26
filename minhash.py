#minhash

import numpy as np
import random

def get_ngrams(sequence, n):
    words = sequence.split()  # Split into words
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    return ngrams

def create_shingle_doc_matrix(sequences, n):
    all_shingles = set()
    for seq in sequences:
        shingles = get_ngrams(seq, n)
        all_shingles.update(shingles)
    all_shingles = sorted(list(all_shingles))

    n_shingles = len(all_shingles)
    n_docs = len(sequences)
    shingle_doc_matrix = np.zeros((n_shingles, n_docs), dtype=int)

    for j, doc in enumerate(sequences):
        doc_shingles = get_ngrams(doc, n)
        for i, shingle in enumerate(all_shingles):
            if shingle in doc_shingles:
                shingle_doc_matrix[i, j] = 1

    return shingle_doc_matrix, all_shingles

def get_example_hash_functions(modulus):
  h1 = lambda r: (1 * r + 1) % modulus
  h2 = lambda r: (3 * r + 1) % modulus
  return [h1, h2]

def generate_hash_functions(num_hashes, modulus):
    hash_functions = []
    for _ in range(num_hashes):
        a = random.randint(1, modulus - 1)
        b = random.randint(0, modulus - 1)
        hash_functions.append(lambda r, a=a, b=b: (a * r + b) % modulus)
    return hash_functions

def compute_signature_matrix(shingle_doc_matrix, num_hashes, modulus=None):
    n_shingles, n_docs = shingle_doc_matrix.shape
    if modulus is None:
        modulus = n_shingles
    hash_functions = get_example_hash_functions(modulus)
    # hash_functions = generate_hash_functions(num_hashes, modulus)  # RANDOM

    INF = np.inf
    sig_matrix = np.full((num_hashes, n_docs), INF)

    for r in range(n_shingles):
        # Compute hash values for this row
        hash_values = []
        for h in hash_functions:
            hash_values.append(h(r))

        for c in range(n_docs):
            if shingle_doc_matrix[r, c] == 1:
                for i in range(num_hashes):
                    if hash_values[i] < sig_matrix[i, c]:
                        sig_matrix[i, c] = hash_values[i]

    return sig_matrix

def compute_signature_matrix_permutations(shingle_doc_matrix, num_hashes):
    n_shingles, n_docs = shingle_doc_matrix.shape

    # Initialize signature matrix with infinity
    INF = np.inf
    sig_matrix = np.full((num_hashes, n_docs), INF)

    # Generate random permutations of row indices
    rows = list(range(n_shingles))
    permutations = [random.sample(rows, len(rows)) for _ in range(num_hashes)]

    # Compute MinHash signature using permutations
    for i, perm in enumerate(permutations):
        for col in range(n_docs):
            for row in perm:
                if shingle_doc_matrix[row, col] == 1:
                    sig_matrix[i, col] = row
                    break  # Take the first row index where a 1 is found

    return sig_matrix

def minhash_similarity(sig_vec1, sig_vec2):
    matches = np.sum(sig_vec1 == sig_vec2)
    return matches / len(sig_vec1)

def jaccard_similarity(shingles1, shingles2):
    set1 = set(shingles1)
    set2 = set(shingles2)
    intersection_size = len(set1 & set2)
    union_size = len(set1 | set2)
    if union_size == 0:
        return 0.0
    else:
        return intersection_size / union_size

import numpy as np

def print_pairwise_similarity_matrices(sequences, sig_matrix, n):
    n_docs = sig_matrix.shape[1]
    minhash_sim_matrix = np.zeros((n_docs, n_docs))
    jaccard_sim_matrix = np.zeros((n_docs, n_docs))

    # Compute shingles for each document (no list comprehension)
    doc_shingles = []
    for seq in sequences:
        shingles = get_ngrams(seq, n)
        doc_shingles.append(shingles)

    # Compute similarity matrices
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                minhash_sim_matrix[i, j] = 1.0
                jaccard_sim_matrix[i, j] = 1.0
            else:
                minhash_sim_matrix[i, j] = minhash_similarity(sig_matrix[:, i], sig_matrix[:, j])
                jaccard_sim_matrix[i, j] = jaccard_similarity(doc_shingles[i], doc_shingles[j])

    # Print MinHash similarity matrix
    print("MinHash Pairwise Similarity Matrix:")
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                print("*", end="\t")
            elif i < j:
                print(f"{minhash_sim_matrix[i, j]:.2f}", end="\t")
            else:
                print("-", end="\t")
        print()

    # Print Jaccard similarity matrix
    print("\nJaccard Pairwise Similarity Matrix:")
    for i in range(n_docs):
        for j in range(n_docs):
            if i == j:
                print("*", end="\t")
            elif i < j:
                print(f"{jaccard_sim_matrix[i, j]:.2f}", end="\t")
            else:
                print("-", end="\t")
        print()
    return minhash_sim_matrix, jaccard_sim_matrix

def detect_duplicates(minhash_sim_matrix, jaccard_sim_matrix, threshold=0.9):
    n_docs = minhash_sim_matrix.shape[1]

    print(f"\n[DUPLICATES - MinHash] Detecting duplicates with similarity >= {threshold}:")
    found_minhash = False
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = minhash_sim_matrix[i, j]
            if similarity >= threshold:
                print(f"Docs {i+1} and {j+1}: MinHash Similarity = {similarity:.2f}")
                found_minhash = True
    if not found_minhash:
        print("No duplicates found.")

    print(f"\n[DUPLICATES - Jaccard] Detecting duplicates with similarity >= {threshold}:")
    found_jaccard = False
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            similarity = jaccard_sim_matrix[i, j]
            if similarity >= threshold:
                print(f"Docs {i+1} and {j+1}: Jaccard Similarity = {similarity:.2f}")
                found_jaccard = True
    if not found_jaccard:
        print("No duplicates found.")


# Example usage
sequences = ["the cat sat", "the dog ran", "cat and dog", "the cat ran"]
n = 2
shingle_doc_matrix, shingle_vocab = create_shingle_doc_matrix(sequences, n)
print("Shingles:", shingle_vocab)
print("Shingle-Doc Matrix:\n", shingle_doc_matrix)

# Using hash functions
num_hashes = 2
modulus = 5
print("\n=== MinHash with Hash Functions ===")
sig_matrix_hash = compute_signature_matrix(shingle_doc_matrix, num_hashes, modulus)
print("Signature Matrix (Hash Functions):")
print(sig_matrix_hash)
minhash_sim_matrix_hash, jaccard_sim_matrix = print_pairwise_similarity_matrices(sequences, sig_matrix_hash, n)
detect_duplicates(minhash_sim_matrix_hash, jaccard_sim_matrix, threshold=0.9)

# Using permutations
num_hashes = 5
print("\n=== MinHash with Permutations ===")
sig_matrix_perm = compute_signature_matrix_permutations(shingle_doc_matrix, num_hashes)
print("Signature Matrix (Permutations):")
print(sig_matrix_perm)
minhash_sim_matrix_perm, jaccard_sim_matrix = print_pairwise_similarity_matrices(sequences, sig_matrix_perm, n)
detect_duplicates(minhash_sim_matrix_perm, jaccard_sim_matrix, threshold=0.9)



#TCSCODE
# -*- coding: utf-8 -*-
"""
Duplicate Detection using Shingles + MinHash
"""

import random

# ------------------------------
# Step 1: Shingling
# ------------------------------
def get_shingles(doc, k=3):
    words = doc.lower().split()
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i:i+k])
        shingles.add(shingle)
    return shingles

# ------------------------------
# Step 2: Jaccard Similarity
# ------------------------------
def jaccard(set1, set2):
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# ------------------------------
# Step 3: MinHash (Random Permutations)
# ------------------------------
def minhash_permutation(shingle_sets, num_perm=10):
    """MinHash signatures using random permutations (conceptual)."""
    all_shingles = list(set().union(*shingle_sets))
    N = len(all_shingles)

    # assign IDs
    shingle_to_id = {s: i for i, s in enumerate(all_shingles)}

    signatures = []
    for doc_set in shingle_sets:
        sig = []
        doc_ids = {shingle_to_id[s] for s in doc_set}
        for _ in range(num_perm):
            perm = list(range(N))
            random.shuffle(perm)

            print(perm)

            # first shingle in permutation that belongs to doc
            for idx in perm:
                if idx in doc_ids:
                    sig.append(idx)
                    break
        signatures.append(sig)
        print(sig)
    return signatures

# ------------------------------
# Step 4: MinHash (Hash Functions)
# ------------------------------
def hash_functions(num_hashes, max_val):
    funcs = []
    for _ in range(num_hashes):
        a = random.randint(1, max_val-1)
        b = random.randint(0, max_val-1)
        funcs.append(lambda x, a=a, b=b: (a*x + b) % max_val)
    return funcs

def minhash_hashing(shingle_sets, num_hashes=10):
    all_shingles = list(set().union(*shingle_sets))
    N = len(all_shingles)
    shingle_to_id = {s: i for i, s in enumerate(all_shingles)}

    funcs = hash_functions(num_hashes, N*2)

    signatures = []
    for doc_set in shingle_sets:
        sig = []
        ids = [shingle_to_id[s] for s in doc_set]
        for h in funcs:
            sig.append(min(h(i) for i in ids))
        signatures.append(sig)
    return signatures

# ------------------------------
# Step 5: Compare signatures
# ------------------------------
def signature_similarity(sig1, sig2):
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)


docs = [
    "information retrieval is the process of obtaining information",
    "search engines use information retrieval models",
    "machine learning is about learning from data",
    "information retrieval system for search"
]

# Shingles
k = 3
shingle_sets = [get_shingles(doc, k) for doc in docs]
print("Shingles (k=3):")
for i, s in enumerate(shingle_sets):
    print(f"D{i}: {s}")
print()

# Exact Jaccard
print("Exact Jaccard Similarities:")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = jaccard(shingle_sets[i], shingle_sets[j])
        print(f"D{i} vs D{j}: {sim:.3f}")
print()

# MinHash with random permutations
perm_sigs = minhash_permutation(shingle_sets, num_perm=3)
print("Approximate Similarity (Random Permutations):")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = signature_similarity(perm_sigs[i], perm_sigs[j])
        print(f"D{i} vs D{j}: {sim:.3f}")
print()

# MinHash with hash functions
hash_sigs = minhash_hashing(shingle_sets, num_hashes=20)
print("Approximate Similarity (Hash Functions):")
for i in range(len(docs)):
    for j in range(i+1, len(docs)):
        sim = signature_similarity(hash_sigs[i], hash_sigs[j])
        print(f"D{i} vs D{j}: {sim:.3f}")
