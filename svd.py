"""
Latent Semantic Indexing (LSI) with SVD
"""

import numpy as np

# ------------------------------
# Step 1: Preprocess docs → term-document matrix
# ------------------------------
def build_term_doc_matrix(docs):
    tokenized = [doc.lower().split() for doc in docs]
    vocab = sorted(set(word for doc in tokenized for word in doc))

    A = np.zeros((len(vocab), len(docs)))
    for j, doc in enumerate(tokenized):
        for word in doc:
            i = vocab.index(word)
            A[i, j] += 1
    return A, vocab


# ------------------------------
# Step 2: Apply SVD and reduce dimensions
# ------------------------------
def lsi(A, k=2):
    U, S, VT = np.linalg.svd(A, full_matrices=False)
    Uk = U[:, :k]
    Sk = np.diag(S[:k])
    Vk = VT[:k, :]
    return U, S, VT, Uk, Sk, Vk


# ------------------------------
# Step 3: Project query into reduced space
# ------------------------------
def project_query(query, vocab, Uk, Sk):
    q_vec = np.zeros((len(vocab), 1))
    for word in query.lower().split():
        if word in vocab:
            i = vocab.index(word)
            q_vec[i, 0] += 1
    q_reduced = np.dot(np.linalg.inv(Sk), np.dot(Uk.T, q_vec))
    return q_vec, q_reduced


# ------------------------------
# Step 4: Search
# ------------------------------
def search_lsi(query, docs, k=2, top_k=3):
    A, vocab = build_term_doc_matrix(docs)
    print("=== Term-Document Matrix A ===")
    print(A)
    print("Vocab:", vocab, "\n")

    U, S, VT, Uk, Sk, Vk = lsi(A, k)
    print("=== Full U Matrix ===")
    print(U)
    print("\n=== Singular Values Σ ===")
    print(S)
    print("\n=== Full V^T Matrix ===")
    print(VT)

    print("\n=== Reduced Uk ===")
    print(Uk)
    print("\n=== Reduced Sk ===")
    print(Sk)
    print("\n=== Reduced Vk ===")
    print(Vk, "\n")

    # Project query
    q_vec, q_reduced = project_query(query, vocab, Uk, Sk)
    print("=== Original Query Vector ===")
    print(q_vec.flatten())
    print("\n=== Reduced Query Vector ===")
    print(q_reduced.flatten(), "\n")

    # Document vectors in reduced space
    doc_reduced = np.dot(Sk, Vk)
    print("=== Document Vectors in Reduced Space ===")
    print(doc_reduced, "\n")

    # Cosine similarity
    sims = []
    for i in range(doc_reduced.shape[1]):
        d = doc_reduced[:, i]
        sim = np.dot(q_reduced.flatten(), d) / (
            np.linalg.norm(q_reduced) * np.linalg.norm(d)
        )
        sims.append((i, sim))
        print(f"Cosine similarity(Query, Doc{i}) = {sim:.3f}")

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]



docs = [
    "information retrieval is the process of obtaining information",
    "machine learning is about learning from data",
    "search engines use information retrieval models",
    "data science uses machine learning techniques"
]

query = "information system"
results = search_lsi(query, docs, k=2, top_k=3)

print("\n=== Final Results ===")
print("Query:", query)
for doc_id, score in results:
    print(f"Doc {doc_id}: {docs[doc_id]} (score={score:.3f})")
