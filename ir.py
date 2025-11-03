def cosine_similarity_hardcoded(vec1, vec2):
    # Common indices where BOTH vectors have ratings (not NaN)
    common_idx = ~(np.isnan(vec1) | np.isnan(vec2))

    if np.sum(common_idx) == 0:
        return 0.0

    # Dot product on common items
    dot_product = np.sum(vec1[common_idx] * vec2[common_idx])

    # Norms on common items
    norm1 = np.sqrt(np.sum(vec1[common_idx] ** 2))
    norm2 = np.sqrt(np.sum(vec2[common_idx] ** 2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def jaccard_similarity(vec1, vec2):
    """
    Considers the SET of rated items, not the rating values.
    J(X,Y) = |X ∩ Y| / |X ∪ Y|
    """
    # Find the indices of rated items (where value is not NaN)
    set1 = set(np.where(pd.notna(vec1))[0])
    set2 = set(np.where(pd.notna(vec2))[0])

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    if union == 0:
        return 0.0

    return intersection / union


def user_based_cf(ratings_df, active_user, target_item, k=3):
    # ---- 1. ALL other users (even if they didn't rate target_item) ----
    all_other_users = []
    for user in ratings_df.index:
        if user != active_user:
            all_other_users.append(user)

    if not all_other_users:
        return 0.0, []

    # ---- 2. Compute similarity with active user (hardcoded cosine) ----
    sim_dict = {}
    au_vec = ratings_df.loc[active_user].values
    for u in all_other_users:
        u_vec = ratings_df.loc[u].values
        sim = cosine_similarity_hardcoded(au_vec, u_vec)
        sim_dict[u] = sim

    print(sim_dict)
    # ---- 3. Sort ALL users by similarity (descending) ----
    sorted_users = sorted(sim_dict, key=sim_dict.get, reverse=True)

    # ---- 4. Top-k display: first k users by similarity (even if X) ----
    top_k_display = sorted_users[:k]

    # ---- 5. Prediction: ONLY use users in top-k who rated target_item ----
    num = den = 0.0
    for u in top_k_display:
        if pd.notna(ratings_df.loc[u, target_item]):  # Has rating
            sim = sim_dict[u]
            r = ratings_df.loc[u, target_item]
            num += sim * r
            den += abs(sim)

    pred = num / den if den > 0 else 0.0
    return pred, top_k_display


def item_based_cf(ratings_df, active_user, target_item, k=3):
    # ---- 1. ALL items except target_item ----
    all_items = []
    for col in ratings_df.columns:
        if col != target_item:
            all_items.append(col)

    if not all_items:
        return 0.0, []




    # ---- 2. Compute similarity with target_item ----
    sim_dict = {}
    target_vec = ratings_df[target_item].values
    for i in all_items:
        i_vec = ratings_df[i].values
        sim = cosine_similarity_hardcoded(target_vec, i_vec)
        sim_dict[i] = sim

    print(sim_dict)

    # ---- 3. Sort ALL items by similarity ----
    sorted_items = sorted(sim_dict, key=sim_dict.get, reverse=True)

    # ---- 4. Top-k display: first k items (even if not rated by AU) ----
    top_k_display = sorted_items[:k]

    # ---- 5. Prediction: ONLY use items that active_user rated ----
    num = den = 0.0
    for i in top_k_display:
        if pd.notna(ratings_df.loc[active_user, i]):  # AU rated this item
            sim = sim_dict[i]
            r = ratings_df.loc[active_user, i]
            num += sim * r
            den += abs(sim)

    pred = num / den if den > 0 else 0.0
    return pred, top_k_display

# ==============================
# DEMO WITH YOUR DATA
# ==============================

# data = {
#     'U1': [5,  2,  np.nan, np.nan, 2, 3],
#     'U2': [2,  3,  2,      3,     5, 1],
#     'U3': [1,  2,  np.nan, np.nan, 2, 3],
#     'U4': [np.nan, 3,  4, 4, 5, np.nan],
#     'U5': [3,  1,  2,      3,     4, 2]
# }

# items = ['I1','I2','I3','I4','I5','I6']
# ratings_df = pd.DataFrame(data, index=items).T   # rows = users, cols = items

# --- CSV input ---
df = pd.read_csv('/content/ratings.txt')
print(df)
ratings_df = df.pivot(index='user', columns='item', values='rating')
print(ratings_df)

# --- Alternate formats ---
# df = pd.read_csv('ratings.txt', sep='\t')   # tab-separated
# df = pd.read_csv('ratings.txt', sep=' ')    # space-separated

# ==============================
# PREDICT U1 → I3 (k=3)
# ==============================
active_user = 'U1'
target_item = 'I3'
k = 3

pred_user, top_user = user_based_cf(ratings_df, active_user, target_item, k)
pred_item, top_item = item_based_cf(ratings_df, active_user, target_item, k)

print("\n=== USER-BASED CF ===")
print(f"Prediction: {pred_user:.3f}")
print(f"Top-{k} neighbors: {top_user}")

print("\n=== ITEM-BASED CF ===")
print(f"Prediction: {pred_item:.3f}")
print(f"Top-{k} neighbors: {top_item}")