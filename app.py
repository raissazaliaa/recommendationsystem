"""
Book-Crossing Recommender System (Streamlit)

Ported from the notebook "Recommendation System Assignment W10".
Methods:  Popularity baseline  |  Item-based Collaborative Filtering (cosine, sparse)
Run:      streamlit run app.py
Data:     BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv inside the folder given
          in the sidebar (default: ./BX-Users, or the BX_DATA_DIR environment variable).
"""
import csv
import html
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# Defaults taken from the notebook
SEED = 42
POSITIVE_RATING = 7
MIN_SUPPORT = 5
K = 10
MAX_EVAL_USERS = 1000
METHOD_LABELS = {"item_cf": "Item-based CF", "popular": "Popularity"}

st.set_page_config(page_title="Book Recommender", page_icon="📚", layout="wide")


# --------------------------------------------------------------------------- #
# Data loading and cleaning (notebook: "Importing Data" + "Data Cleaning")
# --------------------------------------------------------------------------- #
class CleanData:
    """Container for cleaned tables and audit numbers."""


@st.cache_resource(show_spinner="Loading and cleaning the CSV files (first run takes a while)…")
def prepare_data(data_dir: str) -> CleanData:
    d = Path(data_dir)
    users_raw = pd.read_csv(d / "BX-Users.csv", sep=";", encoding="latin-1")
    books_raw = pd.read_csv(
        d / "BX-Books.csv", sep=";", encoding="latin-1",
        dtype={"ISBN": str}, on_bad_lines="skip", low_memory=False,
    )
    ratings_raw = pd.read_csv(
        d / "BX-Book-Ratings.csv", sep=";", encoding="latin-1", dtype={"ISBN": str}
    )

    # Count skipped / incomplete book rows for the audit
    with open(d / "BX-Books.csv", encoding="latin-1", newline="") as f:
        original_book_count = sum(1 for _ in csv.reader(f, delimiter=";")) - 1
    malformed_books = original_book_count - len(books_raw) + int(books_raw["Image-URL-L"].isna().sum())

    books = books_raw.dropna(subset=["Image-URL-L"]).drop_duplicates("ISBN").copy()
    users = users_raw.drop_duplicates("User-ID").copy()

    valid = (
        ratings_raw["Book-Rating"].between(0, 10)
        & ratings_raw["User-ID"].isin(users["User-ID"])
        & ratings_raw["ISBN"].isin(books["ISBN"])
    )
    ratings_clean = ratings_raw.loc[valid].drop_duplicates(["User-ID", "ISBN"]).copy()
    explicit = ratings_clean[ratings_clean["Book-Rating"] > 0].copy()

    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    invalid_age = users["Age"].isna() | ~users["Age"].between(5, 100)
    users.loc[invalid_age, "Age"] = np.nan

    out = CleanData()
    out.users, out.books, out.explicit = users, books, explicit
    out.rating_distribution = ratings_raw["Book-Rating"].value_counts().sort_index()
    out.audit = {
        "raw_users": len(users_raw),
        "raw_books_valid_columns": len(books),
        "raw_ratings": len(ratings_raw),
        "malformed_book_rows_skipped": malformed_books,
        "implicit_zero_rows": int((ratings_raw["Book-Rating"] == 0).sum()),
        "invalid_or_unmatched_rating_rows": int((~valid).sum()),
        "duplicate_rating_rows_removed": int(valid.sum() - len(ratings_clean)),
        "clean_explicit_ratings": len(explicit),
        "clean_explicit_users": explicit["User-ID"].nunique(),
        "clean_explicit_books": explicit["ISBN"].nunique(),
        "missing_or_implausible_age": int(invalid_age.sum()),
    }
    return out


# --------------------------------------------------------------------------- #
# Model (notebook: split, user-book matrix, popularity, item-based CF)
# --------------------------------------------------------------------------- #
class Recommender:
    def __init__(self, data: CleanData, positive_rating: int, min_support: int, seed: int):
        self.positive_rating = positive_rating
        self.min_support = min_support
        self.seed = seed
        explicit = data.explicit

        # Hold out 20% of each user's ratings if the user has >= 5 ratings
        rng = np.random.default_rng(seed)
        test_indices = []
        for _, group in explicit.groupby("User-ID", sort=True):
            if len(group) >= 5:
                n_test = max(1, int(len(group) * 0.2))
                test_indices.extend(rng.choice(group.index.to_numpy(), n_test, replace=False))
        self.train = explicit.drop(index=test_indices).copy()
        self.test = explicit.loc[test_indices].copy()

        # Candidate catalog + sparse binary user-book matrix (training only)
        self.positive_train = self.train[self.train["Book-Rating"] >= positive_rating]
        self.support = self.positive_train.groupby("ISBN").size()
        self.items = np.sort(self.support[self.support >= min_support].index.to_numpy())
        self.user_ids = np.sort(self.train["User-ID"].unique())
        self.item_index = {isbn: i for i, isbn in enumerate(self.items)}
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}

        selected = self.positive_train[self.positive_train["ISBN"].isin(self.items)]
        self.interaction_matrix = csr_matrix(
            (
                np.ones(len(selected)),
                (selected["User-ID"].map(self.user_index).to_numpy(),
                 selected["ISBN"].map(self.item_index).to_numpy()),
            ),
            shape=(len(self.user_ids), len(self.items)),
        )
        self.seen_books = self.train.groupby("User-ID")["ISBN"].agg(set).to_dict()
        self.book_metadata = data.books.set_index("ISBN")
        self.sparsity = float(1 - self.interaction_matrix.nnz / np.prod(self.interaction_matrix.shape))
        self.popularity_scores = self.support.reindex(self.items).to_numpy(dtype=float)
        self.normalized_items = normalize(self.interaction_matrix, axis=0).tocsr()

    def rank_books(self, user_id, method="item_cf", k=10):
        """Return (top item indices, score vector, used_popularity_fallback)."""
        fallback = False
        if method == "popular" or user_id not in self.user_index:
            scores = self.popularity_scores.copy()
            fallback = method != "popular"
        else:
            profile = self.interaction_matrix.getrow(self.user_index[user_id])
            scores = (profile @ self.normalized_items.T @ self.normalized_items).toarray().ravel()
            if not np.any(scores > 0):
                scores = self.popularity_scores.copy()
                fallback = True

        for isbn in self.seen_books.get(user_id, set()):
            idx = self.item_index.get(isbn)
            if idx is not None:
                scores[idx] = -np.inf

        # Ties broken by popularity, then ISBN
        order = np.lexsort((self.items, -self.popularity_scores, -scores))
        top = order[np.isfinite(scores[order])][:k]
        return top, scores, fallback

    def recommend(self, user_id, method="item_cf", k=10):
        top, scores, fallback = self.rank_books(user_id, method, k)
        result = self.book_metadata.reindex(self.items[top])[
            ["Book-Title", "Book-Author", "Image-URL-M"]
        ].reset_index()
        result.insert(0, "Rank", np.arange(1, len(result) + 1))
        result["Score"] = scores[top]
        return result, fallback

    def history(self, user_id):
        h = self.train[self.train["User-ID"] == user_id].merge(
            self.book_metadata[["Book-Title", "Book-Author"]].reset_index(), on="ISBN"
        )
        return h.sort_values("Book-Rating", ascending=False)

    # ---- evaluation (notebook: "Compare Model Performance") ----
    def relevant_test(self):
        return (self.test[self.test["Book-Rating"] >= self.positive_rating]
                .groupby("User-ID")["ISBN"].agg(set))

    def evaluation_users(self, n_users):
        eligible = np.sort(self.relevant_test().index.to_numpy())
        rng = np.random.default_rng(self.seed)
        return np.sort(rng.choice(eligible, min(n_users, len(eligible)), replace=False))

    def evaluate(self, users, k, progress=None):
        relevant = self.relevant_test()
        catalog = set(self.items)
        rows = []
        total = 2 * len(users)
        done = 0
        for method in ("popular", "item_cf"):
            for uid in users:
                truth = relevant[uid]
                top, _, _ = self.rank_books(uid, method, k)
                rec = self.items[top]
                hits = np.array([isbn in truth for isbn in rec], dtype=float)
                dcg = np.sum(hits / np.log2(np.arange(len(hits)) + 2))
                ideal = np.sum(1 / np.log2(np.arange(min(k, len(truth))) + 2))
                rows.append({
                    "User-ID": int(uid), "model": method,
                    "precision": hits.sum() / k,
                    "recall": hits.sum() / len(truth),
                    "ndcg": dcg / ideal,
                    "hit_rate": float(hits.sum() > 0),
                    "relevant_count": len(truth),
                    "reachable_relevant": len(truth & catalog),
                    "recommended": "|".join(rec),
                })
                done += 1
                if progress is not None and done % 25 == 0:
                    progress.progress(done / total)
        if progress is not None:
            progress.progress(1.0)
        detail = pd.DataFrame(rows)

        table = detail.groupby("model")[["precision", "recall", "ndcg", "hit_rate"]].mean()
        table.columns = [f"{c}@{k}" for c in ["Precision", "Recall", "NDCG", "HitRate"]]
        table[f"Coverage@{k}"] = [
            len(set("|".join(detail.loc[detail.model == m, "recommended"]).split("|"))) / len(self.items)
            for m in table.index
        ]
        reach = detail.loc[detail.model == "item_cf", "reachable_relevant"].sum() / \
            detail.loc[detail.model == "item_cf", "relevant_count"].sum()
        table.index = [METHOD_LABELS[m] for m in table.index]
        return table, float(reach)


@st.cache_resource(show_spinner="Building the user–book matrix…")
def build_model(data_dir: str, positive_rating: int, min_support: int, seed: int) -> Recommender:
    return Recommender(prepare_data(data_dir), positive_rating, min_support, seed)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def clean_text(s):
    return html.unescape(str(s)) if pd.notna(s) else ""


def show_covers(recs: pd.DataFrame, per_row: int = 5):
    for start in range(0, len(recs), per_row):
        cols = st.columns(per_row)
        for col, (_, r) in zip(cols, recs.iloc[start:start + per_row].iterrows()):
            with col:
                if isinstance(r["Image-URL-M"], str):
                    st.image(r["Image-URL-M"], width="stretch")
                st.markdown(f"**{r['Rank']}. {clean_text(r['Book-Title'])}**")
                st.caption(f"{clean_text(r['Book-Author'])} · score {r['Score']:.3f}")


def set_user(uid):
    st.session_state["user_id"] = int(uid)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.title("📚 Book Recommender")
st.caption("Book-Crossing dataset · Popularity baseline vs. item-based collaborative filtering")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("Data folder", os.environ.get("BX_DATA_DIR", "BX-Users"),
                             help="Folder containing BX-Users.csv, BX-Books.csv and BX-Book-Ratings.csv")
    method = st.radio("Method", list(METHOD_LABELS), format_func=METHOD_LABELS.get)
    top_n = st.slider("Number of recommendations", 5, 20, K)
    covers = st.checkbox("Show book covers", value=True)
    with st.expander("Advanced (rebuilds the model)"):
        positive_rating = st.slider("Positive rating threshold", 1, 10, POSITIVE_RATING)
        min_support = st.slider("Min. positive ratings per book", 1, 20, MIN_SUPPORT)
        seed = st.number_input("Random seed", value=SEED, step=1)

required = ["BX-Users.csv", "BX-Books.csv", "BX-Book-Ratings.csv"]
missing = [f for f in required if not (Path(data_dir) / f).exists()]
if missing:
    st.error(f"Missing in `{data_dir}`: {', '.join(missing)}. "
             "Set the correct folder in the sidebar (or the BX_DATA_DIR environment variable).")
    st.stop()

data = prepare_data(data_dir)
model = build_model(data_dir, int(positive_rating), int(min_support), int(seed))

tab_rec, tab_eval, tab_data = st.tabs(["Recommendations", "Evaluation", "Data & model"])

# --------------------------------------------------------------------------- #
# Tab 1 – Recommendations
# --------------------------------------------------------------------------- #
with tab_rec:
    heaviest = int(model.positive_train.groupby("User-ID").size().idxmax())
    st.session_state.setdefault("user_id", heaviest)

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    c1.number_input("User-ID (use -1 for a new user)", min_value=-1, step=1, key="user_id")
    c2.button("Most active user", on_click=set_user, args=(heaviest,), width="stretch")
    c3.button("Random user", width="stretch",
              on_click=lambda: set_user(np.random.choice(model.user_ids)))
    c4.button("New user", on_click=set_user, args=(-1,), width="stretch")

    uid = int(st.session_state["user_id"])
    recs, fallback = model.recommend(uid, method, top_n)

    if uid in model.user_index:
        n_train = len(model.seen_books.get(uid, []))
        n_pos = int(model.interaction_matrix.getrow(model.user_index[uid]).nnz)
        st.write(f"**User {uid}** · {n_train} rated books in training · {n_pos} positive "
                 f"(rating ≥ {positive_rating}) in the candidate catalog")
    else:
        st.info("Unknown user → no history, so the most popular books are returned.")
    if fallback and uid in model.user_index:
        st.warning("This user has no positive history in the candidate catalog; showing popular books.")

    st.subheader(f"Top-{top_n} · {METHOD_LABELS[method]}")
    if covers:
        show_covers(recs)
    table = recs[["Rank", "ISBN", "Book-Title", "Book-Author", "Score"]].copy()
    table["Book-Title"] = table["Book-Title"].map(clean_text)
    table["Book-Author"] = table["Book-Author"].map(clean_text)
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Score is a ranking score (sum of cosine similarities for Item-CF, positive-rating count "
               "for Popularity). It is not a predicted rating on the 1–10 scale.")
    st.download_button("Download recommendations (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"recommendations_user_{uid}.csv", mime="text/csv")

    if uid in model.user_index:
        with st.expander("Reading history (top-rated training books)"):
            h = model.history(uid)
            st.dataframe(h.head(25).assign(**{
                "Book-Title": h.head(25)["Book-Title"].map(clean_text),
                "Book-Author": h.head(25)["Book-Author"].map(clean_text),
            })[["ISBN", "Book-Title", "Book-Author", "Book-Rating"]],
                hide_index=True, width="stretch")

# --------------------------------------------------------------------------- #
# Tab 2 – Evaluation
# --------------------------------------------------------------------------- #
with tab_eval:
    st.markdown(
        "Both methods rank the whole candidate catalog for the same randomly sampled users "
        "(no negative sampling). Relevant items = positively rated books hidden in the test split."
    )
    e1, e2 = st.columns(2)
    n_users = e1.slider("Users to evaluate", 100, MAX_EVAL_USERS, MAX_EVAL_USERS, step=100)
    k_eval = e2.slider("Cut-off K", 5, 20, K)

    eval_key = (data_dir, positive_rating, min_support, int(seed), n_users, k_eval)
    if st.button("Run evaluation", type="primary"):
        bar = st.progress(0.0, text="Evaluating…")
        users = model.evaluation_users(n_users)
        st.session_state["eval"] = (eval_key, model.evaluate(users, k_eval, bar), len(users))
        bar.empty()

    saved = st.session_state.get("eval")
    if saved and saved[0] == eval_key:
        table, reach, n_eval = saved[1][0], saved[1][1], saved[2]
        st.write(f"Evaluated on **{n_eval}** users · candidate books: **{len(model.items):,}**")
        st.dataframe(table.round(5), width="stretch")
        st.caption(f"Positive test items available in the candidate catalog: {reach:.2%}")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        table.iloc[:, :4].plot.bar(ax=ax, rot=0)
        ax.set(title=f"Top-{k_eval} recommendation performance", ylabel="Score")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Press **Run evaluation** to compute the metrics for the current settings.")
    st.caption("Reference (notebook, seed 42, 1000 users, K=10, threshold 7, min support 5): "
               "NDCG@10 — Item-CF 0.0263 vs Popularity 0.0066. Single random split, "
               "no significance testing; a missing rating does not mean dislike.")

# --------------------------------------------------------------------------- #
# Tab 3 – Data & model overview
# --------------------------------------------------------------------------- #
with tab_data:
    a, b, c, d = st.columns(4)
    a.metric("Training ratings", f"{len(model.train):,}")
    b.metric("Test ratings", f"{len(model.test):,}")
    c.metric("Candidate books", f"{len(model.items):,}")
    d.metric("Matrix sparsity", f"{model.sparsity:.4%}")
    st.caption(f"User–book matrix shape: {model.interaction_matrix.shape[0]:,} users × "
               f"{model.interaction_matrix.shape[1]:,} candidate books")

    left, right = st.columns(2)
    with left:
        st.subheader("Rating distribution")
        st.caption("0 = implicit feedback (not an explicit score of zero)")
        st.bar_chart(data.rating_distribution)
    with right:
        st.subheader("Valid user age")
        fig, ax = plt.subplots(figsize=(6, 3.2))
        data.users["Age"].dropna().plot.hist(bins=30, ax=ax, color="steelblue")
        ax.set(xlabel="Age", ylabel="Count")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Long-tail distribution of books")
    counts = model.train.groupby("ISBN").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.loglog(np.arange(1, len(counts) + 1), counts.to_numpy())
    ax.set(xlabel="Book rank (log)", ylabel="Training ratings (log)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Most popular books")
    pop = data.books[["ISBN", "Book-Title", "Book-Author"]].merge(
        model.support.rename("Positive_training_ratings"), left_on="ISBN", right_index=True
    ).sort_values("Positive_training_ratings", ascending=False).head(10)
    pop["Book-Title"] = pop["Book-Title"].map(clean_text)
    st.dataframe(pop, hide_index=True, width="stretch")

    st.subheader("Cleaning audit")
    st.dataframe(pd.Series(data.audit, name="Count").to_frame(), width="stretch")

    with st.expander("How item-based CF works"):
        st.latex(r"s(i,j)=\frac{X_i^\top X_j}{\|X_i\|\,\|X_j\|},\qquad "
                 r"\text{score}(u,j)=\sum_{i\in I_u^{+}} s(i,j)")
        st.markdown(
            "Books are similar when the same users rate them positively. A user's score for a book is the "
            "sum of its similarity to books the user liked; books already rated in training are excluded. "
            "Users without usable history fall back to the popularity ranking."
        )
