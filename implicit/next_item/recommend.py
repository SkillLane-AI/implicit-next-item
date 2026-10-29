import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .._nearest_neighbours import NearestNeighboursScorer, all_pairs_knn


def bm25_weight(user_item_matrix, tf_sat=100, len_norm=0):
    """Apply BM25 weights to a user-item sparse matrix."""
    matrix = coo_matrix(user_item_matrix, copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    number_of_users = float(matrix.shape[0])
    item_frequency = np.bincount(matrix.col, minlength=matrix.shape[1])
    item_idf = np.log(number_of_users) - np.log1p(item_frequency)
    user_lengths = np.asarray(matrix.sum(axis=1)).ravel()
    average_user_length = user_lengths.mean()
    length_factor = (1.0 - len_norm) + len_norm * user_lengths / average_user_length

    matrix.data = (
        matrix.data
        * (tf_sat + 1.0)
        / (tf_sat * length_factor[matrix.row] + matrix.data)
        * item_idf[matrix.col]
    )
    return matrix


class ItemItemRecommender:
    """Item-item KNN recommender using a precomputed similarity matrix."""

    def __init__(self, num_neighbors=20, num_threads=0):
        self.num_neighbors = num_neighbors
        self.num_threads = num_threads
        self.similarity = None
        self.copurchase_counts = None
        self.lambda_mean = None
        self.scoring_similarity = None
        self.confidence_adjusted_similarity = None
        self.scorer = None

    def fit(self, weighted_matrix, show_progress=True):
        self.similarity = all_pairs_knn(
            weighted_matrix,
            self.num_neighbors,
            show_progress=show_progress,
            num_threads=self.num_threads,
        ).tocsr()
        self.scoring_similarity = self.similarity
        self.scorer = NearestNeighboursScorer(self.scoring_similarity)
        return self

    def build_confidence_adjusted_similarity(self, lambda_mean):
        """Apply co-purchase confidence weights to a copy of the similarity matrix."""
        self.lambda_mean = lambda_mean
        self.confidence_adjusted_similarity = self.similarity.copy()
        if self.copurchase_counts is not None and lambda_mean is not None:
            counts = self.copurchase_counts.data
            self.confidence_adjusted_similarity.data *= counts / (counts + lambda_mean)

    def save(self, path):
        """Save the fitted model to a NumPy .npz file."""
        values = {
            "num_neighbors": self.num_neighbors,
            "num_threads": self.num_threads,
            "tf_sat": getattr(self, "tf_sat", 1.2),
            "len_norm": getattr(self, "len_norm", 0.75),
            "shape": self.similarity.shape,
            "data": self.similarity.data,
            "indices": self.similarity.indices,
            "indptr": self.similarity.indptr,
        }
        if self.copurchase_counts is not None:
            values["copurchase_data"] = self.copurchase_counts.data
            values["lambda_mean"] = (
                self.lambda_mean if self.lambda_mean is not None else np.nan
            )
        np.savez(path, **values)

    @classmethod
    def load(cls, path):
        """Load a fitted model from a NumPy .npz file."""
        with np.load(path, allow_pickle=False) as saved:
            model = cls()
            model.similarity = csr_matrix(
                (saved["data"], saved["indices"], saved["indptr"]),
                shape=saved["shape"],
            )
            model.copurchase_counts = None
            model.lambda_mean = None
            if "copurchase_data" in saved.files:
                model.copurchase_counts = csr_matrix(
                    (
                        saved["copurchase_data"],
                        model.similarity.indices.copy(),
                        model.similarity.indptr.copy(),
                    ),
                    shape=model.similarity.shape,
                )
                saved_lambda = float(saved["lambda_mean"])
                if np.isfinite(saved_lambda):
                    model.lambda_mean = saved_lambda
            model.num_neighbors = int(saved["num_neighbors"])
            model.num_threads = int(saved["num_threads"])
            model.tf_sat = float(saved["tf_sat"])
            model.len_norm = float(saved["len_norm"])

        model.scoring_similarity = model.similarity
        model.scorer = NearestNeighboursScorer(model.similarity)
        model.build_confidence_adjusted_similarity(model.lambda_mean)
        return model

    def recommend(self, user_items, n=10, remove_seen=True):
        """Return the top n item ids and scores for one user's history row."""
        user_items = csr_matrix(user_items)
        return self.scorer.recommend(
            user_items.indptr,
            user_items.indices,
            user_items.data,
            K=n,
            remove_own_likes=remove_seen,
        )

    def similar_items(self, item_id, n=10):
        """Return the n nearest item ids and their similarity scores."""
        row = self.similarity.getrow(item_id)
        order = np.argsort(row.data)[::-1][:n]
        return row.indices[order], row.data[order]


class BM25Recommender(ItemItemRecommender):
    """Item-item KNN recommender with BM25 weighting and co-purchase counts."""

    def __init__(self, num_neighbors=20, tf_sat=1.2, len_norm=0.75, num_threads=0):
        super().__init__(num_neighbors, num_threads)
        self.tf_sat = tf_sat
        self.len_norm = len_norm

    def fit(self, counts, show_progress=True):
        weighted = bm25_weight(counts.T, self.tf_sat, self.len_norm).T.tocsr()
        super().fit(weighted, show_progress)

        course_users = csr_matrix(counts.T)
        course_users.sum_duplicates()
        course_users.eliminate_zeros()
        course_users.data[:] = 1
        copurchase_data = np.zeros(len(self.similarity.data), dtype=np.int32)

        for course_index in range(self.similarity.shape[0]):
            start, end = self.similarity.indptr[course_index : course_index + 2]
            neighbor_indices = self.similarity.indices[start:end]
            if len(neighbor_indices):
                shared_buyers = course_users[neighbor_indices].dot(
                    course_users.getrow(course_index).T
                )
                copurchase_data[start:end] = np.asarray(shared_buyers.toarray()).ravel()

        self.copurchase_counts = csr_matrix(
            (
                copurchase_data,
                self.similarity.indices.copy(),
                self.similarity.indptr.copy(),
            ),
            shape=self.similarity.shape,
        )
        return self
