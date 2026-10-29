import calendar
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .recommend import BM25Recommender

DEFAULT_PARAMS = {
    "neighbors": 20,
    "max_history": 10,
    "bm25_k1": 0.4830184698832563,
    "bm25_b": 0.5,
    "num_threads": 0,
}


def build_user_sequences(records):
    grouped = defaultdict(list)
    for record in records:
        if all(record.get(key) is not None for key in ("user_id", "course_id", "event_time")):
            grouped[record["user_id"]].append(record)

    sequences = []
    for user_records in grouped.values():
        sequence = []
        seen_courses = set()
        for record in sorted(user_records, key=lambda row: (row["event_time"], row["course_id"])):
            if record["course_id"] not in seen_courses:
                seen_courses.add(record["course_id"])
                sequence.append(record)
        if len(sequence) >= 2:
            sequences.append(sequence)
    return sequences


def _subtract_months(value, months):
    month = value.month - months
    year = value.year
    while month <= 0:
        month += 12
        year -= 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def _latest_index(sequence, start_time, end_time=None):
    for index in range(len(sequence) - 1, -1, -1):
        event_time = sequence[index]["event_time"]
        if event_time >= start_time and (end_time is None or event_time < end_time):
            return index
    return None


def _build_case(sequence, target_index):
    target = sequence[target_index]
    return {
        "user_id": target["user_id"],
        "history": [row["course_id"] for row in sequence[:target_index]],
        "target": target["course_id"],
    }


def split_records(records, validation_months=2, test_months=2):
    sequences = build_user_sequences(records)
    if not sequences:
        raise ValueError("No user has at least 2 enrollments, cannot split")

    latest_event_time = max(sequence[-1]["event_time"] for sequence in sequences)
    test_start = _subtract_months(latest_event_time, test_months)
    validation_start = _subtract_months(test_start, validation_months)
    train, validation, test = [], [], []

    for sequence in sequences:
        validation_index = _latest_index(sequence, validation_start, test_start)
        test_index = _latest_index(sequence, test_start)
        if validation_index:
            validation.append(_build_case(sequence, validation_index))
        if test_index:
            test.append(_build_case(sequence, test_index))
        train.extend(
            {"user_id": row["user_id"], "course_id": row["course_id"]}
            for row in sequence
            if row["event_time"] < validation_start
        )
    return train, validation, test


def build_user_item_matrix(records):
    course_ids = sorted({row["course_id"] for row in records})
    user_ids = sorted({row["user_id"] for row in records})
    course_index = {course_id: index for index, course_id in enumerate(course_ids)}
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    rows = np.fromiter(
        (user_index[row["user_id"]] for row in records), dtype=np.int32, count=len(records)
    )
    columns = np.fromiter(
        (course_index[row["course_id"]] for row in records), dtype=np.int32, count=len(records)
    )
    matrix = coo_matrix(
        (np.ones(len(records), dtype=np.float64), (rows, columns)),
        shape=(len(user_ids), len(course_ids)),
    ).tocsr()
    matrix.data[:] = 1.0
    return matrix, course_ids, course_index


def build_history_row(history, n_items, max_history=10):
    """Create a history vector for the latest unique item indexes."""
    latest_first = list(dict.fromkeys(reversed(history)))[:max_history]
    return csr_matrix(
        (
            np.ones(len(latest_first), dtype=np.float64),
            (np.zeros(len(latest_first), dtype=np.int32), latest_first),
        ),
        shape=(1, n_items),
    )


def evaluate(model, cases, course_index, k=10, max_history=10):
    hits = 0
    ndcg = 0.0
    cold_cases = 0

    for case in cases:
        history = [course_index[item] for item in case["history"] if item in course_index]
        target = course_index.get(case["target"])
        if not history or target is None:
            cold_cases += 1
            continue
        recommended, _ = model.recommend(
            build_history_row(history, len(course_index), max_history), n=k
        )
        matches = np.flatnonzero(recommended == target)
        if matches.size:
            hits += 1
            ndcg += 1.0 / np.log2(matches[0] + 2)

    total = len(cases)
    return {
        "cases": total,
        "cold_cases": cold_cases,
        f"hit_rate@{k}": round(hits / total, 4) if total else 0.0,
        f"ndcg@{k}": round(ndcg / total, 4) if total else 0.0,
    }


def _confidence_stats(model, top_k=10):
    total = 0
    pair_count = 0
    similarity = model.similarity
    copurchase_counts = model.copurchase_counts

    for row_index in range(similarity.shape[0]):
        start, end = similarity.indptr[row_index : row_index + 2]
        neighbors = similarity.indices[start:end]
        scores = similarity.data[start:end]
        positions = np.flatnonzero((neighbors != row_index) & (scores > 0))
        positions = positions[np.argsort(scores[positions])[::-1][:top_k]]
        counts = copurchase_counts.data[start + positions]
        counts = counts[counts > 0]
        total += int(counts.sum())
        pair_count += len(counts)

    return total / pair_count if pair_count else None


def _fit_and_evaluate(
    records,
    params=None,
    validation_months=2,
    test_months=2,
    eval_k=10,
    confidence_neighbor_top_k=10,
):
    """Fit the next-item model and return model, course ids, metrics, and lambda mean."""
    params = {**DEFAULT_PARAMS, **(params or {})}
    train_rows, validation, test = split_records(records, validation_months, test_months)
    matrix, course_ids, course_index = build_user_item_matrix(train_rows)

    model = BM25Recommender(
        num_neighbors=params["neighbors"],
        tf_sat=params["bm25_k1"],
        len_norm=params["bm25_b"],
        num_threads=params["num_threads"],
    ).fit(matrix, show_progress=False)
    lambda_mean = _confidence_stats(model, confidence_neighbor_top_k)
    model.build_confidence_adjusted_similarity(lambda_mean)
    metrics = {
        "validation": evaluate(model, validation, course_index, eval_k, params["max_history"]),
        "test": evaluate(model, test, course_index, eval_k, params["max_history"]),
    }
    return model, course_ids, metrics, lambda_mean
