from __future__ import annotations

from src.chunking import compute_similarity
from src.embeddings import MockEmbedder


PAIRS = [
    {
        "question": 1,
        "sentence_a": "Hàng hóa giao đến bị hỏng và tôi muốn đổi trả.",
        "sentence_b": "Sản phẩm nhận được không đúng tình trạng, tôi cần hoàn trả.",
        "prediction": "cao",
    },
    {
        "question": 2,
        "sentence_a": "Tôi muốn đổi trả do sản phẩm bị lỗi.",
        "sentence_b": "Mặt trời mọc ở phía đông mỗi sáng.",
        "prediction": "thấp",
    },
    {
        "question": 3,
        "sentence_a": "Tôi muốn biết thời hạn bảo hành của sản phẩm.",
        "sentence_b": "Sản phẩm này được bảo hành trong bao lâu?",
        "prediction": "cao",
    },
    {
        "question": 4,
        "sentence_a": "Nhà Bán cần lưu video đóng gói hàng hóa.",
        "sentence_b": "Khách hàng muốn theo dõi tình trạng giao hàng.",
        "prediction": "thấp",
    },
    {
        "question": 5,
        "sentence_a": "Tôi cần đổi sản phẩm vì nhận sai màu.",
        "sentence_b": "Tôi muốn hoàn tiền vì sản phẩm bị giao nhầm màu.",
        "prediction": "cao",
    },
]


def classify_score(score: float, all_scores: list[float]) -> str:
    """Classify a score relative to the five scores in this experiment."""
    ordered_scores = sorted(all_scores)
    midpoint = (ordered_scores[1] + ordered_scores[3]) / 2
    return "cao" if score >= midpoint else "thấp"


def main() -> None:
    embedder = MockEmbedder()

    print("Similarity predictions")
    print("Predictions are recorded before calculating the actual scores.\n")
    for pair in PAIRS:
        print(f"Pair {pair['question']}: predicted={pair['prediction']}")

    scores: list[float] = []
    for pair in PAIRS:
        vector_a = embedder(pair["sentence_a"])
        vector_b = embedder(pair["sentence_b"])
        scores.append(compute_similarity(vector_a, vector_b))

    print("\nResults")
    correct_count = 0
    for pair, score in zip(PAIRS, scores):
        actual = classify_score(score, scores)
        correct = pair["prediction"] == actual
        correct_count += int(correct)
        print(f"\nPair {pair['question']}")
        print(f"  A: {pair['sentence_a']}")
        print(f"  B: {pair['sentence_b']}")
        print(f"  Prediction: {pair['prediction']}")
        print(f"  Actual cosine similarity: {score:.4f}")
        print(f"  Relative result: {actual}")
        print(f"  Correct: {'yes' if correct else 'no'}")

    lowest_index = min(range(len(scores)), key=scores.__getitem__) + 1
    highest_index = max(range(len(scores)), key=scores.__getitem__) + 1
    print("\nReflection")
    print(
        f"The highest score was pair {highest_index} ({max(scores):.4f}) and "
        f"the lowest score was pair {lowest_index} ({min(scores):.4f})."
    )
    print(
        "The MockEmbedder is deterministic but generates vectors from an MD5 hash, "
        "so its cosine scores do not reliably represent Vietnamese sentence meaning. "
        f"The predictions matched the relative scores for {correct_count}/{len(PAIRS)} pairs."
    )


if __name__ == "__main__":
    main()
