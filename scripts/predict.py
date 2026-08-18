import sys
from src.inference import SemantIQ


def main() -> None:
    engine = SemantIQ()

    print("\n" + "=" * 60)
    print("SemantIQ — Semantic Deduplication & Verification CLI")
    print(f"Policy: High Precision (T* = {engine.high_precision_threshold:.4f}) | Default (T = {engine.default_threshold:.2f})")
    print("=" * 60)
    print("Type 'exit' to quit.\n")

    if len(sys.argv) == 3:
        q1 = sys.argv[1]
        q2 = sys.argv[2]
        res = engine.predict_pair(q1, q2)
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print(f"Decision:              {res['decision']} (Confidence: {res['confidence']})")
        print(f"Duplicate Score:       {res['score']:.4f}")
        print(f"Contradiction Warning: {res['contradiction_warning']}")
        print(f"MiniLM Semantic Cos:   {res['evidence']['semantic_similarity_minilm']:.4f}")
        print(f"Word TF-IDF Cos:       {res['evidence']['word_tfidf_similarity']:.4f}")
        return

    while True:
        try:
            question1 = input("Question 1: ").strip()
            if question1.lower() == "exit" or not question1:
                break

            question2 = input("Question 2: ").strip()
            if question2.lower() == "exit" or not question2:
                break

            res = engine.predict_pair(question1, question2)

            print("\n" + "-" * 50)
            print(f"Decision:              {res['decision']} ({res['confidence']} confidence)")
            print(f"Duplicate Score:       {res['score']:.4f}")
            print(f"Contradiction Warning: {res['contradiction_warning']}")
            if res['contradiction_warning']:
                print(f"  [ALERT] Potential entity, numeric, or negation mismatch detected!")
            print(f"Evidence:")
            print(f"  - MiniLM Semantic Cosine: {res['evidence']['semantic_similarity_minilm']:.4f}")
            print(f"  - Word TF-IDF Cosine:     {res['evidence']['word_tfidf_similarity']:.4f}")
            print(f"  - Char TF-IDF Cosine:     {res['evidence']['char_tfidf_similarity']:.4f}")
            print(f"  - Fuzzy Token Set Ratio:  {res['evidence']['fuzzy_token_set_ratio']:.4f}")
            print(f"Latency:               {res['latency_ms']:.2f} ms")
            print("-" * 50 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
