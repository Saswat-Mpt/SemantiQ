import sys
from src.inference import SemantIQ


def main() -> None:

    model = SemantIQ()

    print("\n" + "=" * 50)
    print("SemantIQ — Semantic Deduplication Live CLI")
    print("=" * 50)
    print("Type 'exit' to quit.\n")

    # If arguments are passed via CLI directly, run single test
    if len(sys.argv) == 3:
        q1 = sys.argv[1]
        q2 = sys.argv[2]
        res = model.predict_pair(q1, q2)
        print(f"Q1: {q1}")
        print(f"Q2: {q2}")
        print(f"Decision:    {res['decision']}")
        print(f"Probability: {res['duplicate_probability']:.4f} (Threshold = {res['threshold']:.4f})")
        print(f"MiniLM Cos:  {res['semantic_similarity']:.4f}")
        return

    # Interactive loop
    while True:
        try:
            question1 = input("Question 1: ").strip()
            if question1.lower() == "exit" or not question1:
                break

            question2 = input("Question 2: ").strip()
            if question2.lower() == "exit" or not question2:
                break

            result = model.predict_pair(question1, question2)

            print("\n" + "-" * 40)
            print(f"Decision:             {result['decision']}")
            print(f"Duplicate Probability: {result['duplicate_probability']:.4f}")
            print(f"Operating Threshold:   {result['threshold']:.4f}")
            print(f"MiniLM Cosine:         {result['semantic_similarity']:.4f}")
            print(f"Word TF-IDF Cosine:    {result['features']['word_tfidf_cosine']:.4f}")
            print(f"Fuzzy Token Set Ratio: {result['features']['token_set_ratio']:.4f}")
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            break
        except Exception as error:
            print(f"\nError: {error}\n")


if __name__ == "__main__":
    main()
