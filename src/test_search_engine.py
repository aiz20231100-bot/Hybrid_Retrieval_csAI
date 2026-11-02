import sys
sys.stdout.reconfigure(encoding="utf-8")

from search_engine import SearchEngine


def show(results, n=3):
    for r in results[:n]:
        print(f"{r['rank']:>2}. [{r['score_name']} {r['score']:.4f}] {r['title'][:75]}")
        print(f"      {r['subtopic']} | {r['url']}")
        print(f"      snippet ({len(r['snippet'])} chars): {r['snippet']}")


def main():
    eng = SearchEngine()

    print("\n=== TEST 1: three modes, same query ===")
    for mode in eng.MODES:
        res, _ = eng.search("heuristic search for automated planning", mode=mode)
        print(f"\n[{mode.upper()}] results: {len(res)}")
        show(res)

    print("\n=== TEST 2: case insensitivity ===")
    a, _ = eng.search("HEURISTIC SEARCH PLANNING")
    b, _ = eng.search("heuristic search planning")
    print("Same results for upper/lower case:",
          [r["arxiv_id"] for r in a] == [r["arxiv_id"] for r in b])

    print("\n=== TEST 3: spelling + abbreviation ===")
    res, info = eng.search("MARL cooperatoin and plannning with LLM")
    print("Corrections:", info["corrections"])
    print("Expansions: ", info["expansions"])
    print("Search text:", info["search_text"])
    show(res)

    print("\n=== TEST 4: phrase matching ===")
    res, info = eng.search('"multi-agent reinforcement learning" communication')
    print("Phrases:", info["phrases"], "| papers containing all phrases:",
          info.get("phrase_matches"))
    show(res, n=5)


if __name__ == "__main__":
    main()