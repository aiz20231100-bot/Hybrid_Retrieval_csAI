import json
import re
import sys
from collections import Counter

from config import INDEXES
from preprocessing import clean_text

VOCAB_PATH = INDEXES / "vocab.json"
LETTERS = "abcdefghijklmnopqrstuvwxyz"
PHRASE_RE = re.compile(r'"([^"]+)"')
MIN_KNOWN = 3        # كلمة تكررت 3 مرات أو أكثر تُعتبر صحيحة
MIN_CANDIDATE = 10   # لا نصحح إلا إلى كلمة شائعة (10 مرات أو أكثر)

ABBREVIATIONS = {
    "llm": "large language model", "llms": "large language models",
    "rl": "reinforcement learning",
    "marl": "multi agent reinforcement learning",
    "mas": "multi agent systems",
    "kr": "knowledge representation", "kg": "knowledge graph",
    "kgs": "knowledge graphs",
    "pddl": "planning domain definition language",
    "htn": "hierarchical task network",
    "mdp": "markov decision process",
    "pomdp": "partially observable markov decision process",
    "nlp": "natural language processing", "ml": "machine learning",
    "gnn": "graph neural network", "cnn": "convolutional neural network",
    "ai": "artificial intelligence", "agi": "artificial general intelligence",
    "bdi": "belief desire intention", "asp": "answer set programming",
    "csp": "constraint satisfaction problem",
    "sat": "boolean satisfiability",
    "owl": "web ontology language", "rdf": "resource description framework",
    "vla": "vision language action", "vlm": "vision language model",
    "slam": "simultaneous localization and mapping",
    "ros": "robot operating system", "mcts": "monte carlo tree search",
    "rrt": "rapidly exploring random tree", "cot": "chain of thought",
    "rag": "retrieval augmented generation", "dqn": "deep q network",
    "ppo": "proximal policy optimization", "ltl": "linear temporal logic",
    "hri": "human robot interaction", "mapf": "multi agent path finding",
    "tamp": "task and motion planning",
}
# الاتجاه المعاكس: العبارة الكاملة -> الاختصار
REVERSE = {}
for _ab, _full in ABBREVIATIONS.items():
    REVERSE.setdefault(_full, _ab)


def normalize(text):
    """lowercase + أي رمز غير حرف/رقم يصبح مسافة (multi-agent == multi agent)."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [l + r[1:] for l, r in splits if r]
    transposes = [l + r[1] + r[0] + r[2:] for l, r in splits if len(r) > 1]
    replaces = [l + c + r[1:] for l, r in splits if r for c in LETTERS]
    inserts = [l + c + r for l, r in splits for c in LETTERS]
    return set(deletes + transposes + replaces + inserts)


class QueryProcessor:
    def __init__(self, docs=None):
        if VOCAB_PATH.exists():
            with open(VOCAB_PATH, encoding="utf-8") as f:
                self.vocab = json.load(f)
        else:
            if docs is None:
                raise ValueError("vocab.json not found: pass docs to build it.")
            self.vocab = self._build_vocab(docs)

    @staticmethod
    def _build_vocab(docs):
        counter = Counter()
        for d in docs:
            text = f"{d['title']} {d['abstract']} {clean_text(d['text'])}"
            counter.update(re.findall(r"[a-z]{2,}", text.lower()))
        vocab = {w: c for w, c in counter.items() if c >= MIN_KNOWN}
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f)
        return vocab

    def correct_word(self, word):
        if (len(word) < 4 or word in ABBREVIATIONS
                or self.vocab.get(word, 0) >= MIN_KNOWN):
            return word
        e1 = edits1(word)
        cands = [w for w in e1 if self.vocab.get(w, 0) >= MIN_CANDIDATE]
        if not cands and len(word) >= 6:
            e2 = {x for e in e1 for x in edits1(e)}
            cands = [w for w in e2 if self.vocab.get(w, 0) >= MIN_CANDIDATE]
        return max(cands, key=lambda w: self.vocab[w]) if cands else word

    def process(self, query):
        q = query.replace("\u201c", '"').replace("\u201d", '"')
        phrases = [p for p in (normalize(x) for x in PHRASE_RE.findall(q)) if p]
        rest = PHRASE_RE.sub(" ", q)

        corrections = []

        def fix(m):
            w = m.group(0).lower()          # Case Insensitivity
            c = self.correct_word(w)
            if c != w:
                corrections.append((w, c))
            return c

        corrected = re.sub(r"\s+", " ", re.sub(r"[A-Za-z]+", fix, rest)).strip()

        norm = " " + normalize(corrected) + " "
        expansions = []
        for tok in norm.split():
            if tok in ABBREVIATIONS and (tok, ABBREVIATIONS[tok]) not in expansions:
                expansions.append((tok, ABBREVIATIONS[tok]))
        for full, ab in REVERSE.items():
            if f" {full} " in norm and (full, ab) not in expansions:
                expansions.append((full, ab))

        parts = [corrected] + [to for _, to in expansions] + phrases
        search_text = " ".join(p for p in parts if p).strip()
        return {
            "original": query, "corrected": corrected,
            "corrections": corrections, "expansions": expansions,
            "phrases": phrases, "search_text": search_text,
        }


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    from corpus import load_corpus

    qp = QueryProcessor(load_corpus())
    print(f"Vocabulary size: {len(qp.vocab)}")
    tests = [
        "heuristc serch for plannning",
        "reinforcment lerning multi-agnet",
        "LLM planning",
        "MARL cooperation",
        "large language models for robots",
        "HEURISTIC SEARCH",
        '"multi-agent reinforcement learning" cooperation',
    ]
    for t in tests:
        r = qp.process(t)
        print(f"\nQuery:       {t}")
        print(f"Corrected:   {r['corrected']}  {r['corrections']}")
        print(f"Expansions:  {r['expansions']}")
        print(f"Phrases:     {r['phrases']}")
        print(f"Search text: {r['search_text']}")