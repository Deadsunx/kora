"""Find articles that explicitly reference a different acte uniforme.

The `cross_act` category was cut from eight questions to four during gold-set
validation because most of its questions were not genuinely cross-act: they
paired two topically related articles that happened to sit in different acts,
which a retriever can satisfy from either one alone. Four questions each worth
0.25 is a count, not a measurement, and the row did not move across seven
systems.

A genuine cross-act question needs a *textual bridge*: an article in act A that
names act B, so that following the reference is the only way to answer. Those
bridges are findable mechanically, which is what this does.

    python scripts/mine_cross_refs.py                 # summary
    python scripts/mine_cross_refs.py --pair AUS-2010 AUPSRVE-2023
    python scripts/mine_cross_refs.py --article-level  # only exact pointers

The article-level pointers are the most valuable: the source article names the
exact target article, so the gold pair is given by the text rather than by
judgement.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from kora import paths

# Distinctive fragments of each act's official title, as the texts cite them.
# Deliberately narrow: "societes commerciales" matches AUSCGIE and not AUSCOOP,
# whose title is "societes cooperatives".
ACT_PATTERNS: dict[str, list[str]] = {
    "AUDCG-2010": [r"droit commercial g[ée]n[ée]ral"],
    "AUS-2010": [r"organisation des s[ûu]ret[ée]s", r"\bdes s[ûu]ret[ée]s\b"],
    "AUSCGIE-2014": [
        r"soci[ée]t[ée]s commerciales et du groupement",
        r"droit des soci[ée]t[ée]s commerciales",
    ],
    "AUSCOOP-2010": [r"soci[ée]t[ée]s coop[ée]ratives"],
    "AUPSRVE-2023": [r"proc[ée]dures simplifi[ée]es de recouvrement", r"voies d'ex[ée]cution"],
    "AUPCAP-2015": [r"proc[ée]dures collectives d'apurement"],
    "AUA-2017": [r"droit de l'arbitrage", r"\b[àa] l'arbitrage\b"],
    "AUM-2017": [r"relatif [àa] la m[ée]diation"],
    "AUDCIF-2017": [
        r"droit comptable et [àa] l'information financi[èe]re",
        r"syst[èe]me comptable ohada",
    ],
    "AUCTMR-2003": [r"contrats de transport de marchandises par route"],
}

# An explicit reference is preceded by "Acte uniforme", which is what makes it a
# reference to a *text* rather than a passing mention of a subject.
ACTE_UNIFORME = r"[Aa]cte\s+uniforme"

# "article 51 de l'Acte uniforme portant organisation des suretes" -- a pointer
# to one article of another act, which hands over the gold pair exactly.
ARTICLE_LEVEL = re.compile(
    "articles?\\s+([0-9]+(?:-[0-9]+)?)\\s+(?:et\\s+suivants\\s+)?"
    # Both apostrophes: PDF extraction emits the typographic one, and legal
    # French uses it constantly.
    "de\\s+l['\\u2019]\\s*Acte\\s+uniforme([^.;]{0,140})",
    re.IGNORECASE,
)


def load_articles() -> dict[str, dict]:
    articles: dict[str, dict] = {}
    for path in sorted(paths.INTERIM_DIR.glob("*.articles.jsonl")):
        document_id = path.name.removesuffix(".articles.jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                article = json.loads(line)
                articles[f"{document_id}#art{article['number']}"] = article | {
                    "document_id": document_id
                }
    return articles


def find_bridges(articles: dict[str, dict]) -> dict[tuple[str, str], list[dict]]:
    """Articles naming another act, keyed by (source act, target act)."""
    bridges: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for chunk_id, article in articles.items():
        source = article["document_id"]
        text = " ".join(article["text"].split())
        if not re.search(ACTE_UNIFORME, text):
            continue

        for target, patterns in ACT_PATTERNS.items():
            if target == source:
                continue
            if _mentions(text, patterns):
                bridges[(source, target)].append(
                    {"chunk_id": chunk_id, "number": article["number"], "text": text}
                )
    return bridges


def _mentions(text: str, patterns: list[str]) -> bool:
    """Whether an act name appears close after an "Acte uniforme" mention.

    The window matters: an article can discuss sûretés at length and reference a
    different act entirely, and treating both as one reference would manufacture
    bridges that the text does not contain.
    """
    for match in re.finditer(ACTE_UNIFORME, text):
        window = text[match.start() : match.start() + 220]
        if any(re.search(pattern, window, re.IGNORECASE) for pattern in patterns):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, metavar=("SOURCE", "TARGET"))
    parser.add_argument("--article-level", action="store_true")
    parser.add_argument("--chars", type=int, default=300)
    args = parser.parse_args()

    articles = load_articles()
    print(f"articles: {len(articles)}\n")

    if args.article_level:
        print("Article-level pointers -- the source names the exact target article.\n")
        for chunk_id, article in articles.items():
            text = " ".join(article["text"].split())
            for match in ARTICLE_LEVEL.finditer(text):
                tail = " ".join(match.group(2).split())[:100]
                print(f"{chunk_id:24} -> article {match.group(1):8} de l'Acte uniforme {tail}")
        return

    bridges = find_bridges(articles)

    if args.pair:
        key = (args.pair[0], args.pair[1])
        for item in bridges.get(key, []):
            print(f"--- {item['chunk_id']}")
            print(f"    {item['text'][: args.chars]}\n")
        if not bridges.get(key):
            print(f"no bridge found from {key[0]} to {key[1]}")
        return

    total = sum(len(v) for v in bridges.values())
    print(f"act pairs with a textual bridge: {len(bridges)}")
    print(f"bridging articles: {total}\n")
    for (source, target), items in sorted(bridges.items(), key=lambda kv: -len(kv[1])):
        print(f"  {source:14} -> {target:14} {len(items):3}")


if __name__ == "__main__":
    main()
