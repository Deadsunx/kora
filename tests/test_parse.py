"""Tests for PDF-independent parsing logic.

No PDF and no corpus is required: `articles_from_lines` takes the same
(line, page) pairs the extractor produces, so every segmentation rule can be
exercised against synthetic text in CI.
"""

from __future__ import annotations

from datetime import date

import pytest

from kora.corpus import Document
from kora.documents import Article, ParsedDocument
from kora.ingest.parse import (
    _detect_furniture,
    _is_rubric,
    _match_hierarchy,
    articles_from_lines,
)

DOC = Document(
    id="TEST-2014",
    abbrev="TEST",
    year=2014,
    adopted=date(2014, 1, 30),
    title="Acte uniforme de test",
    domain="test",
    status="in_force",
)


def as_lines(text: str, page: int = 0) -> list[tuple[str, int]]:
    return [(line.strip(), page) for line in text.strip().splitlines() if line.strip()]


# -- basic segmentation -----------------------------------------------------


def test_articles_are_split_on_headings() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article 1
            Le present Acte uniforme s'applique aux societes commerciales.
            Article 2
            Les statuts ne peuvent y deroger.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["1", "2"]
    assert articles[0].text.startswith("Le present Acte")
    assert "Article 2" not in articles[0].text


def test_multiline_bodies_are_joined() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article 1
            Premiere ligne du corps.
            Deuxieme ligne du corps.
            Troisieme ligne.
            """
        ),
        DOC,
    )
    assert len(articles) == 1
    assert "Deuxieme ligne" in articles[0].text
    assert "Troisieme ligne" in articles[0].text


def test_heading_with_trailing_text_on_same_line() -> None:
    """Some headings run straight into the body: `Article 5. Le texte...`"""
    articles = articles_from_lines(
        as_lines(
            """
            Article 5. Le capital social est divise en actions.
            Suite du corps.
            """
        ),
        DOC,
    )
    assert len(articles) == 1
    assert articles[0].number == "5"
    assert articles[0].text.startswith("Le capital social")


# -- inserted articles: the failure that silently drops provisions ----------


@pytest.mark.parametrize("dash", ["-", "\u2011", "\u2013", "\u2014"])
def test_inserted_articles_survive_every_dash_variant(dash: str) -> None:
    """`Article 133-1` must not be truncated to `133`, whichever dash is used.

    Truncation would collide with the real article 133, and the duplicate
    guard would then discard a genuine provision without a word.
    """
    articles = articles_from_lines(
        as_lines(
            f"""
            Article 133
            Corps de l'article cent trente-trois.
            Article 133{dash}1
            Corps de l'article insere.
            Article 134
            Corps de l'article cent trente-quatre.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["133", "133-1", "134"]
    assert articles[1].text.startswith("Corps de l'article insere")


def test_inserted_articles_sort_between_their_neighbours() -> None:
    parsed = ParsedDocument(
        document_id="TEST-2014",
        abbrev="TEST",
        status="in_force",
        title="t",
        page_count=1,
        articles=tuple(
            sorted(
                (
                    Article(
                        document_id="TEST-2014",
                        abbrev="TEST",
                        status="in_force",
                        number=number,
                        text="x",
                        page_start=0,
                        page_end=0,
                    )
                    for number in ["134", "99", "133-2", "100", "133", "133-1"]
                ),
                key=lambda a: a.sort_key,
            )
        ),
        source_sha256="0" * 64,
        parser_version="test",
    )
    assert [a.number for a in parsed.articles] == ["99", "100", "133", "133-1", "133-2", "134"]


# -- "Article premier" ------------------------------------------------------
#
# French legal drafting numbers the first article in words. AUPSRVE-2023 extends
# the convention to inserted articles ("Article premier-11") while referring to
# the same provisions numerically in the body ("l'article 1-10"). A digits-only
# pattern does not merely miss article 1 -- every premier-N heading goes
# unrecognised and its text is absorbed into the preceding article.


@pytest.mark.parametrize("written", ["premier", "Premier", "1er", "1 er"])
def test_first_article_written_in_words(written: str) -> None:
    articles = articles_from_lines(
        as_lines(
            f"""
            Article {written}
            Le present Acte uniforme s'applique aux procedures.
            Article 2
            Corps du deuxieme article.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["1", "2"]
    assert articles[0].text.startswith("Le present Acte")


def test_inserted_articles_under_article_premier() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article premier
            Corps de l'article premier.
            Article premier-10
            Corps de l'article insere dix.
            Article premier-11
            Corps de l'article insere onze.
            Article 2
            Corps du deuxieme.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["1", "1-10", "1-11", "2"]
    assert articles[2].text == "Corps de l'article insere onze."


def test_premier_normalises_to_the_same_key_as_a_numeric_reference() -> None:
    """`premier-11` and `1-11` must denote one article, not two."""
    articles = articles_from_lines(
        as_lines(
            """
            Article premier-11
            Corps unique.
            Article 1-11
            Ceci est une reference, pas une seconde copie.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["1-11"]
    assert "reference" in articles[0].text


def test_article_premier_missing_shows_as_a_numbering_gap() -> None:
    """The check that surfaced this convention in the first place."""
    assert make_parsed(["2", "3", "4"]).numbering_gaps() == [1]


# -- duplicate headings -----------------------------------------------------


def test_repeated_number_is_treated_as_body_text() -> None:
    """A cross-reference printed on its own line must not start a new article."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 425
            Corps de l'article.
            Article 425
            Ceci est une reference, pas une seconde copie.
            """
        ),
        DOC,
    )
    assert len(articles) == 1
    assert "reference" in articles[0].text


# -- table of contents ------------------------------------------------------
#
# The subtlest failure found so far, and the one that defeated both quality
# checks at once. A contents page repeats every article heading followed by dot
# leaders. Naively parsed, each entry becomes a phantom article that claims its
# number, so the genuine article later in the document is dismissed as a
# duplicate and its text appended to whatever article is open. Numbering stays
# contiguous (the phantoms cover every number) and coverage stays high (the text
# is still there, just misfiled), so only the length distribution reveals it.


def test_toc_entries_do_not_become_articles() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            SOMMAIRE
            Article 1
            ..................................................... 3
            Article 2
            ..................................................... 7
            Article 1
            Le present Acte uniforme s'applique au recouvrement.
            Article 2
            Les procedures sont engagees devant la juridiction competente.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["1", "2"]
    assert articles[0].text.startswith("Le present Acte")
    assert articles[1].text.startswith("Les procedures")
    assert "....." not in articles[0].text


def test_toc_does_not_swallow_real_text_into_one_article() -> None:
    """The signature of the bug: one enormous article holding everything."""
    toc = "\n".join(f"Article {n}\n{'.' * 60} {n}" for n in range(1, 6))
    body = "\n".join(f"Article {n}\nCorps de l'article numero {n}." for n in range(1, 6))
    articles = articles_from_lines(as_lines(toc + "\n" + body), DOC)

    assert len(articles) == 5
    lengths = [len(a.text) for a in articles]
    assert max(lengths) < 60, f"one article absorbed the rest: {lengths}"
    for index, article in enumerate(articles, start=1):
        assert article.text == f"Corps de l'article numero {index}."


def test_dot_leader_lines_are_dropped_anywhere() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article 7
            Premiere phrase du corps.
            ............................ 12
            Deuxieme phrase du corps.
            """
        ),
        DOC,
    )
    assert "...." not in articles[0].text
    assert "Premiere phrase" in articles[0].text
    assert "Deuxieme phrase" in articles[0].text


def test_ordinary_ellipsis_is_not_treated_as_a_dot_leader() -> None:
    """Three dots are punctuation; four or more are a contents leader."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 8
            La juridiction statue... sans delai.
            """
        ),
        DOC,
    )
    assert "sans delai" in articles[0].text


@pytest.mark.parametrize(
    "marker",
    ["page 3 / 217", "Page 12 sur 40", "- 14 -", "217", "— 7 —"],
)
def test_page_numbering_is_dropped(marker: str) -> None:
    """Found by spot-checking bodies: one article began 'page 3 / 217'.

    Frequency-based furniture detection cannot catch these -- the text differs
    on every page, so nothing repeats often enough to be recognised.
    """
    articles = articles_from_lines(
        as_lines(
            f"""
            Article 9
            Des epoux ne peuvent etre associes d'une societe.
            {marker}
            Suite du corps.
            """
        ),
        DOC,
    )
    assert marker not in articles[0].text
    assert "Des epoux" in articles[0].text
    assert "Suite du corps" in articles[0].text


def test_numbered_list_items_with_text_survive() -> None:
    """Only a bare numeral is a page marker; an enumeration item is content."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 10
            Les mentions suivantes sont exigees :
            1. la denomination sociale ;
            2. la forme de la societe.
            """
        ),
        DOC,
    )
    assert "denomination sociale" in articles[0].text
    assert "forme de la societe" in articles[0].text


def test_longest_article_flags_merged_text() -> None:
    normal = make_parsed(["1", "2"])
    assert normal.longest_article_chars == len("corps")


# -- repealed articles ------------------------------------------------------
#
# A repealed article prints as its heading followed by the single word
# "Abroge". That word is short, has no terminal full stop and is not a
# structural marker -- so the rubric heuristic classified it as a marginal
# title, stole it from the repealed article, and attached it to the next one.
# Two errors from one heuristic: the repealed article lost its only content,
# and a valid article was made to look repealed.


def test_repealed_article_keeps_its_body() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article 12
            Abrogé
            Article 13
            Les petites entites sont assujetties au systeme minimal de tresorerie.
            """
        ),
        DOC,
    )
    assert [a.number for a in articles] == ["12", "13"]
    assert articles[0].text == "Abrogé"
    assert articles[0].repealed is True


def test_the_following_article_is_not_marked_repealed() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Article 12
            Abrogé
            Article 13
            Les petites entites sont assujetties au systeme minimal de tresorerie.
            """
        ),
        DOC,
    )
    assert articles[1].repealed is False
    assert articles[1].rubric == "", "the repealed marker must not become the next rubric"
    assert articles[1].text.startswith("Les petites entites")


@pytest.mark.parametrize("marker", ["Abrogé", "abrogé", "Abrogée", "Abrogés", "Abrogé."])
def test_repeal_marker_variants(marker: str) -> None:
    articles = articles_from_lines(
        as_lines(f"Article 27\n{marker}\nArticle 28\nCorps reel de l'article."),
        DOC,
    )
    assert articles[0].repealed is True
    assert articles[1].repealed is False


def test_article_discussing_abrogation_is_not_marked_repealed() -> None:
    """Matched against the whole body, so ordinary prose is untouched."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 112
            Sont abrogées à compter de la date d'entree en vigueur du present Acte
            uniforme les dispositions de l'Acte uniforme du 24 mars 2000.
            """
        ),
        DOC,
    )
    assert articles[0].repealed is False


def test_genuine_rubric_is_still_reclaimed() -> None:
    """The fix must not disable rubric capture where it was working."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 476
            Corps du premier article, assez long pour survivre au retrait.
            Nomination du president
            Article 477
            Le conseil designe un president.
            """
        ),
        DOC,
    )
    assert articles[1].rubric == "Nomination du president"
    assert articles[0].text.endswith("au retrait.")
    assert "Nomination" not in articles[0].text


def test_repealed_articles_are_flagged_in_indexed_text() -> None:
    article = Article(
        document_id="AUDCIF-2017",
        abbrev="AUDCIF",
        status="in_force",
        number="12",
        text="Abrogé",
        repealed=True,
        page_start=0,
        page_end=0,
    )
    assert "ABROGÉ" in article.to_indexed_text()


# -- hierarchy --------------------------------------------------------------


def test_hierarchy_is_attached_and_nested() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            LIVRE 1 DISPOSITIONS GENERALES
            TITRE 1 CONSTITUTION
            CHAPITRE 2 APPORTS
            Article 1
            Corps.
            """
        ),
        DOC,
    )
    assert articles[0].hierarchy == (
        "LIVRE 1 DISPOSITIONS GENERALES",
        "TITRE 1 CONSTITUTION",
        "CHAPITRE 2 APPORTS",
    )


def test_entering_a_shallower_level_clears_deeper_ones() -> None:
    """A new TITRE must discard the previous TITRE's CHAPITRE."""
    articles = articles_from_lines(
        as_lines(
            """
            LIVRE 1 A
            TITRE 1 B
            CHAPITRE 1 C
            Article 1
            Corps un.
            TITRE 2 D
            Article 2
            Corps deux.
            """
        ),
        DOC,
    )
    assert articles[0].hierarchy == ("LIVRE 1 A", "TITRE 1 B", "CHAPITRE 1 C")
    assert articles[1].hierarchy == ("LIVRE 1 A", "TITRE 2 D")


@pytest.mark.parametrize(
    ("line", "expected_level"),
    [
        ("LIVRE 2 LES SOCIETES", "LIVRE"),
        ("Titre 3 des apports", "TITRE"),
        # French legal texts number the first division in words, not digits.
        ("CHAPITRE PREMIER", "CHAPITRE"),
        ("Section 3 - President du conseil", "SECTION"),
        ("Le present Acte uniforme s'applique.", None),
    ],
)
def test_hierarchy_matching(line: str, expected_level: str | None) -> None:
    result = _match_hierarchy(line)
    assert (result[0] if result else None) == expected_level


# -- rubrics ----------------------------------------------------------------


def test_rubric_is_captured_from_the_preceding_line() -> None:
    articles = articles_from_lines(
        as_lines(
            """
            Nomination et duree du mandat du president
            Article 477
            Le conseil d'administration designe un president.
            """
        ),
        DOC,
    )
    assert articles[0].rubric == "Nomination et duree du mandat du president"
    assert "Nomination" not in articles[0].text


def test_rubric_is_not_stolen_from_the_previous_article_body() -> None:
    """A sentence ending in a full stop is body text, never a rubric."""
    articles = articles_from_lines(
        as_lines(
            """
            Article 1
            Une phrase qui se termine normalement.
            Article 2
            Corps deux.
            """
        ),
        DOC,
    )
    assert articles[0].text.endswith("normalement.")
    assert articles[1].rubric == ""


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("Nomination du president", True),
        ("Une phrase complete.", False),
        ("Enumeration :", False),
        ("LIVRE 1 GENERALITES", False),
        ("Article 5", False),
        ("x" * 200, False),
        ("", False),
    ],
)
def test_is_rubric(line: str, expected: bool) -> None:
    assert _is_rubric(line) is expected


# -- page furniture ---------------------------------------------------------


def test_furniture_detected_by_frequency_not_hardcoding() -> None:
    pages = [
        f"ACTE UNIFORME DE TEST\nContenu unique de la page {n}.\nwww.example.org/source"
        for n in range(10)
    ]
    furniture = _detect_furniture(pages)
    assert "ACTE UNIFORME DE TEST" in furniture
    assert "www.example.org/source" in furniture
    assert not any("Contenu unique" in line for line in furniture)


def test_repeated_body_sentence_is_not_furniture() -> None:
    """Legal texts repeat formulae; only page edges count as furniture."""
    pages = [
        "HEADER\nligne A\nDans les conditions prevues par le present Acte uniforme.\n"
        "ligne B\nFOOTER"
        for _ in range(10)
    ]
    furniture = _detect_furniture(pages)
    assert "HEADER" in furniture
    assert "FOOTER" in furniture
    assert "Dans les conditions prevues par le present Acte uniforme." not in furniture


# -- quality self-checks ----------------------------------------------------


def make_parsed(numbers: list[str], *, source_chars: int = 0) -> ParsedDocument:
    return ParsedDocument(
        document_id="TEST-2014",
        abbrev="TEST",
        status="in_force",
        title="t",
        page_count=1,
        articles=tuple(
            Article(
                document_id="TEST-2014",
                abbrev="TEST",
                status="in_force",
                number=n,
                text="corps",
                page_start=0,
                page_end=0,
            )
            for n in numbers
        ),
        source_sha256="0" * 64,
        parser_version="test",
        source_chars=source_chars,
    )


def test_numbering_gaps_detects_swallowed_headings() -> None:
    assert make_parsed(["1", "2", "4", "5"]).numbering_gaps() == [3]
    assert make_parsed(["1", "2", "3"]).numbering_gaps() == []


def test_inserted_articles_do_not_create_false_gaps() -> None:
    assert make_parsed(["1", "1-1", "1-2", "2"]).numbering_gaps() == []


def test_text_coverage() -> None:
    parsed = make_parsed(["1", "2"], source_chars=20)
    assert parsed.text_coverage == pytest.approx(10 / 20)
    assert make_parsed(["1"]).text_coverage == 0.0


# -- indexed text -----------------------------------------------------------


def test_indexed_text_carries_citation_and_context() -> None:
    article = Article(
        document_id="AUSCGIE-2014",
        abbrev="AUSCGIE",
        status="in_force",
        number="477",
        rubric="Nomination du president",
        text="Le conseil designe un president.",
        hierarchy=("LIVRE 2", "TITRE 1"),
        page_start=3,
        page_end=3,
    )
    indexed = article.to_indexed_text()
    assert "article 477 AUSCGIE (2014)" in indexed
    assert "LIVRE 2 > TITRE 1" in indexed
    assert "Nomination du president" in indexed
    assert "Le conseil designe" in indexed
    assert article.chunk_id == "AUSCGIE-2014#art477"
