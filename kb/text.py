"""Text helpers shared by every stage: cleaning crawled markdown, tokenising, numbers in words."""
from __future__ import annotations

import re
from typing import List

# Headings that introduce teasers for *other* articles. A teaser repeats an old fact out of
# context (last year's press release quoted under this year's), so the teaser block is dropped:
# from this heading up to the page's own next section.
_RELATED = re.compile(r"^#{1,6}\s*(handpicked for you|related (articles|posts|stories)|you (may|might) also like)\b", re.I)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LONG_LINK = re.compile(r"\[([^\]]*)\]\([^)\s]{180,}\)")
_EMPTY_LINK = re.compile(r"\[\s*\]\([^)]*\)")
_MODAL = re.compile(r"^\s*(?:✕|\[Continue\]\([^)]*\)\s*Cancel|Cancel)\s*$")
_ESCAPED_CRLF = re.compile(r"\\{1,2}r\\{1,2}n")
_COUNTRY_CODES = re.compile(r"\+\d{1,3}\b")
# A footnote number glued to the value it annotates: "0.20%1", "6.05% APR1", "(APY)1".
_CELL_FOOTNOTE = re.compile(r"(%|APR|APY|\))\s?\d(?=\s*\|)")
_LINE_FOOTNOTE = re.compile(r"^[*_]*\d(?=[A-Z][a-z])")


def clean_markdown(md: str) -> str:
    """Return the page's own content as plain markdown lines, without crawl noise."""
    md = _ESCAPED_CRLF.sub("\n", md)
    out: List[str] = []
    teaser_level = 0          # > 0 while inside a block of teasers for other articles
    for raw in md.splitlines():
        stripped = raw.strip()
        if _RELATED.match(stripped):
            teaser_level = len(stripped) - len(stripped.lstrip("#"))
            continue
        if teaser_level:
            # The teasers end where the page's own next section begins: a heading of the same
            # level or higher. On a product page that is often the FAQ, which must be kept.
            level = len(stripped) - len(stripped.lstrip("#")) if stripped.startswith("#") else 0
            if not 0 < level <= teaser_level:
                continue
            teaser_level = 0
        line = _IMAGE.sub("", raw)
        line = _LONG_LINK.sub(r"\1", line)
        line = _EMPTY_LINK.sub("", line).rstrip().rstrip("\\").rstrip()
        if _MODAL.match(line):
            continue
        if len(_COUNTRY_CODES.findall(line)) >= 15:       # phone-prefix picker of a web form
            continue
        if len(line) > 400 and line.count(" ") < len(line) / 40:   # encoded blobs, data URIs
            continue
        if re.match(r"^#{1,6}\s*\$?0\+?%?\s*$", line):               # animated counter caught at zero
            continue
        if line.lstrip().startswith("|"):
            line = _CELL_FOOTNOTE.sub(r"\1", line)
        else:
            line = _LINE_FOOTNOTE.sub("", line)
        out.append(line)
    text = "\n".join(out)
    text = text.replace("\\*", "*").replace("\\#", "#").replace("\\_", "_").replace("\\$", "$")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# ---------------------------------------------------------------------------------- tokens

_STOP = frozenset(
    "a an the of on in at to for and or is are was were be been do does did what whats which who whom "
    "how can could would should will you your yours i me my we our us it its with from by as that this "
    "these those have has had any there their they them about tell please give get show need want know "
    "if so than then into not no yes also just some more most much many very".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}\b)")


def normalise(text: str) -> str:
    """Lower-case and unify the punctuation variants that crawled pages mix freely."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("–", "-").replace("—", "-")
    return _THOUSANDS.sub("", text)


def stem(token: str) -> str:
    """Crude stemmer: strips plurals ('rates' -> 'rate') and cuts long words to seven letters, so
    that 'recognition', 'recognized' and 'recognizes' all become 'recogni'."""
    if len(token) > 4 and token.endswith("ies"):
        token = token[:-3] + "y"
    elif len(token) > 4 and token.endswith(("ches", "shes", "sses", "xes")):
        token = token[:-2]
    elif len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    return token[:7] if len(token) >= 9 and token.isalpha() else token


def words(text: str) -> List[str]:
    """Every stemmed token, stop words included (used where 'no' or 'it' can carry meaning)."""
    return [stem(t) for t in _TOKEN.findall(normalise(text))]


def tokens(text: str) -> List[str]:
    """Stemmed content tokens for keyword search."""
    return [stem(t) for t in _TOKEN.findall(normalise(text)) if t not in _STOP and (len(t) > 1 or t.isdigit())]


# ------------------------------------------------------------------- shorthand in questions

# How people type in a chat box. Each is written out before a question is read or searched for,
# so "acct" finds "account" and "hrs" finds "hours". Only spellings with one common meaning.
_SHORTHAND = {
    "acct": "account", "accts": "accounts", "info": "information", "hrs": "hours", "hr": "hour",
    "yr": "year", "yrs": "years", "mos": "months", "pls": "please", "plz": "please", "thru": "through",
    "amt": "amount", "addr": "address", "appt": "appointment", "bal": "balance", "pymt": "payment", "pmt": "payment",
    "biz": "business", "mtg": "mortgage", "cert": "certificate", "certs": "certificates", "chkg": "checking",
    "svgs": "savings", "wat": "what", "wht": "what", "ur": "your", "u": "you", "r": "are", "b4": "before",
    "dept": "department", "mgr": "manager", "msg": "message", "tel": "telephone", "ph": "phone", "num": "number",
    "docs": "documents", "doc": "document", "req": "requirement", "reqs": "requirements", "w/": "with", "w/o": "without",
}
_SHORTHAND_WORD = re.compile(r"(?<![\w.'/&-])(%s)(?![\w'/&-]|\.\w)" % "|".join(sorted(map(re.escape, _SHORTHAND), key=len, reverse=True)))


def expand_shorthand(question: str) -> str:
    """'wat are ur hrs' -> 'what are your hours'. Lower-case shorthand only: 'U' in 'U.S.' and
    'R' in 'R&D' are left alone."""
    return _SHORTHAND_WORD.sub(lambda m: _SHORTHAND[m.group(1)], question)


# ------------------------------------------------------------------- the shape of a question

# A greeting, and the sounds people make before a spoken question ("um, so, ..."): voice
# transcripts are full of them, and they hide the word that shows what kind of question follows.
_GREETING = re.compile(r"^\s*(?:(?:hi|hello|hey|good (?:morning|afternoon|evening)|um+|uh+|er+m?|hm+|oh|ah|so|well|ok(?:ay)?|alright|yeah)"
                       r"\b[\s,.!-]*)+", re.I)
_REQUEST = re.compile(
    r"^\s*(?:please\s+)?(?:"
    r"(?:can|could|would|will) (?:you|someone|anyone)(?: please| kindly)?\s+(?:tell|give|show|share|provide|let|get|repeat|confirm|remind|send|list|explain|help)(?: me| us)?(?: with| know| out)?"
    r"|do you (?:happen to )?know|i(?:'d| would) (?:like|love) to (?:know|find out|see|ask)|i (?:want|need|wanted|was hoping) to (?:know|find out|see|ask)"
    r"|i(?:'m| am| was) (?:wondering|curious|trying to find out|looking for)|may i (?:know|ask)|(?:tell|give|show|get) me|let me know|wondering"
    r")\b[\s,:]*(?:about\s+)?", re.I)
_YES_NO = re.compile(r"^\s*(?:and|so|but|also)?\s*(?:(?:do|does|did|is|are|was|were|has|have|had|will|would|can|could|should)(?:n't)?\b|if\b|whether\b)", re.I)


def strip_opening(question: str) -> str:
    """Drop a greeting and spoken fillers from the front: 'Hi, um, when do you open on Saturday?' -> 'when do you open on Saturday?'"""
    return _GREETING.sub("", question)


def strip_request(question: str) -> str:
    """Drop a greeting, spoken fillers and a polite lead-in: 'Hi, um, can you tell me the rate...'
    -> 'the rate...'. What remains shows what kind of question it is."""
    rest = _GREETING.sub("", question)
    return _REQUEST.sub("", rest, count=1)


def is_yes_no(question: str) -> bool:
    """'Do any tiers pay 0.65%?' is; 'Can you tell me the 30-year rate?' is a request, not a yes/no."""
    return bool(_YES_NO.match(strip_request(question)))


# ---------------------------------------------------------------- numbers written as words

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), start=2)}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000}
_NUMBER_WORD = "|".join(list(_UNITS) + list(_TENS) + list(_SCALES))
_NUMBER_PHRASE = re.compile(r"\b(?:(?:%s)(?:[\s-]+(?:and[\s-]+)?(?=(?:%s)\b))?)+\b" % (_NUMBER_WORD, _NUMBER_WORD), re.I)


def _phrase_value(phrase: str) -> int:
    total, current = 0, 0
    for w in re.findall(r"[a-z]+", phrase.lower()):
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = max(current, 1) * 100
        elif w in _SCALES:
            total += max(current, 1) * _SCALES[w]
            current = 0
    return total + current


def digits_for_number_words(text: str) -> str:
    """'a five year loan' -> 'a 5 year loan'; 'fifteen thousand dollars' -> '15000 dollars'."""

    def swap(m: "re.Match[str]") -> str:
        phrase = m.group(0)
        if phrase.strip().lower() == "one":          # "which one", "no one": usually not a quantity
            if not re.match(r"[\s-]*(year|month|percent)", m.string[m.end():], re.I):
                return phrase
        tail = phrase[len(phrase.rstrip()):]
        return str(_phrase_value(phrase)) + tail

    return _NUMBER_PHRASE.sub(swap, text)


def sentences(text: str) -> List[str]:
    """Split prose into sentences; table rows and list items stay whole lines."""
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("|"):
            out.append(line)
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(*\[])", line) if s.strip())
    return out
