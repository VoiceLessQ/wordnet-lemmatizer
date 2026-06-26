"""Differential test: the built binary (baked-in data) vs NLTK's WordNetLemmatizer.

NLTK is the oracle; any mismatch is a bug on the Rust side. The corpus is every
WordNet exception-file surface (the irregular path) plus regular rule-based
inflections off a sample of lemmas (the rule path), across all four parts of speech.

Usage: python difftest.py [path-to-binary]
  Defaults to target/release/wordnet-lemmatizer[.exe], falling back to debug.
Exits non-zero on any mismatch.
"""
import itertools
import os
import subprocess
import sys

import nltk

# Ensure the WordNet corpus is available (CI downloads it the first time).
try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer

LEMMA_SAMPLE = 4000  # lemmas per POS to inflect for the rule path


def find_binary():
    if len(sys.argv) > 1:
        return sys.argv[1]
    exe = "wordnet-lemmatizer.exe" if os.name == "nt" else "wordnet-lemmatizer"
    for profile in ("release", "debug"):
        path = os.path.join("target", profile, exe)
        if os.path.exists(path):
            return path
    sys.exit(f"binary not found; build it first (looked in target/release, target/debug)")


def build_corpus():
    wn.ensure_loaded()
    pos_map = {"n": wn.NOUN, "v": wn.VERB, "a": wn.ADJ, "r": wn.ADV}
    pairs = []

    # Exception path: every surface in the .exc lists.
    for code, p in pos_map.items():
        for surface in wn._exception_map[p]:
            pairs.append((surface, code))

    # Rule path: regular inflections off sampled lemmas.
    rules = {
        "n": (wn.NOUN, ["s", "es"]),
        "v": (wn.VERB, ["ing", "ed", "s"]),
        "a": (wn.ADJ, ["er", "est"]),
    }
    for code, (p, suffixes) in rules.items():
        lemmas = itertools.islice(sorted(wn.all_lemma_names(pos=p)), LEMMA_SAMPLE)
        for lemma in lemmas:
            if lemma.isalpha():
                for suf in suffixes:
                    pairs.append((lemma + suf, code))

    return pairs


def main():
    binary = find_binary()
    pairs = build_corpus()
    wnl = WordNetLemmatizer()
    expected = [wnl.lemmatize(word, pos) for word, pos in pairs]

    stdin = "".join(f"{word} {pos}\n" for word, pos in pairs)
    proc = subprocess.run(
        [binary], input=stdin, capture_output=True, text=True, encoding="utf-8"
    )
    if proc.returncode != 0:
        sys.exit(f"binary failed (exit {proc.returncode}):\n{proc.stderr}")
    got = proc.stdout.splitlines()

    if len(got) != len(expected):
        sys.exit(f"line count mismatch: oracle {len(expected)}, binary {len(got)}")

    mismatches = [
        (pairs[i], expected[i], got[i]) for i in range(len(expected)) if expected[i] != got[i]
    ]
    print(f"cases: {len(expected)}  mismatches: {len(mismatches)}")
    for (word, pos), exp, act in mismatches[:25]:
        print(f"  {word} {pos}: nltk={exp!r} binary={act!r}")
    if mismatches:
        sys.exit(1)
    print("ALL MATCH")


if __name__ == "__main__":
    main()
