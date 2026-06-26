# wordnet-lemmatizer

[![ci](https://github.com/VoiceLessQ/wordnet-lemmatizer/actions/workflows/ci.yml/badge.svg)](https://github.com/VoiceLessQ/wordnet-lemmatizer/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/wordnet-lemmatizer.svg)](https://crates.io/crates/wordnet-lemmatizer)

A dependency-free Rust port of WordNet's morphological lemmatizer, the "morphy"
algorithm exposed by NLTK's `WordNetLemmatizer`.

Unlike a binding, it carries its own data: a compact slice of WordNet is baked into
the crate, so it needs no runtime files and builds for `#![no_std]` (with `alloc`).
It is verified to reproduce NLTK exactly.

Three modes, matching NLTK:

- `morphy_all` (`_morphy`): every lemma WordNet knows for a surface form + POS.
- `morphy` (`morphy`): the first such lemma, or `None`.
- `lemmatize` (`lemmatize`): the shortest lemma, or the input unchanged.

```rust
use wordnet_lemmatizer::{Lemmatizer, Pos};

let wnl = Lemmatizer::embedded();            // baked-in data, no files needed
assert_eq!(wnl.lemmatize("dogs", Pos::Noun), "dog");
assert_eq!(wnl.lemmatize("aardwolves", Pos::Noun), "aardwolf");
assert_eq!(wnl.lemmatize("better", Pos::Adj), "good");
```

## no_std

Default features pull in `std` for `Lemmatizer::from_wordnet_dir` (read a live
WordNet install) and the CLI. Turn them off for a `no_std` + `alloc` build that uses
only the embedded data:

```toml
[dependencies]
wordnet-lemmatizer = { version = "0.1", default-features = false }
```

## Verification

`Lemmatizer::embedded()` is differential-tested against Python `nltk` 3.9.4
(`WordNetLemmatizer().lemmatize`): **67,000+ cases, zero mismatches**. The corpus
covers every WordNet exception-file surface, regular rule-based inflections, and an
edge-case fuzz pass (random/garbage strings, mixed case, multiword underscore lemmas,
non-ASCII), across all five POS codes (n/v/a/r/s).

## Licensing

The algorithm is ported from NLTK (Apache-2.0). The bundled data under `data/` is
derived from the Princeton WordNet 3.0 database; see `LICENSE-WORDNET`.
