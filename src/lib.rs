//! WordNet morphological lemmatizer, the "morphy" algorithm.
//!
//! A faithful, dependency-free port of NLTK's `WordNetLemmatizer` and the corpus
//! reader it delegates to (`nltk.corpus.reader.wordnet._morphy`). The algorithm is
//! small and deterministic; the work is the WordNet data it reads.
//!
//! Unlike a binding, this carries its own data: [`Lemmatizer::embedded`] loads a
//! compact, baked-in slice of WordNet, so the crate needs no runtime files and
//! works in `#![no_std]` (with `alloc`). With the default `std` feature,
//! [`Lemmatizer::from_wordnet_dir`] reads a live WordNet install instead.
//!
//! Provenance: the algorithm is ported from NLTK (Apache-2.0). The bundled data
//! under `data/` is derived from the Princeton WordNet database (see `LICENSE-WORDNET`).
//!
//! Three modes, mirroring NLTK:
//! - [`Lemmatizer::morphy_all`] = `_morphy`: every lemma found in WordNet.
//! - [`Lemmatizer::morphy`]     = `morphy`: the first such lemma, or `None`.
//! - [`Lemmatizer::lemmatize`]  = `lemmatize`: the shortest lemma, or the input
//!   unchanged when nothing is found.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;

/// WordNet part-of-speech tag (`n`, `v`, `a`, `r`, `s`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pos {
    Noun,
    Verb,
    Adj,
    Adv,
    /// Satellite adjective. Shares the adjective rules, exceptions, and index.
    AdjSat,
}

impl Pos {
    /// The WordNet single-letter code.
    pub fn code(self) -> char {
        match self {
            Pos::Noun => 'n',
            Pos::Verb => 'v',
            Pos::Adj => 'a',
            Pos::Adv => 'r',
            Pos::AdjSat => 's',
        }
    }

    /// Fold the satellite adjective onto the plain adjective for data lookups.
    ///
    /// Faithful to NLTK: `_exception_map[ADJ_SAT] = _exception_map[ADJ]`, the
    /// substitution tables are shared, and `_load_lemma_pos_offset_map` sets an
    /// `ADJ_SAT` key for *every* adjective lemma (a possibly-empty list). Since
    /// `filter_forms` only tests key presence, satellite membership equals plain
    /// adjective membership, so the fold is exact.
    fn data_pos(self) -> Pos {
        match self {
            Pos::AdjSat => Pos::Adj,
            other => other,
        }
    }

    /// Index into the four base-POS data slots (Noun, Verb, Adj, Adv).
    fn slot(self) -> usize {
        match self.data_pos() {
            Pos::Noun => 0,
            Pos::Verb => 1,
            Pos::Adj => 2,
            Pos::Adv => 3,
            Pos::AdjSat => unreachable!("folded by data_pos"),
        }
    }
}

/// Suffix-rewrite rules applied once to a surface form, per POS.
///
/// Mirrors NLTK's `MORPHOLOGICAL_SUBSTITUTIONS` verbatim, including the
/// significant ordering (the first valid rewrite wins later tie-breaks).
pub fn substitutions(pos: Pos) -> &'static [(&'static str, &'static str)] {
    match pos.data_pos() {
        Pos::Noun => &[
            ("s", ""),
            ("ses", "s"),
            ("ves", "f"),
            ("xes", "x"),
            ("zes", "z"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("ies", "y"),
        ],
        Pos::Verb => &[
            ("s", ""),
            ("ies", "y"),
            ("es", "e"),
            ("es", ""),
            ("ed", "e"),
            ("ed", ""),
            ("ing", "e"),
            ("ing", ""),
        ],
        Pos::Adj => &[("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
        Pos::Adv => &[],
        Pos::AdjSat => unreachable!("folded by data_pos"),
    }
}

/// Apply every matching suffix rule once. ASCII suffixes only (WordNet is ASCII),
/// so byte slicing is safe here.
fn apply_rules(form: &str, pos: Pos) -> Vec<String> {
    substitutions(pos)
        .iter()
        .filter(|(old, _)| form.ends_with(old))
        .map(|(old, new)| {
            let stem = &form[..form.len() - old.len()];
            let mut s = String::with_capacity(stem.len() + new.len());
            s.push_str(stem);
            s.push_str(new);
            s
        })
        .collect()
}

/// A loaded WordNet lemmatizer.
///
/// Per base POS (Noun, Verb, Adj, Adv) it holds the exception list and the set of
/// valid lemmas, both stored as slices sorted by UTF-8 bytes so lookups are a
/// `binary_search` — no hashing, which keeps the crate `no_std` and allocation-light.
pub struct Lemmatizer {
    /// `lemmas[slot]` = sorted valid lemmas for that POS.
    lemmas: [Vec<String>; 4],
    /// `exceptions[slot]` = `(surface, [lemma, ...])` sorted by surface.
    exceptions: [Vec<(String, Vec<String>)>; 4],
}

impl Lemmatizer {
    /// Build from the compact WordNet slice baked into the crate. No runtime files,
    /// works in `no_std`. The data is pre-sorted, so this only splits lines.
    pub fn embedded() -> Self {
        Lemmatizer {
            lemmas: [
                parse_lemmas(include_str!("../data/lemmas.noun")),
                parse_lemmas(include_str!("../data/lemmas.verb")),
                parse_lemmas(include_str!("../data/lemmas.adj")),
                parse_lemmas(include_str!("../data/lemmas.adv")),
            ],
            exceptions: [
                parse_exc(include_str!("../data/exc.noun")),
                parse_exc(include_str!("../data/exc.verb")),
                parse_exc(include_str!("../data/exc.adj")),
                parse_exc(include_str!("../data/exc.adv")),
            ],
        }
    }

    /// Is `(form, pos)` a real WordNet lemma? (`form in _lemma_pos_offset_map and
    /// pos in _lemma_pos_offset_map[form]`.)
    fn is_lemma(&self, form: &str, pos: Pos) -> bool {
        self.lemmas[pos.slot()]
            .binary_search_by(|l| l.as_str().cmp(form))
            .is_ok()
    }

    /// The exception forms for `(form, pos)`, if any.
    fn exception(&self, form: &str, pos: Pos) -> Option<&[String]> {
        let table = &self.exceptions[pos.slot()];
        table
            .binary_search_by(|(s, _)| s.as_str().cmp(form))
            .ok()
            .map(|i| table[i].1.as_slice())
    }

    /// `_morphy`: every WordNet lemma reachable from `form` under `pos`, in the
    /// order NLTK produces (original form first, then rule rewrites), de-duplicated.
    pub fn morphy_all(&self, form: &str, pos: Pos) -> Vec<String> {
        // Step 0: an exception short-circuits the rules. Step 1: else apply rules.
        let derived = match self.exception(form, pos) {
            Some(forms) => forms.to_vec(),
            None => apply_rules(form, pos),
        };

        // Step 2: keep the original plus every derivation that is a real lemma.
        // Result sets are tiny, so a linear `contains` dedup beats a hash set and
        // keeps us `no_std`.
        let mut result: Vec<String> = Vec::new();
        for cand in core::iter::once(String::from(form)).chain(derived) {
            if self.is_lemma(&cand, pos) && !result.contains(&cand) {
                result.push(cand);
            }
        }
        result
    }

    /// `morphy`: the first lemma, or `None`.
    pub fn morphy(&self, form: &str, pos: Pos) -> Option<String> {
        self.morphy_all(form, pos).into_iter().next()
    }

    /// `lemmatize`: the shortest lemma, or `word` unchanged if none is found.
    ///
    /// On a length tie NLTK's `min(..., key=len)` keeps the *first* candidate;
    /// `reduce` with a strict `<` reproduces that (a later equal-length form does
    /// not replace an earlier one).
    pub fn lemmatize(&self, word: &str, pos: Pos) -> String {
        self.morphy_all(word, pos)
            .into_iter()
            .reduce(|best, f| if f.len() < best.len() { f } else { best })
            .unwrap_or_else(|| String::from(word))
    }
}

/// Parse one baked lemma file: one lemma per line, already sorted.
fn parse_lemmas(text: &str) -> Vec<String> {
    text.lines().map(String::from).collect()
}

/// Parse one baked exception file: `surface lemma...` per line, sorted by surface.
fn parse_exc(text: &str) -> Vec<(String, Vec<String>)> {
    text.lines()
        .filter_map(|line| {
            let mut it = line.split_whitespace();
            let surface = it.next()?;
            Some((String::from(surface), it.map(String::from).collect()))
        })
        .collect()
}

#[cfg(feature = "std")]
mod std_loader {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader};
    use std::path::Path;

    impl Lemmatizer {
        /// Load exception lists and the lemma index from a live WordNet data
        /// directory (the `nltk_data/corpora/wordnet` layout: `noun.exc`,
        /// `verb.exc`, `adj.exc`, `adv.exc`, and `index.{noun,verb,adj,adv}`).
        ///
        /// Mirrors NLTK's `_load_exception_map` / `_load_lemma_pos_offset_map`:
        /// `.exc` lines are `surface lemma...`; `index.*` lines are skipped while
        /// they start with a space (the license header), and column 0 is the lemma.
        /// The result is sorted to match the binary-search lookups.
        pub fn from_wordnet_dir(dir: &Path) -> std::io::Result<Self> {
            const FILEMAP: [&str; 4] = ["noun", "verb", "adj", "adv"];

            let mut lemmas: [Vec<String>; 4] = Default::default();
            let mut exceptions: [Vec<(String, Vec<String>)>; 4] = Default::default();

            for (slot, suffix) in FILEMAP.iter().enumerate() {
                let mut exc = Vec::new();
                for line in BufReader::new(File::open(dir.join(format!("{suffix}.exc")))?).lines() {
                    let line = line?;
                    let mut it = line.split_whitespace();
                    if let Some(surface) = it.next() {
                        exc.push((surface.to_string(), it.map(str::to_string).collect()));
                    }
                }
                exc.sort_by(|a, b| a.0.cmp(&b.0));
                exceptions[slot] = exc;

                let mut lem = Vec::new();
                for line in
                    BufReader::new(File::open(dir.join(format!("index.{suffix}")))?).lines()
                {
                    let line = line?;
                    if line.starts_with(' ') {
                        continue;
                    }
                    if let Some(lemma) = line.split_whitespace().next() {
                        lem.push(lemma.to_string());
                    }
                }
                lem.sort();
                lem.dedup();
                lemmas[slot] = lem;
            }

            Ok(Lemmatizer { lemmas, exceptions })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny hand-built slice covering the NLTK docstring examples, in the same
    /// sorted-slice shape `embedded()` produces.
    fn fixture() -> Lemmatizer {
        let nouns: Vec<String> = ["church", "dog", "u", "us"]
            .iter()
            .map(|s| s.to_string())
            .collect(); // already byte-sorted
        Lemmatizer {
            lemmas: [nouns, Vec::new(), Vec::new(), Vec::new()],
            exceptions: Default::default(),
        }
    }

    #[test]
    fn morphy_all_matches_nltk_us() {
        // >>> wnl()._morphy('us', 'n') -> ['us', 'u']
        assert_eq!(fixture().morphy_all("us", Pos::Noun), vec!["us", "u"]);
    }

    #[test]
    fn lemmatize_picks_shortest() {
        assert_eq!(fixture().lemmatize("us", Pos::Noun), "u");
        assert_eq!(fixture().lemmatize("dogs", Pos::Noun), "dog");
        assert_eq!(fixture().lemmatize("churches", Pos::Noun), "church");
    }

    #[test]
    fn unknown_returns_input_unchanged() {
        assert_eq!(
            fixture().lemmatize("Anythinggoeszxcv", Pos::Noun),
            "Anythinggoeszxcv"
        );
    }

    #[test]
    fn morphy_first_or_none() {
        assert_eq!(fixture().morphy("us", Pos::Noun).as_deref(), Some("us"));
        assert_eq!(fixture().morphy("catss", Pos::Noun), None);
    }

    #[test]
    fn embedded_smoke() {
        // Real baked data: the lemmatize() docstring examples.
        let wnl = Lemmatizer::embedded();
        assert_eq!(wnl.lemmatize("dogs", Pos::Noun), "dog");
        assert_eq!(wnl.lemmatize("aardwolves", Pos::Noun), "aardwolf");
        assert_eq!(wnl.lemmatize("abaci", Pos::Noun), "abacus");
        assert_eq!(wnl.lemmatize("hardrock", Pos::Noun), "hardrock");
        assert_eq!(wnl.lemmatize("us", Pos::Noun), "u");
        // POS matters: 'better' as an adjective lemmatizes via exceptions to 'good'.
        assert_eq!(wnl.lemmatize("better", Pos::Adj), "good");
    }
}
