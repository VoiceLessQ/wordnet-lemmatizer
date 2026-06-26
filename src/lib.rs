//! WordNet morphological lemmatizer, the "morphy" algorithm.
//!
//! Ported from NLTK's `WordNetLemmatizer` and the corpus reader it delegates to
//! (`nltk.corpus.reader.wordnet._morphy`). The algorithm is small and fully
//! deterministic; the work is the data it reads.
//!
//! Provenance: NLTK's code is Apache-2.0. The WordNet corpus (exception lists and
//! the lemma index) carries the separate Princeton WordNet license and is NOT
//! vendored here, it is loaded from WordNet data. This crate ports the algorithm
//! only.
//!
//! The three modes mirror NLTK:
//! - [`Lemmatizer::morphy_all`] = `_morphy`: every lemma found in WordNet.
//! - [`Lemmatizer::morphy`]     = `morphy`: the first such lemma, or `None`.
//! - [`Lemmatizer::lemmatize`]  = `lemmatize`: the shortest lemma, or the input
//!   unchanged when nothing is found.

use std::collections::{HashMap, HashSet};

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

    /// Fold the satellite adjective onto the plain adjective for data lookups,
    /// matching NLTK's `_exception_map[ADJ_SAT] = _exception_map[ADJ]` and the
    /// shared substitution table.
    fn data_pos(self) -> Pos {
        match self {
            Pos::AdjSat => Pos::Adj,
            other => other,
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
        .map(|(old, new)| format!("{}{}", &form[..form.len() - old.len()], new))
        .collect()
}

/// A loaded WordNet lemmatizer: the per-POS exception lists plus the set of valid
/// `(lemma, pos)` pairs that a candidate must be in to count.
pub struct Lemmatizer {
    /// `exceptions[pos][surface] = [lemma, ...]` (the `*.exc` files).
    exceptions: HashMap<Pos, HashMap<String, Vec<String>>>,
    /// Membership test for `_lemma_pos_offset_map`: is `(lemma, pos)` a real entry?
    /// Built from the WordNet `index.{noun,verb,adj,adv}` files (lemma + POS columns).
    lemmas: HashSet<(String, Pos)>,
}

impl Lemmatizer {
    /// Build directly from in-memory data. Mainly for tests and for callers that
    /// load WordNet themselves; see [`Lemmatizer::from_wordnet_dir`] for the file path.
    pub fn new(
        exceptions: HashMap<Pos, HashMap<String, Vec<String>>>,
        lemmas: HashSet<(String, Pos)>,
    ) -> Self {
        Self { exceptions, lemmas }
    }

    /// `_morphy`: every WordNet lemma reachable from `form` under `pos`, in the
    /// order NLTK produces (original form first, then rule rewrites), de-duplicated.
    pub fn morphy_all(&self, form: &str, pos: Pos) -> Vec<String> {
        let p = pos.data_pos();

        // Step 0: exception list short-circuits the rules. Step 1: else apply rules.
        let derived = match self.exceptions.get(&p).and_then(|m| m.get(form)) {
            Some(exc) => exc.clone(),
            None => apply_rules(form, p),
        };

        // Step 2: keep the original plus every derivation that is a real lemma.
        let mut seen = HashSet::new();
        std::iter::once(form.to_string())
            .chain(derived)
            .filter(|f| self.lemmas.contains(&(f.clone(), p)))
            .filter(|f| seen.insert(f.clone()))
            .collect()
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
            .unwrap_or_else(|| word.to_string())
    }

    /// Load exception lists and the lemma index from a WordNet data directory
    /// (the `nltk_data/corpora/wordnet` layout: `noun.exc`, `verb.exc`, `adj.exc`,
    /// `adv.exc`, and `index.{noun,verb,adj,adv}`).
    ///
    /// TODO: first real task. Parse the `.exc` files (two+ whitespace columns:
    /// `surface lemma...`) into `exceptions`, and the `index.*` files (lemma is
    /// column 0, POS is column 1) into `lemmas`. Decide whether to read at runtime
    /// or bake the data in at build time.
    pub fn from_wordnet_dir(_dir: &std::path::Path) -> std::io::Result<Self> {
        todo!("parse *.exc and index.* from the WordNet data dir")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> Lemmatizer {
        // Tiny hand-built WordNet slice covering the NLTK docstring examples.
        let mut exceptions: HashMap<Pos, HashMap<String, Vec<String>>> = HashMap::new();
        exceptions.entry(Pos::Noun).or_default(); // no exception for 'us' or 'dogs'

        let lemmas: HashSet<(String, Pos)> = [
            ("us", Pos::Noun),
            ("u", Pos::Noun),
            ("dog", Pos::Noun),
            ("church", Pos::Noun),
        ]
        .iter()
        .map(|(w, p)| (w.to_string(), *p))
        .collect();

        Lemmatizer::new(exceptions, lemmas)
    }

    #[test]
    fn morphy_all_matches_nltk_us() {
        // >>> wnl()._morphy('us', 'n') -> ['us', 'u']
        assert_eq!(fixture().morphy_all("us", Pos::Noun), vec!["us", "u"]);
    }

    #[test]
    fn lemmatize_picks_shortest() {
        // >>> wnl().lemmatize('us', 'n') -> 'u'  (shortest of ['us','u'])
        assert_eq!(fixture().lemmatize("us", Pos::Noun), "u");
        // >>> wnl().lemmatize('dogs') -> 'dog'
        assert_eq!(fixture().lemmatize("dogs", Pos::Noun), "dog");
        // >>> wnl().lemmatize('churches') -> 'church'
        assert_eq!(fixture().lemmatize("churches", Pos::Noun), "church");
    }

    #[test]
    fn unknown_returns_input_unchanged() {
        // >>> wnl().lemmatize('Anythinggoeszxcv') -> 'Anythinggoeszxcv'
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
}
