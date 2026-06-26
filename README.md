# morphy

A pure-Rust port of WordNet's morphological lemmatizer, the "morphy" algorithm,
as exposed by NLTK's `WordNetLemmatizer`.

Three modes, matching NLTK:

- `morphy_all` (`_morphy`): every lemma WordNet knows for a surface form + POS.
- `morphy` (`morphy`): the first such lemma, or `None`.
- `lemmatize` (`lemmatize`): the shortest lemma, or the input unchanged.

```rust
use morphy::{Lemmatizer, Pos};

let wnl = Lemmatizer::from_wordnet_dir("path/to/nltk_data/corpora/wordnet".as_ref())?;
assert_eq!(wnl.lemmatize("dogs", Pos::Noun), "dog");
assert_eq!(wnl.lemmatize("aardwolves", Pos::Noun), "aardwolf");
```

## Status

Core algorithm and the substitution table are ported and unit-tested against
NLTK's docstring examples (with injected data). The WordNet data loader
(`from_wordnet_dir`) is the next task. See the development notes for the plan.

## Licensing

The algorithm is ported from NLTK (Apache-2.0). The WordNet corpus it reads
(exception lists, lemma index) is under the separate Princeton WordNet license
and is not bundled in this repository.
