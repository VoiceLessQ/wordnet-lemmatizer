//! Thin CLI over the lemmatizer, for differential testing against NLTK.
//!
//! Reads `word [pos]` per line from stdin (pos one of n/v/a/r/s, default n) and
//! prints `lemmatize(word, pos)` per line. The WordNet data dir comes from argv[1]
//! or `$NLTK_DATA/corpora/wordnet`.

use std::io::{self, BufRead, Write};
use std::path::PathBuf;

use morphy::{Lemmatizer, Pos};

fn parse_pos(s: &str) -> Option<Pos> {
    Some(match s {
        "n" => Pos::Noun,
        "v" => Pos::Verb,
        "a" => Pos::Adj,
        "r" => Pos::Adv,
        "s" => Pos::AdjSat,
        _ => return None,
    })
}

fn main() -> io::Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var_os("NLTK_DATA")
                .map(|d| PathBuf::from(d).join("corpora").join("wordnet"))
        })
        .expect("pass the WordNet data dir as argv[1] or set NLTK_DATA");

    let wnl = Lemmatizer::from_wordnet_dir(&dir)?;

    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout().lock());
    for line in stdin.lock().lines() {
        let line = line?;
        let mut it = line.split_whitespace();
        let Some(word) = it.next() else { continue };
        let pos = it.next().and_then(parse_pos).unwrap_or(Pos::Noun);
        writeln!(out, "{}", wnl.lemmatize(word, pos))?;
    }
    Ok(())
}
