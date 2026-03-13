use rand::prelude::*;
use std::fs;

fn main() {
    let mut contents: Vec<String> = fs::read_to_string("../input.txt")
        .unwrap_or("".to_string())
        .lines()
        .map(|line| line.trim().to_string())
        .collect();

    println!("num docs: {}", contents.len());

    contents.shuffle(&mut rand::rng());

    let mut uchars: Vec<char> = contents.iter().flat_map(|doc| doc.chars()).collect();
    uchars.sort();
    uchars.dedup();

    let bos = uchars.len();
    let vocab_size = bos + 1;

    println!("vocab size: {}", vocab_size);
}
