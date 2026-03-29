mod model;
mod value;

use model::Model;
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

    let model = Model::new(uchars, 1, 16, 16, 4);
    println!("vocab size: {}", model.vocab_size());
    println!("num params: {}", model.params().len());

    model.train(1000, contents);

    println!("--- inference (new, hallucinated names) ---");

    for (index, sample) in model.hallucinate(0.5, 20).iter().enumerate() {
        println!("sample {:4}: {}", index, sample);
    }
}
