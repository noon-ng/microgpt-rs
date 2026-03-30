mod model;
mod value;

use clap::Parser;
use ctrlc;
use model::Model;
use rand::prelude::*;
use std::fs;
use std::path::Path;

#[derive(Parser)]
#[command(name = "microgpt-rs", about = "A tiny GPT in Rust")]
struct Args {
    /// Checkpoint file (loads if exists, saves after training if not)
    #[arg(short, long)]
    file: Option<String>,

    /// Number of training steps
    #[arg(short, long, default_value_t = 1000)]
    steps: usize,

    /// Sampling temperature (0.0-1.0, lower = more conservative)
    #[arg(short, long, default_value_t = 0.5)]
    temperature: f64,

    /// Number of names to generate
    #[arg(short, long, default_value_t = 20)]
    num_samples: usize,

    /// Number of transformer layers
    #[arg(long, default_value_t = 1)]
    layers: usize,

    /// Embedding dimension
    #[arg(long, default_value_t = 16)]
    embeddings: usize,

    /// Maximum context length
    #[arg(long, default_value_t = 16)]
    block_size: usize,

    /// Number of attention heads
    #[arg(long, default_value_t = 4)]
    heads: usize,
}

fn main() {
    let args = Args::parse();

    let model = if let Some(ref path) = args.file {
        if Path::new(path).exists() {
            println!("loading checkpoint from {}", path);
            let model = Model::load(path).expect("failed to load checkpoint");
            println!("vocab size: {}", model.vocab_size());
            println!("num params: {}", model.params().len());
            model
        } else {
            let model = train(&args);
            println!("saving checkpoint to {}", path);
            model.save(path).expect("failed to save checkpoint");
            model
        }
    } else {
        train(&args)
    };

    println!("--- inference (new, hallucinated names) ---");

    for (index, sample) in model
        .hallucinate(args.temperature, args.num_samples)
        .iter()
        .enumerate()
    {
        println!("sample {:4}: {}", index, sample);
    }
}

fn train(args: &Args) -> Model {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let should_stop = Arc::new(AtomicBool::new(false));
    let stop_flag = should_stop.clone();

    ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::Relaxed);
    })
    .expect("failed to set ctrl-c handler");

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

    let model = Model::new(
        uchars,
        args.layers,
        args.embeddings,
        args.block_size,
        args.heads,
    );
    println!("vocab size: {}", model.vocab_size());
    println!("num params: {}", model.params().len());

    model.train(args.steps, contents, &should_stop);

    model
}
