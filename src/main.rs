mod model;
mod value;

use clap::Parser;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use model::Model;
use rand::prelude::*;
use std::fs;
use std::io::Write;
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

    // Sonify training steps
    #[arg(long, default_value_t = false)]
    sonify: bool,
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
            println!();
            println!("saving checkpoint to {}", path);
            model.save(path).expect("failed to save checkpoint");
            model
        }
    } else {
        train(&args)
    };

    println!();
    println!("--- inference (new, hallucinated names) ---");

    model
        .hallucinate(args.temperature, args.num_samples)
        .iter()
        .enumerate()
        .for_each(|(index, sample)| {
            println!("sample {:4}: {}", index, sample);
        })
}

fn train(args: &Args) -> Model {
    use std::sync::atomic::Ordering;

    let should_stop = setup_ctrlc_handler();
    let (contents, uchars) = import_training_set("../input.txt");

    let model = Model::new(
        uchars,
        args.layers,
        args.embeddings,
        args.block_size,
        args.heads,
    );
    println!("vocab size: {}", model.vocab_size());
    println!("num params: {}", model.params().len());

    let mut sonification = args.sonify.then(setup_sonifier).flatten();

    let steps = args.steps;

    let _ = model
        .train(args.steps, contents)
        .inspect(|(step, loss, _)| {
            print!("\rstep {step:4} / {steps:4} | loss {loss:.4}");
            std::io::stdout().flush().unwrap();
        })
        .inspect(|step_data| {
            if let Some((_, sonify)) = &mut sonification {
                sonify(step_data);
            }
        })
        .map_while(|_| {
            if should_stop.load(Ordering::Relaxed) {
                None
            } else {
                Some(())
            }
        })
        .collect::<Vec<_>>();

    model
}

fn setup_sonifier() -> Option<(cpal::Stream, impl FnMut(&(usize, f64, Vec<f64>)))> {
    let (tx, rx) = std::sync::mpsc::channel::<Vec<f32>>();
    let device = cpal::default_host().default_output_device()?;
    let supported_config = device.default_output_config().ok()?;
    dbg!(&supported_config);

    let mut wavetable: Vec<f32> = vec![0.0];
    let mut pos = 0;

    let stream = device
        .build_output_stream(
            &supported_config.config(),
            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                while let Ok(new) = rx.try_recv() {
                    wavetable = new;
                }

                for out in output.iter_mut() {
                    *out = wavetable[pos % wavetable.len()];
                    pos += 1
                }
            },
            |err| eprintln!("{err}"),
            None,
        )
        .ok()?;

    let sonifier = move |(_step, _loss, params): &(usize, f64, Vec<f64>)| {
        let max_abs = params
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
            .max(1e-8);
        let values: Vec<f32> = params.iter().map(|v| (v / max_abs * 0.3) as f32).collect();
        let _ = tx.send(values);
    };

    stream.play().unwrap();

    Some((stream, sonifier))
}

fn import_training_set(filename: &str) -> (Vec<String>, Vec<char>) {
    let mut contents: Vec<String> = fs::read_to_string(filename)
        .unwrap_or("".to_string())
        .lines()
        .map(|line| line.trim().to_string())
        .collect();

    println!("num docs: {}", contents.len());

    contents.shuffle(&mut rand::rng());

    let mut uchars: Vec<char> = contents.iter().flat_map(|doc| doc.chars()).collect();
    uchars.sort();
    uchars.dedup();

    (contents, uchars)
}

fn setup_ctrlc_handler() -> std::sync::Arc<std::sync::atomic::AtomicBool> {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    let should_stop = Arc::new(AtomicBool::new(false));
    let stop_flag = should_stop.clone();

    ctrlc::set_handler(move || {
        stop_flag.store(true, Ordering::Relaxed);
    })
    .expect("failed to set ctrl-c handler");

    should_stop
}
