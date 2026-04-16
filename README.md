# microgpt-rs

A more-or-less literal port of [Andrej Karpathy's microgpt.py](https://karpathy.github.io/2026/02/12/microgpt/) to Rust that you can hear as it trains the model.

## Usage

Basic usage is:

```
cargo run --release -- <input_file> [options]
```

Where `<input_file>` is the training set.

Unlike the python version, this won't download a training set for you. You'll have to set it up yourself.

For example, to hallucinate Portuguese/English/French mongrel words:

```
base=https://raw.githubusercontent.com/kkrypt0nn/wordlists/refs/heads/main/wordlists/languages

for lang in portuguese english french; do
  curl -s $base/$lang.txt
done >> words

cargo run --release -- words
```

Output:
```
    Finished `release` profile [optimized] target(s) in 0.04s
     Running `target/release/microgpt-rs words`
num docs: 1003193
vocab size: 83
num params: 5984
step  999 / 1000 | loss 2.8018
--- inference (new, hallucinated names) ---
sample    0: bongonos
sample    1: itisices
sample    2: canaçãos
sample    3: bauiase
sample    4: comsconeras
sample    5: tamontica
sample    6: qulontacas
sample    7: diserilba
sample    8: caluitain
sample    9: restisis
sample   10: cacate
sample   11: diciais
sample   12: suilins
sample   13: conthites
sample   14: pidires
sample   15: luisico
sample   16: conres
sample   17: paconco
sample   18: engins
sample   19: rameser
```

(When using the output of any transformer, please be mindful of the copyrighted creative works that contributed to the model's training data.)

To synthesize a drone from the model weights as it trains, add the `--sonify` flag. Play with the model parameters and see how that changes the timbre of the drone. You'll probably want to increase the number of steps to make it sound for a long time.

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `-f, --file` | Checkpoint path (loads if exists, saves after training) | — |
| `-s, --steps` | Training steps | 1000 |
| `-t, --temperature` | Sampling temperature (0.0–1.0) | 0.5 |
| `-n, --num-samples` | Number of samples to generate | 20 |
| `--layers` | Transformer layers | 1 |
| `--embeddings` | Embedding dimension | 16 |
| `--block-size` | Max context length | 16 |
| `--heads` | Attention heads | 4 |
| `--sonify` | Output weights as audio during training | false |
