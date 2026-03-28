use rand::prelude::*;
use rand_distr::Normal;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::hash::Hash;
use std::io::Write;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;

struct ValueInner {
    data: f64,
    grad: f64,
    children: Vec<Value>,
    local_grads: Vec<f64>,
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueInner>>);

#[allow(clippy::mutable_key_type, dead_code)]
impl Value {
    pub fn new(data: f64, children: Vec<Value>, local_grads: Vec<f64>) -> Value {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }

    pub fn sub_data(&mut self, amount: f64) {
        self.0.borrow_mut().data -= amount
    }

    pub fn reset_grad(&mut self) {
        self.0.borrow_mut().grad = 0.0
    }

    pub fn backward(&self) {
        let mut topology: Vec<Value> = Vec::new();
        let mut visited: HashSet<Value> = HashSet::new();
        self.build_topology(&mut topology, &mut visited);

        self.0.borrow_mut().grad = 1.0;

        for value in topology.iter().rev() {
            let (value_grad, children, local_grads) = {
                let inner = value.0.borrow();
                (
                    inner.grad,
                    inner.children.clone(),
                    inner.local_grads.clone(),
                )
            };

            for (child, local_grad) in children.iter().zip(local_grads.iter()) {
                child.0.borrow_mut().grad += local_grad * value_grad;
            }
        }
    }

    fn build_topology(&self, topology: &mut Vec<Value>, visited: &mut HashSet<Value>) {
        if !visited.contains(self) {
            visited.insert(self.clone());

            for child in &self.0.borrow().children {
                child.build_topology(topology, visited);
            }

            topology.push(self.clone());
        }
    }

    pub fn pow(self, exponent: f64) -> Self {
        Value::new(
            self.data().powf(exponent),
            vec![self.clone()],
            vec![exponent * self.data().powf(exponent - 1.0)],
        )
    }

    pub fn log(self) -> Self {
        Value::new(
            self.data().ln(),
            vec![self.clone()],
            vec![1.0 / self.data()],
        )
    }

    pub fn exp(self) -> Self {
        Value::new(
            self.data().exp(),
            vec![self.clone()],
            vec![self.data().exp()],
        )
    }

    pub fn relu(self) -> Self {
        Value::new(
            f64::max(0.0, self.data()),
            vec![self.clone()],
            vec![if self.data() > 0.0 { 1.0 } else { 0.0 }],
        )
    }
}

impl Add for Value {
    type Output = Value;

    fn add(self, other: Self) -> Self::Output {
        Value::new(
            self.data() + other.data(),
            vec![self, other],
            vec![1.0, 1.0],
        )
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, other: f64) -> Self::Output {
        let other: Value = Value::new(other, vec![], vec![]);

        self + other
    }
}

impl Mul for Value {
    type Output = Value;

    fn mul(self, other: Self) -> Self::Output {
        Value::new(
            self.data() * other.data(),
            vec![self.clone(), other.clone()],
            vec![other.data(), self.data()],
        )
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, other: f64) -> Self::Output {
        let other: Value = Value::new(other, vec![], vec![]);

        self * other
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        other * self
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * -1.0
    }
}

impl Sub for Value {
    type Output = Value;

    fn sub(self, other: Self) -> Self::Output {
        self + -other
    }
}

impl Sub<f64> for Value {
    type Output = Value;

    fn sub(self, other: f64) -> Self::Output {
        self + -other
    }
}

impl Div for Value {
    type Output = Value;

    fn div(self, other: Self) -> Self::Output {
        self * other.pow(-1.0)
    }
}

impl Div<f64> for Value {
    type Output = Value;

    fn div(self, other: f64) -> Self::Output {
        self * (1.0 / other)
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Value {}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        Rc::as_ptr(&self.0).hash(state)
    }
}

type Matrix = Vec<Vec<Value>>;

fn matrix(rows: usize, cols: usize) -> Matrix {
    let std_dev: f64 = 0.08;

    (0..rows)
        .map(|_| {
            (0..cols)
                .map(|_| {
                    Value::new(
                        Normal::new(0.0, std_dev).unwrap().sample(&mut rand::rng()),
                        vec![],
                        vec![],
                    )
                })
                .collect()
        })
        .collect()
}

struct Model {
    network: HashMap<String, Matrix>,
    uchars: Vec<char>,
    block_size: usize,
    layers: usize,
    embeddings: usize,
    heads: usize,
}

impl Model {
    pub fn new(
        uchars: Vec<char>,
        layers: usize,
        embeddings: usize,
        block_size: usize,
        heads: usize,
    ) -> Self {
        let vocab_size = uchars.len() + 1;

        let mut network: HashMap<String, Matrix> = HashMap::from([
            (String::from("wte"), matrix(vocab_size, embeddings)),
            (String::from("wpe"), matrix(block_size, embeddings)),
            (String::from("lm_head"), matrix(vocab_size, embeddings)),
        ]);

        (0..layers).for_each(|i| {
            network.insert(
                format!("layer{}.attn_wq", i),
                matrix(embeddings, embeddings),
            );

            network.insert(
                format!("layer{}.attn_wk", i),
                matrix(embeddings, embeddings),
            );

            network.insert(
                format!("layer{}.attn_wv", i),
                matrix(embeddings, embeddings),
            );

            network.insert(
                format!("layer{}.attn_wo", i),
                matrix(embeddings, embeddings),
            );

            network.insert(
                format!("layer{}.mlp_fc1", i),
                matrix(4 * embeddings, embeddings),
            );

            network.insert(
                format!("layer{}.mlp_fc2", i),
                matrix(embeddings, 4 * embeddings),
            );
        });

        Model {
            network,
            uchars,
            block_size,
            layers,
            embeddings,
            heads,
        }
    }

    pub fn params(&self) -> Vec<Value> {
        self.network
            .values()
            .flat_map(|m| m.iter().flatten())
            .cloned()
            .collect()
    }

    pub fn vocab_size(&self) -> usize {
        self.bos() + 1
    }

    fn bos(&self) -> usize {
        self.uchars.len()
    }

    pub fn train(self, steps: usize, documents: Vec<String>) {
        let mut params: Vec<Value> = self.params();
        let mut m = vec![0.0; params.len()];
        let mut v = vec![0.0; params.len()];

        (0..steps).for_each(|step| {
            let doc: String = documents[step % documents.len()].to_string();

            let mut tokens: Vec<usize> = Vec::new();
            tokens.push(self.bos());
            tokens.append(
                &mut doc
                    .chars()
                    .map(|ch| self.uchars.iter().position(|uc| ch == *uc).unwrap())
                    .collect(),
            );
            tokens.push(self.bos());

            let n = usize::min(self.block_size, tokens.len() - 1);

            let mut keys: Vec<Matrix> = vec![vec![]; self.layers];
            let mut values: Vec<Matrix> = vec![vec![]; self.layers];

            let losses: Vec<Value> = (0..n)
                .map(|position_id| {
                    let probabilities: Vec<Value> = Self::softmax(self.gpt(
                        tokens[position_id],
                        position_id,
                        &mut keys,
                        &mut values,
                    ));

                    let target_id: usize = tokens[position_id + 1];
                    -probabilities[target_id].clone().log()
                })
                .collect();

            let loss: Value = losses
                .iter()
                .cloned()
                .reduce(|acc, next| acc + next)
                .unwrap()
                / (n as f64);

            loss.backward();

            // learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
            let learning_rate = 0.01;
            let beta1 = 0.85;
            let beta2 = 0.99;
            let eps_adam = 1e-8;
            let decayed_learning_rate = learning_rate * (1.0 - step as f64 / steps as f64);

            params.iter_mut().enumerate().for_each(|(i, param)| {
                m[i] = beta1 * m[i] + (1.0 - beta1) * param.grad();
                v[i] = beta2 * v[i] + (1.0 - beta2) * param.grad().powi(2);
                let m_hat = m[i] / (1.0 - beta1.powi(step as i32 + 1));
                let v_hat = v[i] / (1.0 - beta2.powi(step as i32 + 1));

                param.sub_data(decayed_learning_rate * m_hat / (v_hat.powf(0.5) + eps_adam));
                param.reset_grad();
            });

            print!(
                "\rstep {:4} / {:4} | loss {:.4}",
                step + 1,
                steps,
                loss.data()
            );
            std::io::stdout().flush().unwrap();
        })
    }

    fn softmax(logits: Vec<Value>) -> Vec<Value> {
        let max_value: f64 = logits
            .iter()
            .map(|v| v.data())
            .max_by(|a, b| a.total_cmp(b))
            .unwrap();

        let exponentiated: Vec<Value> = logits
            .iter()
            .map(|v| (v.clone() - max_value).exp())
            .collect();

        let total: Value = exponentiated
            .iter()
            .cloned()
            .reduce(|acc, next| acc + next)
            .unwrap();

        exponentiated
            .iter()
            .map(|v| v.clone() / total.clone())
            .collect()
    }

    fn gpt(
        &self,
        token_id: usize,
        position_id: usize,
        keys: &mut [Matrix],
        values: &mut [Matrix],
    ) -> Vec<Value> {
        let token_embedding = self.network["wte"][token_id].clone();
        let position_embedding = self.network["wpe"][position_id].clone();
        let mut x: Vec<Value> = Self::rmsnorm(
            token_embedding
                .into_iter()
                .zip(position_embedding)
                .map(|(token_value, position_value)| token_value + position_value)
                .collect(),
        );

        (0..self.layers).for_each(|layer| {
            let mut x_residual = x.clone();
            x = Self::rmsnorm(x.clone());

            let q: Vec<Value> = Self::linear(&x, &self.network[&format!("layer{}.attn_wq", layer)]);
            let k: Vec<Value> = Self::linear(&x, &self.network[&format!("layer{}.attn_wk", layer)]);
            let v: Vec<Value> = Self::linear(&x, &self.network[&format!("layer{}.attn_wv", layer)]);

            keys[layer].push(k);
            values[layer].push(v);

            let x_attention: Vec<Value> = (0..self.heads)
                .flat_map(|head| {
                    let head_offset: usize = head * self.head_dim();
                    let query_at_head = &q[head_offset..head_offset + self.head_dim()];
                    let key_at_head: Vec<&[Value]> = keys[layer]
                        .iter()
                        .map(|key| &key[head_offset..head_offset + self.head_dim()])
                        .collect();
                    let value_at_head: Vec<&[Value]> = values[layer]
                        .iter()
                        .map(|key| &key[head_offset..head_offset + self.head_dim()])
                        .collect();

                    let attention_scores = (0..key_at_head.len())
                        .map(|key_position| {
                            let dot_product = (0..self.head_dim())
                                .map(|head_offset| {
                                    query_at_head[head_offset].clone()
                                        * key_at_head[key_position][head_offset].clone()
                                })
                                .reduce(|acc, next| acc + next)
                                .unwrap();

                            dot_product / (self.head_dim() as f64).sqrt()
                        })
                        .collect();

                    let attention_scores_weighted = Self::softmax(attention_scores);
                    (0..self.head_dim())
                        .map(|head_offset| {
                            (0..value_at_head.len())
                                .map(|value_position| {
                                    attention_scores_weighted[value_position].clone()
                                        * value_at_head[value_position][head_offset].clone()
                                })
                                .reduce(|acc, next| acc + next)
                                .unwrap()
                        })
                        .collect::<Vec<Value>>()
                })
                .collect();

            x = Self::linear(
                &x_attention,
                &self.network[&format!("layer{}.attn_wo", layer)],
            );

            x = x
                .iter()
                .cloned()
                .zip(x_residual)
                .map(|(a, b)| a + b)
                .collect();

            x_residual = x.clone();

            x = Self::rmsnorm(x.clone());
            x = Self::linear(&x, &self.network[&format!("layer{}.mlp_fc1", layer)]);
            x = x.iter().cloned().map(|value| value.relu()).collect();
            x = Self::linear(&x, &self.network[&format!("layer{}.mlp_fc2", layer)]);
            x = x
                .iter()
                .cloned()
                .zip(x_residual)
                .map(|(a, b)| a + b)
                .collect();
        });

        Self::linear(&x, &self.network["lm_head"])
    }

    fn rmsnorm(values: Vec<Value>) -> Vec<Value> {
        let ms: Value = values
            .iter()
            .map(|v| v.clone().pow(2.0))
            .reduce(|acc, next| acc + next)
            .unwrap()
            / (values.len() as f64);
        let scale: Value = (ms + 1e-5).pow(-0.5);

        values
            .into_iter()
            .map(|v| v.clone() * scale.clone())
            .collect()
    }

    fn linear(x: &[Value], w: &Matrix) -> Vec<Value> {
        w.iter()
            .map(|row| {
                row.iter()
                    .zip(x.iter())
                    .map(|(wi, xi)| wi.clone() * xi.clone())
                    .reduce(|acc, next| acc + next)
                    .unwrap()
            })
            .collect()
    }

    fn head_dim(&self) -> usize {
        self.embeddings / self.heads
    }
}

#[allow(unused_variables)]
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

    model.train(1000, contents)
}
