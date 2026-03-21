use rand::prelude::*;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fs;
use std::hash::Hash;
use std::ops::Add;
use std::rc::Rc;

struct ValueInner {
    data: f64,
    grad: f64,
    children: Vec<Value>,
    local_grads: Vec<f64>,
}

#[derive(Clone)]
struct Value(Rc<RefCell<ValueInner>>);

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

    #[allow(clippy::mutable_key_type)]
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

    let a = Value::new(2.0, Vec::new(), Vec::new());
    let b = Value::new(3.0, Vec::new(), Vec::new());
    let c = a.clone() + b.clone();

    println!("a.data(): {}", a.data());
    println!("a.grad(): {}", a.0.borrow().grad);
    println!("b.data(): {}", b.data());
    println!("b.grad(): {}", b.0.borrow().grad);
    println!("c.data(): {}", c.data());
    println!("c.grad(): {}", c.0.borrow().grad);

    println!("running backprop...");
    c.backward();

    println!("a.data(): {}", a.data());
    println!("a.grad(): {}", a.0.borrow().grad);
    println!("b.data(): {}", b.data());
    println!("b.grad(): {}", b.0.borrow().grad);
    println!("c.data(): {}", c.data());
    println!("c.grad(): {}", c.0.borrow().grad);
}
