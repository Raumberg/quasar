use criterion::{black_box, criterion_group, criterion_main, Criterion};
use quasar::prelude::*;

fn benchmark_addition(c: &mut Criterion) {
    let a = Tensor::new(vec![1.0; 1000], Shape::from(&[1000])).unwrap();
    let b = Tensor::new(vec![2.0; 1000], Shape::from(&[1000])).unwrap();
    
    c.bench_function("tensor_addition_1000", |bench| {
        bench.iter(|| {
            let result = black_box(&a) + black_box(&b);
            black_box(result)
        })
    });
}

fn benchmark_multiplication(c: &mut Criterion) {
    let a = Tensor::new(vec![1.5; 1000], Shape::from(&[1000])).unwrap();
    let b = Tensor::new(vec![2.5; 1000], Shape::from(&[1000])).unwrap();
    
    c.bench_function("tensor_multiplication_1000", |bench| {
        bench.iter(|| {
            let result = black_box(&a) * black_box(&b);
            black_box(result)
        })
    });
}

fn benchmark_backward_pass(c: &mut Criterion) {
    c.bench_function("backward_pass_chain_rule", |bench| {
        bench.iter(|| {
            let x = Tensor::new(vec![2.0, 3.0], Shape::from(&[2])).unwrap().requires_grad(true);
            let y = Tensor::new(vec![4.0, 5.0], Shape::from(&[2])).unwrap().requires_grad(true);
            
            let sum = (&x + &y).unwrap();
            let diff = (&x - &y).unwrap();
            let mut result = (&sum * &diff).unwrap();
            
            result.backward().unwrap();
            black_box(result)
        })
    });
}

criterion_group!(benches, benchmark_addition, benchmark_multiplication, benchmark_backward_pass);
criterion_main!(benches); 