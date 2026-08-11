use cuda_operator::{CudaStream, Tensor, flash_attention, flash_attention_on_stream};

fn input(len: usize, modulus: usize, scale: f32, offset: isize) -> Vec<f32> {
    (0..len)
        .map(|index| ((index % modulus) as isize + offset) as f32 * scale)
        .collect()
}

fn reference(q: &[f32], k: &[f32], v: &[f32], shape: [usize; 4], causal: bool) -> Vec<f32> {
    let [batch_size, head_num, seq_len, head_dim] = shape;
    let mut output = vec![0.0; q.len()];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for batch in 0..batch_size {
        for head in 0..head_num {
            let base = (batch * head_num + head) * seq_len * head_dim;
            for row in 0..seq_len {
                let key_end = if causal { row + 1 } else { seq_len };
                let mut scores = vec![0.0; key_end];
                for column in 0..key_end {
                    let mut dot = 0.0;
                    for dimension in 0..head_dim {
                        dot += q[base + row * head_dim + dimension]
                            * k[base + column * head_dim + dimension];
                    }
                    scores[column] = dot * scale;
                }

                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores
                    .iter_mut()
                    .map(|score| {
                        *score = (*score - max_score).exp();
                        *score
                    })
                    .sum();

                for dimension in 0..head_dim {
                    let mut value = 0.0;
                    for column in 0..key_end {
                        value +=
                            scores[column] / denominator * v[base + column * head_dim + dimension];
                    }
                    output[base + row * head_dim + dimension] = value;
                }
            }
        }
    }
    output
}

fn assert_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let tolerance = 2.0e-4 + 2.0e-4 * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "mismatch at {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}

fn run_case(causal: bool) {
    let shape = [1, 2, 17, 64];
    let len = shape.into_iter().product();
    let q_host = input(len, 31, 0.01, -15);
    let k_host = input(len, 29, 0.0125, -14);
    let v_host = input(len, 23, 0.02, -11);
    let expected = reference(&q_host, &k_host, &v_host, shape, causal);

    let q = Tensor::from_slice(&q_host, shape).unwrap();
    let k = Tensor::from_slice(&k_host, shape).unwrap();
    let v = Tensor::from_slice(&v_host, shape).unwrap();
    let actual = flash_attention(&q, &k, &v, causal)
        .unwrap()
        .to_vec()
        .unwrap();

    assert_close(&actual, &expected);
}

#[test]
fn non_causal_matches_cpu_reference() {
    run_case(false);
}

#[test]
fn causal_matches_cpu_reference() {
    run_case(true);
}

#[test]
fn rejects_unsupported_head_dimension() {
    let shape = [1, 1, 2, 32];
    let tensor = Tensor::zeros(shape).unwrap();
    let error = flash_attention(&tensor, &tensor, &tensor, false).unwrap_err();
    assert!(error.to_string().contains("head_dim=64"));
}

#[test]
fn runs_on_non_default_stream() {
    let shape = [1, 1, 5, 64];
    let len = shape.into_iter().product();
    let q_host = input(len, 19, 0.01, -9);
    let k_host = input(len, 17, 0.015, -8);
    let v_host = input(len, 13, 0.02, -6);
    let expected = reference(&q_host, &k_host, &v_host, shape, false);

    let q = Tensor::from_slice(&q_host, shape).unwrap();
    let k = Tensor::from_slice(&k_host, shape).unwrap();
    let v = Tensor::from_slice(&v_host, shape).unwrap();
    let stream = CudaStream::new().unwrap();
    let output = flash_attention_on_stream(&q, &k, &v, false, &stream).unwrap();
    stream.synchronize().unwrap();

    assert_close(&output.to_vec().unwrap(), &expected);
}
