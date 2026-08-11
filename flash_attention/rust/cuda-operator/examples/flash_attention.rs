use cuda_operator::{Result, Tensor, flash_attention};

fn main() -> Result<()> {
    let shape = [1, 1, 4, 64];
    let len = shape.into_iter().product();
    let q_host: Vec<f32> = (0..len).map(|index| (index % 17) as f32 * 0.01).collect();
    let k_host: Vec<f32> = (0..len).map(|index| (index % 13) as f32 * 0.02).collect();
    let v_host: Vec<f32> = (0..len).map(|index| (index % 11) as f32 * 0.03).collect();

    let q = Tensor::from_slice(&q_host, shape)?;
    let k = Tensor::from_slice(&k_host, shape)?;
    let v = Tensor::from_slice(&v_host, shape)?;

    let output = flash_attention(&q, &k, &v, true)?;
    println!("output shape: {:?}", output.shape());
    println!("first row: {:?}", &output.to_vec()?[..8]);
    Ok(())
}
