use candle_core::{DType, Device, Tensor};
use image::{ImageBuffer, Rgb};

use rand::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use rand_distr::{Distribution, StudentT};

pub fn image_to_tensor<T: AsRef<std::path::Path>>(path: T, dtype: DType) -> anyhow::Result<Tensor> {
    let img = image::io::Reader::open(path)?.decode()?;
    let (height, width) = (img.height() as usize, img.width() as usize);
    //    let height = height - height % 32;
    //let width = width - width % 32;
    /* let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::CatmullRom,
    ); */
    let img = img.to_rgb8();
    let img = img.into_raw();
   // println!("image {:?}", img);
    //let dtype = if use_f16 { DType::F16 } else { DType::F32 };
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(dtype)?
        .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    Ok(img)
}

pub fn student_noise(width: usize, height: usize, channels: usize, degrees_of_freedom: f64, seed: u64) -> Vec<f64> {
    let num_pixels = width * height * channels; // 3 values per pixel (R, G, B)
    let student = StudentT::new(degrees_of_freedom).unwrap();
    //let mut rng = thread_rng();
    let mut rng = StdRng::seed_from_u64(seed);
    

    // Generate raw noise values using the Student's t-distribution
    let noise_values: Vec<f64> = (0..num_pixels).map(|_| student.sample(&mut rng)).collect();

    //let noise_values = standardize(&noise_values);
    //let noise_values = normalize_to_range(&noise_values, 0., 1.);
    //normalize_minmax(&v);
    // println!("{:?}", noise_values);

    noise_values
}



pub fn gpu_student_noise(
    //size: usize,
    df: f64,
    latent_image: &Tensor,
    mean: f64,
    stdev: f64,
    //seed: Option<u64>,
) -> anyhow::Result<Tensor, anyhow::Error> {
    // Generate two uniform random tensors between 0 and 1
    let dtype = latent_image.dtype();
    let device = latent_image.device();
    let shape = latent_image.shape();

    // gaussian is not correct here
    // let u1 = latent_image.rand_like(mean, stdev)?; 
    //let u2 = latent_image.rand_like(mean, stdev)?;

    //unform is correctf or the box-muller
    let u1 = Tensor::rand(mean, stdev, shape, &device)?;
    let u2 = Tensor::rand(mean, stdev, shape, &device)?;

    // Box-Muller transform to get normal distribution
    let two_pi = 2.0 * std::f64::consts::PI;
    let r = (-2.0 * u1.log()?)?.sqrt()?;
    let theta = u2.mul(&Tensor::full(two_pi, shape, device)?)?;
    let z1 = r.mul(&theta.cos()?)?;

    // Generate chi-square distribution with df degrees of freedom
    let mut chi_square = latent_image.zeros_like()?;
    for _ in 0..df as i32 {
        //let normal = latent_image.rand_like(mean, stdev)?;
        let normal = Tensor::rand(mean, stdev, shape, &device)?;
        chi_square = chi_square.add(&normal.sqr()?)?;
    }

    // Convert to Student's t-distribution
    let df_tensor = Tensor::full(df, shape, device)?;
    let student_t = z1.mul(&(df_tensor.div(&chi_square)?.sqrt().unwrap()))?;
    //println!("student noise on gpu {:?}", student_t);
    Ok(student_t)
}



pub fn create_rgb_image_from_1d(
    normalized_data: &[f64],
    width: usize,
    height: usize,
    channels: usize,
    output_path: &str,
) {
    // The total number of elements must match width * height * 3 (for R, G, B)
    assert_eq!(
        normalized_data.len(),
        (width * height * channels) as usize,
        "Data size must match image dimensions"
    );

    let img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        let index = ((y as usize * width + x as usize) * 3) as usize; // Calculate the starting index for (R, G, B)
        let r = (normalized_data[index] * 255.0).round() as u8;
        let g = (normalized_data[index + 1] * 255.0).round() as u8;
        let b = (normalized_data[index + 2] * 255.0).round() as u8;

        Rgb([r, g, b])
    });

    img.save(output_path).expect("Failed to save image");
}

pub fn create_rgb_image(
    normalized_data: &[(f64, f64, f64)],
    width: u32,
    height: u32,
    output_path: &str,
) {
    assert_eq!(
        normalized_data.len(),
        (width * height) as usize,
        "Data size must match image dimensions"
    );

    let img = ImageBuffer::from_fn(width, height, |x, y| {
        let index = (y * width + x) as usize;
        let (r, g, b) = normalized_data[index];
        Rgb([
            (r * 255.0).round() as u8,
            (g * 255.0).round() as u8,
            (b * 255.0).round() as u8,
        ])
    });

    img.save(output_path).expect("Failed to save image");
}

pub fn unit_vec_to_char(data: &[f64]) -> Vec<u8> {
    let mut channels = vec![];
    for d in data {
        channels.push((*d * 255.) as u8);
    }
    channels
}

pub fn normalize_to_range(data: &[f64], min: f64, max: f64) -> Vec<f64> {
    let data_min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let data_max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!("{data_min} and max {data_max}");

    data.iter()
        .map(|&x| min + (x - data_min) / (data_max - data_min) * (max - min))
        .collect()
}

pub fn normalize_minmax(values: &[f64]) -> Option<Vec<f64>> {
    if values.is_empty() {
        return None;
    }

    let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    // Check if all values are the same (max == min)
    if (max - min).abs() < f64::EPSILON {
        return Some(vec![0.5; values.len()]); // Return all 0.5 if no variance
    }

    Some(values.iter().map(|&x| (x - min) / (max - min)).collect())
}

pub fn standardize(data: &[f64], target_std:f64, target_mean: f64) -> Vec<f64> {
    let mean = data.iter().copied().sum::<f64>() / data.len() as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;
    let std_dev = variance.sqrt();

    let standardized_data: Vec<f64> = data.iter().map(|&x| (x - mean) / std_dev).collect();

    // Transform to target mean and standard deviation
    standardized_data
        .iter()
        .map(|&x| x * target_std + target_mean)
        .collect()
}
