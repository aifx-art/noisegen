use candle_core::{DType, Device, IndexOp, Tensor};
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

pub fn student_t_noise(
    width: usize,
    height: usize,
    channels: usize,
    degrees_of_freedom: f64,
    seed: u64,
) -> Vec<f64> {
    let num_pixels = width * height * channels; 
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

pub fn gpu_student_t_noise(
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

    // Generate standard normal (Box-Muller transform)
    let u1 = Tensor::rand(0.0, 1.0, shape, &device)?;
    let u2 = Tensor::rand(0.0, 1.0, shape, &device)?;
    let two_pi = 2.0 * std::f64::consts::PI;
    let r = (-2.0 * u1.log()?)?.sqrt()?;
    let theta = u2.mul(&Tensor::full(two_pi, shape, device)?)?;
    let normal = r.mul(&theta.cos()?)?;

    // Generate chi-squared distribution (df degrees of freedom)
    let mut chi_square = latent_image.zeros_like()?;
    for _ in 0..df as i32 {
        let u1 = Tensor::rand(0.0, 1.0, shape, &device)?;
        let u2 = Tensor::rand(0.0, 1.0, shape, &device)?;

        let r = (-2.0 * u1.log()?)?.sqrt()?;
        let theta = u2.mul(&Tensor::full(two_pi, shape, device)?)?;
        let normal = r.mul(&theta.cos()?)?;
        chi_square = chi_square.add(&normal.sqr()?)?;
    }

    // Divide the standard normal by the square root of the chi-squared distribution
    let df_tensor = Tensor::full(df, shape, device)?;
    let student_t = normal.mul(&(df_tensor.div(&chi_square)?.sqrt().unwrap()))?;

    // Scale by stdev and mean
    let scaled_t = student_t
        .mul(&Tensor::full(stdev, shape, device)?)?
        .add(&Tensor::full(mean, shape, device)?)?;

    // Normalize to desired range [-1, 1]
    //let normal_noise = normalize_tensor(&scaled_t, -1.0, 1.0)?;
    //let scaled_t = standardize_tensor(&scaled_t)?;
    Ok(scaled_t)

    /* let device = latent_image.device();
    let shape = latent_image.shape();

    //unform is correctf or the box-muller
    //let u1 = Tensor::rand(mean, stdev, shape, &device)?;
    //let u2 = Tensor::rand(mean, stdev, shape, &device)?;

    let u1 = Tensor::rand(0.0, 1.0, shape, &device)?;
    let u2 = Tensor::rand(0.0, 1.0, shape, &device)?;
    // Box-Muller transform to get normal distribution
    let two_pi = 2.0 * std::f64::consts::PI;
    let r = (-2.0 * u1.log()?)?.sqrt()?;
    let theta = u2.mul(&Tensor::full(two_pi, shape, device)?)?;
    let z1 = r.mul(&theta.cos()?)?;

    // Generate chi-square distribution with df degrees of freedom
    let mut chi_square = latent_image.zeros_like()?;
    for _ in 0..df as i32 {
        //let normal = latent_image.rand_like(mean, stdev)?;
        //uniform
        //let normal = Tensor::rand(mean, stdev, shape, &device)?;
        //chi_square = chi_square.add(&normal.sqr()?)?;
        // Generate standard normal using Box-Muller again for consistency
        let u1 = Tensor::rand(0.0, 1.0, shape, &device)?;
        let u2 = Tensor::rand(0.0, 1.0, shape, &device)?;

        let r = (-2.0 * u1.log()?)?.sqrt()?;
        let theta = u2.mul(&Tensor::full(two_pi, shape, device)?)?;
        let normal = r.mul(&theta.cos()?)?;

        chi_square = chi_square.add(&normal.sqr()?)?;
    }

    // Convert to Student's t-distribution
    let df_tensor = Tensor::full(df, shape, device)?;
    let student_t = z1.mul(&(df_tensor.div(&chi_square)?.sqrt().unwrap()))?;

    // Student's t-distribution (standardized)
    //let student_t = z1.div(&(chi_square.div(&df_tensor)?.sqrt()?))?;

    //println!("student noise on gpu {:?}", student_t);
    let scaled_t = student_t
        .mul(&Tensor::full(stdev, shape, device)?)?
        .add(&Tensor::full(mean, shape, device)?)?;

    let normal_noise = normalize_tensor(&scaled_t, -1.0, 1.0)?;
    Ok(normal_noise) */
}

pub fn flatten_tensor_rgb(image: &Tensor) -> anyhow::Result<Vec<(f64, f64, f64)>> {
    let data = image.to_vec3::<f64>()?;
    let mut flattened_data = Vec::new();
    for row in data {
        for pixel in row {
            if let [r, g, b] = pixel.as_slice() {
                flattened_data.push((*r, *g, *b));
            } else {
                anyhow::bail!("Each pixel must have exactly 3 values (RGB).");
            }
        }
    }
    Ok(flattened_data)
}

pub fn flatten_tensor(image: &Tensor) -> anyhow::Result<Vec<(f64, f64, f64)>> {
    let data = image.to_vec3::<f64>()?;
    let mut flattened_data = Vec::new();

    // Iterate over the spatial dimensions.
    let height = data[0].len();
    let width = data[0][0].len();

    for row in 0..height {
        for col in 0..width {}
    }
    for row in data {
        for pixel in row {
            if let [r, g, b] = pixel.as_slice() {
                flattened_data.push((*r, *g, *b));
            } else {
                anyhow::bail!("Each pixel must have exactly 3 values (RGB).");
            }
        }
    }
    Ok(flattened_data)
}

/* pub fn adjust_contrast(latent_image: &Tensor, amount: f64) -> anyhow::Result<Tensor> {
    println!(
        "adjust contrast on latent image {:?} amount {:?}",
        latent_image, amount
    );
    // Get the shape of the input tensor
    let (c, h, w) = latent_image.dims3()?;

    let mut channels = Vec::with_capacity(c);

    let batch_tensor_image = latent_image; //.i(0)?;
                                           //let batch_tensor_noise = noise.i(0)?;

    for i in 0..c {
        // Slice out the `i`th channel, shape: [H, W]
        let channel_image = batch_tensor_image.i(i)?;
        println!("channel image {:?}", channel_image);
        let mut adjusted_channel = channel_image.zeros_like()?;
        for y in 0..h {
            for x in 0..w {
                // Get value at (x,y) coordinate
                let value = channel_image.i(y)?.i(x)?;

                // Apply your contrast adjustment here
                let adjusted_value = (value * amount)?; // Example adjustment
                println!("x:{value},y:{adjusted_value}");
                // Set the new value
                //adjusted_channel.i(y)?.i(x)?. copy(&adjusted_value)?;
                //adjusted_channel.i(y)?.i(x)?.set(adjusted_value)?;
            }
        }

        channels.push(adjusted_channel);
    }

    let mut new_image = Tensor::stack(&channels, 0)?;
    println!("new image {:?}", new_image);
    // new_image = new_image.unsqueeze(0)?;
    //println!("unsquuezzed new image {:?}", new_image);
    Ok(new_image)
}
 */

pub fn adjust_contrast(latent_image: &Tensor, amount: f64) -> anyhow::Result<Tensor> {
    let (_, c, h, w) = latent_image.dims4()?;
    let mut channels = Vec::with_capacity(c);

    for i in 0..c {
        let channel_image = latent_image.i(0)?.i(i)?; // Get the i-th channel
        println!(
            "channel max: {:?} min: {:?}",
            channel_image.max_all(),
            channel_image.min_all()
        );
        let mut data = Vec::with_capacity((h * w) as usize);

        // Extract values, apply adjustments, and collect into a Vec
        for y in 0..h {
            for x in 0..w {
                let value: f64 = channel_image.i(y)?.i(x)?.to_scalar()?;
                //  println!("{value}");
                data.push(value * amount);
            }
        }

        // Create a new tensor from the adjusted values
        let  adjusted_channel = Tensor::from_vec(data, (h, w), &latent_image.device())?;
        println!(
            "adjustd max: {:?} min: {:?}",
            adjusted_channel.max_all(),
            adjusted_channel.min_all()
        );

        channels.push(adjusted_channel);
    }

    // Stack the adjusted channels back together into a single tensor
    let mut new_image = Tensor::stack(&channels, 0)?;
    new_image = new_image.unsqueeze(0)?;

    Ok(new_image)
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

pub fn standardize(data: &[f64], target_std: f64, target_mean: f64) -> Vec<f64> {
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

pub fn standardize_tensor(tensor: &Tensor) -> anyhow::Result<Tensor, anyhow::Error> {
    //println!("standardize tensor {:?}", tensor);
    // Get the shape of the tensor
    let (_, c, h, w) = tensor.dims4()?;

    // Compute the mean and std deviation along H and W for each channel
    //let mean = tensor.mean(1)?.mean(1)?; // Shape: (C,)
    //let std = tensor.std(1)?.std_dim(1)?;   // Shape: (C,)

    let mean_f = tensor.mean_all()?.to_scalar::<f64>()?;

    let mean_tensor = Tensor::full(mean_f, tensor.shape(), tensor.device())?;

    let stdev = std_all(&tensor, &mean_tensor)?.to_scalar::<f64>()?;
    println!("input stdev {:?} mean {:?}", stdev, mean_f);
    let stdev_tensor = Tensor::full(stdev, tensor.shape(), tensor.device())?;
    ///println!("stdev {:?} mean {:?}", stdev_tensor, mean_tensor);
    // Standardize the tensor
    let standardized_tensor = tensor.sub(&mean_tensor)?.div(&stdev_tensor)?;
    let mean_f = standardized_tensor.mean_all()?.to_scalar::<f64>()?;
    let mean_tensor = Tensor::full(mean_f, tensor.shape(), tensor.device())?;
    let stdev = std_all(&standardized_tensor, &mean_tensor)?.to_scalar::<f64>()?;
    println!("output stdev {:?} mean {:?}", stdev, mean_f);
    //let stdev_tensor = Tensor::full(stdev, tensor.shape(), tensor.device())?;
    //println!("stdev {:?} mean {:?}", stdev_tensor, mean_tensor);

    Ok(standardized_tensor)
}

pub fn std_all(tensor: &Tensor, mean: &Tensor) -> anyhow::Result<Tensor> {
    // this can be used to create  a tensor from a single float
    //let mean_tensor = Tensor::full(mean, tensor.shape(), tensor.device())?;

    let diff = (tensor - mean)?; // Difference from mean
    let squared_diff = diff.sqr()?; // Square the difference
    let variance = squared_diff.mean_all()?; // Mean of squared differences
    Ok(variance.sqrt()?) // Standard deviation is the square root of variance
}

pub fn normalize_tensor(tensor: &Tensor, min: f64, max: f64) -> anyhow::Result<Tensor> {
    println!("normalize tensor {:?}", tensor);
    // Compute the mean and standard deviation of the tensor
    let min_val = tensor.min_all()?.to_scalar::<f64>()?;

    let max_val = tensor.max_all()?.to_scalar::<f64>()?;
    println!("min {:?}, max {:?}", min_val, max_val);
    let min_val_tensor = Tensor::full(min_val, tensor.shape(), tensor.device())?;
    let max_val_tensor = Tensor::full(max_val, tensor.shape(), tensor.device())?;

    // Scale to [0, 1] by subtracting min and dividing by (max - min)
    let zero_to_one = tensor
        .sub(&min_val_tensor)?
        .div(&(max_val_tensor.sub(&min_val_tensor))?)?;

    // Scale to [min_range, max_range]
    let scaled_tensor = zero_to_one
        .mul(&Tensor::full(max - min, tensor.shape(), tensor.device())?)?
        .add(&Tensor::full(min, tensor.shape(), tensor.device())?)?;

    Ok(scaled_tensor)
}

/*
println!("tensor {:?}", tensor);

let mean_tensor = Tensor::full(mean, tensor.shape(), tensor.device())?; //.to_dtype(DType::F16)?;
println!("mean_tensor {:?}", mean_tensor);

let std_tensor = Tensor::full(stdev, tensor.shape(), tensor.device())?; //.to_dtype(DType::F16)?;
println!("std_tensor {:?}", std_tensor);
// Normalize the tensor: (tensor - mean) / std
Ok(((tensor - mean_tensor) / std_tensor)?) */
