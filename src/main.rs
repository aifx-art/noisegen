use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::DTYPE;
use image::{ImageBuffer, Rgb};

use noisegen::{
    create_rgb_image, create_rgb_image_from_1d, gpu_student_t_noise, image_to_tensor, standardize,
    student_t_noise, unit_vec_to_char,
};
use rand::Rng;
use rand_distr::{Distribution, StudentT};

fn main() -> anyhow::Result<()> {
    let stdev = 1.0;
    let mean = -0.0;
    let width = 1024;
    let height = 1024;

    //student
    let degrees_of_freedom = 8.0;
    let noise = student_t_noise(width, height, 3, degrees_of_freedom, 420);
    let noise = standardize(&noise, stdev, mean);
    // let channels = unit_vec_to_char(&noise);
    let img = Tensor::from_vec(noise.clone(), (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F16)?
        // .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    let meani = img.mean_all()?;
    println!("student tensors noise: {:?}", meani);

    let filename = format!("student-{:}.png", degrees_of_freedom);
    create_rgb_image_from_1d(&noise, width, height, 3, &filename);

    //
    //gpu student
    //
    let latent_image = Tensor::zeros((height, width, 3), DType::F64, &Device::Cpu)?;
    let noise = gpu_student_t_noise(degrees_of_freedom, &latent_image, mean, stdev)?;
    //   let noise = standardize(&noise, stdev,mean);
    // let channels = unit_vec_to_char(&noise);
    //let img = Tensor::from_vec(noise.clone(), (height, width,3), &Device::Cpu)?
    let img = noise
        .permute((2, 0, 1))?
        .to_dtype(DType::F16)?
        // .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    println!("gpu tensor img shape {:?}", img);
    let meani = img.mean_all()?;
    println!("gpu tensors noise mean: {:?}", meani);
    let data = noise.to_vec3::<f64>()?;    
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
    let filename = format!("student-gpu-{:}.png", degrees_of_freedom);
    create_rgb_image(&flattened_data, width.try_into().unwrap(), height.try_into().unwrap(),  &filename);

    //let image_path = "test.png";
    // let dtype = DType::F16;
    // let image = image_to_tensor(image_path, dtype)?.to_device(&candle_core::Device::Cpu)?;
    // println!("{:?}", image);



    // some strange gradient
    let mut img = ImageBuffer::from_fn(width as u32, height as u32, |x, y| {
        Rgb([x as u8, y as u8, 128])
    });

    let mut rng = rand::thread_rng();
    for pixel in img.pixels_mut() {
        for (i, channel) in pixel.0.iter_mut().enumerate() {
            let noise: i16 = rng.gen_range(-10..10); // Random noise
            *channel = (*channel as i16 + noise).clamp(0, 255) as u8;
        }
    }

    img.save("gradient.png").expect("Failed to save image");

    //

    Ok(())
}
