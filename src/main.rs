use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::DTYPE;
use image::{ImageBuffer, Rgb};

use noisegen::{
    adjust_contrast, create_rgb_image, create_rgb_image_from_1d, flatten_tensor, flatten_tensor_rgb, gpu_student_t_noise, image_to_tensor, normalize_tensor, standardize, standardize_tensor, std_all, student_t_noise, unit_vec_to_char
};
use rand::Rng;
use rand_distr::{Distribution, StudentT};

fn main() -> anyhow::Result<()> {
    let stdev = 1.0;
    let mean = -0.0;
    let width = 128;
    let height = 128;
    let degrees_of_freedom = 11.0;

    //StudentT CPU    
    let noise = student_t_noise(width, height, 3, degrees_of_freedom, 420);
    let noise = standardize(&noise, stdev, mean);
    // let channels = unit_vec_to_char(&noise);
    let img = Tensor::from_vec(noise.clone(), (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F16)?
        // .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    let meani = img.mean_all()?;
    println!("studentT tensor noise mean: {:?}", meani);

    let filename = format!("student-{:}.png", degrees_of_freedom);
    create_rgb_image_from_1d(&noise, width, height, 3, &filename);

    //
    //gpu student
    //
    let latent_image_rgb = Tensor::zeros((height, width,3), DType::F64, &Device::Cpu)?;
    let noise = gpu_student_t_noise(degrees_of_freedom, &latent_image_rgb, mean, stdev)?;
    println!("gpu noise tensor {:?}", noise);   
    
    let img = noise
        .permute((2, 0, 1))?
        .to_dtype(DType::F16)?
        // .affine(2. / 255., -1.)?
        .unsqueeze(0)?;
    println!("gpu tensor img shape {:?}", img);
    let meani = img.mean_all()?;
    println!("gpu tensors noise mean: {:?}", meani);

    let flattened_image = flatten_tensor_rgb(&noise)?;    
    let filename = format!("student-gpu-{:}.png", degrees_of_freedom);
    create_rgb_image(&flattened_image, width.try_into().unwrap(), height.try_into().unwrap(),  &filename);

    let reshaped_noise = noise.permute((2,0,1))?;
/*     let mean_f = reshaped_noise.mean_all()?.to_scalar::<f64>()?;
    let mean_tensor = Tensor::full(mean_f, reshaped_noise.shape(), reshaped_noise.device())?;
    let stdev = std_all(&reshaped_noise, &mean_tensor)?.to_scalar::<f64>()?;
    println!("stdev {:?} mean {:?}", stdev, mean_f);
    let stdev_tensor = Tensor::full(stdev, reshaped_noise.shape(), reshaped_noise.device())?;
    println!("stdev {:?} mean {:?}", stdev_tensor, mean_tensor); */
  
    let normal_tensor = normalize_tensor(&reshaped_noise,-1., 1.0)?.permute((1,2,0))?; 
    //let std_noise = stadardize_tensor?.permute((1,2,0))?;  
    let flattened_image_normal = flatten_tensor_rgb(&normal_tensor)?;    
    let filename = format!("student-gpu-{:}-nrml.png", degrees_of_freedom);
    create_rgb_image(&flattened_image_normal, width.try_into().unwrap(), height.try_into().unwrap(),  &filename);


    //let latent_image = Tensor::zeros((3, height, width), DType::F64, &Device::Cpu)?;
    //let noise = gpu_student_t_noise(degrees_of_freedom, &latent_image, mean, stdev)?;
    let reshaped_noise = normal_tensor.permute((2,0,1))?;
    println!("reshaped gpu noise tensor {:?}", reshaped_noise);
    let adjusted_noise = adjust_contrast(&reshaped_noise, 4.0)?;
    //let adjusted_normal_tensor = normalize_tensor(&adjusted_noise,-1., 1.0)?;

    let reshaped_ajusted_noise = adjusted_noise.permute((1,2,0))?;
    println!("reshaped gpu adjusted and normalized tensor {:?}", reshaped_ajusted_noise);
    
    let flattened_image = flatten_tensor_rgb(&reshaped_ajusted_noise)?;    
    let filename = format!("student-gpu-{:}-adjusted.png", degrees_of_freedom);
    create_rgb_image(&flattened_image, width.try_into().unwrap(), height.try_into().unwrap(),  &filename);

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
