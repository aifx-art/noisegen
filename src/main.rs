use candle_core::{DType, Device, Tensor};
use candle_transformers::models::bert::DTYPE;
use image::{ImageBuffer, Rgb};

use noisegen::{create_rgb_image_from_1d, image_to_tensor, standardize, student_noise, unit_vec_to_char};
use rand_distr::{StudentT, Distribution};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    
    let width = 100;
    let height = 100;
    let degrees_of_freedom = 11.0;
     let noise = student_noise(width, height, 3, degrees_of_freedom);
     let noise = standardize(&noise);
    // let channels = unit_vec_to_char(&noise);
     let img = Tensor::from_vec(noise.clone(), (height, width,3), &Device::Cpu)?
     .permute((2, 0, 1))?
    .to_dtype(DType::F16)?
    .affine(2. / 255., -1.)?
    .unsqueeze(0)?;
    let meani = img.mean_all()?;
    println!("tensors noise: {:?}",meani);

    create_rgb_image_from_1d(&noise, width, height, 3,"output.png");

    //let image_path = "test.png";
    // let dtype = DType::F16;
    // let image = image_to_tensor(image_path, dtype)?.to_device(&candle_core::Device::Cpu)?;
    // println!("{:?}", image);

    // let mut img = ImageBuffer::from_fn(width, height, |x, y| {
    //     Rgb([x as u8, y as u8, 128])
    // });

    // let mut rng = rand::thread_rng();
    // for pixel in img.pixels_mut() {
    //     for (i,channel) in pixel.0.iter_mut().enumerate() {
    //         //let noise: i16 = rng.gen_range(-10..10); // Random noise
    //         //*channel = (*channel as i16 + noise).clamp(0, 255) as u8;
    //         //let noise =  t.sample(&mut rand::thread_rng());
    //         let n = (v[i] * 255.0).round() as u8;
    //         *channel = n;
    //     }
    // }

    // img.save("output.png").expect("Failed to save image");

    //

    Ok(())
}

