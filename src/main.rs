use image::{ImageBuffer, Rgb};

use noisegen::{create_rgb_image_from_1d, student_noise};
use rand_distr::{StudentT, Distribution};
use rand::Rng;

fn main() -> anyhow::Result<()> {
    

    
    let width = 100;
    let height = 100;
    let degrees_of_freedom = 1.0;
    let noise = student_noise(width, height, degrees_of_freedom);
    
    
    create_rgb_image_from_1d(&noise, width, height, "output.png");

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

