use image::io::Reader;
use image::{DynamicImage, GrayImage};
use std::f64;

fn get_gaussian_kernel(sigma: f64) -> [[f64; 5]; 5] {
    // calculating coefficient 1/(2 * sigma² * PI)
    let normal: f64 = 1.0 / (sigma.exp2() * f64::consts::PI * 2.0);

    // generating kernel with size 5
    let mut kernel: [[f64; 5]; 5] = [[0.0; 5]; 5];
    let mut sum: f64 = 0.0;
    for i in 0..=5 {
        for j in 0..=5 {
            kernel[i + 2][j + 2] = normal
                * (-(((i as f64) - 2.0).exp2() + ((i as f64) - 2.0).exp2()) / 2.0 * sigma.exp2());
            sum += kernel[i][j]
        }
    }

    for row in kernel {
        for mut col in row {
            col /= sum;
        }
    }

    kernel
}

fn apply_gaussian_blur(pict: &GrayImage, kernel: &[[f64; 5]; 5]) {
    let x_length = pict.dimensions().0;
    let y_length = pict.dimensions().1;

    for i in 1..x_length - 1 {
        for j in 1..y_length - 1 {
            //pict[i][j] =
        }
    }
}

fn main() {
    let mut img =
        Reader::open("/home/philipp/IdeaProjects/FilmScaningProject/canny_edges/Photo-1.jpeg")
            .unwrap()
            .decode()
            .expect("Reading Image failed!");

    let img = img.as_luma8();

    let kernel = get_gaussian_kernel(1.0);

    apply_gaussian_blur(&img, &kernel)
}
