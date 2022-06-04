use image::io::Reader;
use image::{GrayImage, ImageBuffer, Luma, Pixel};
use std::f64;

fn get_gaussian_kernel(sigma: f64) -> [[f64; 5]; 5] {
    let mut out: [[f64; 5]; 5] = [[0.0; 5]; 5];

    // calculating coefficient 1/(2 * sigma² * PI)
    let normal: f64 = 1.0 / (sigma.exp2() * f64::consts::PI * 2.0);

    // generating kernel with size 5
    let mut kernel_coords: [[(f64, f64); 5]; 5] = [[(0.0, 0.0); 5]; 5];

    for x_kern in 0..kernel_coords.len() {
        for y_kern in 0..kernel_coords.len() {
            kernel_coords[x_kern][y_kern] = ((x_kern - 2) as f64, (y_kern - 2) as f64);
        }
    }

    for i in 0..out.len() {
        for j in 0..out.len() {
            out[i][j] = normal
                * (-(kernel_coords[i][j].0.exp2() + (kernel_coords[i][j].1.exp2())) / 2.0
                    * sigma.exp2())
                .exp();
        }
    }

    out
}

fn apply_gaussian_blur(pict: &GrayImage, kernel: &[[f64; 5]; 5]) -> GrayImage {
    let mut out: ImageBuffer<Luma<u8>, Vec<u8>> = GrayImage::new(pict.width(), pict.height());

    for x in 0..pict.width() {
        for y in 0..pict.height() {
            let mut weighted_avg: f64 = 0.0;
            for x_kern in 0..5 {
                for y_kern in 0..5 {
                    weighted_avg +=
                        pict.get_pixel(x, y).channels()[0] as f64 * kernel[x_kern][y_kern];
                }
            }
            let weighted_avg = weighted_avg / 25.0;
            out.put_pixel(x, y, image::Luma::<u8>([weighted_avg as u8]))
        }
    }
    out
}
fn main() {
    let img = Reader::open("/home/philipp/DEV/FilmScaningProject/canny_edges/Photo-1.jpeg")
        .unwrap()
        .decode()
        .expect("Reading Image failed!");

    let img = img.into_luma8();

    let kernel = get_gaussian_kernel(1.0);

    let blurred_img = apply_gaussian_blur(&img, &kernel);

    blurred_img
        .save("/home/philipp/DEV/FilmScaningProject/canny_edges/Photo-1_blurred.jpeg")
        .expect("failed to save")
}
