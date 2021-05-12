use noise::{Fbm, MultiFractal, NoiseFn, Seedable};
use png::{BitDepth, ColorType, Encoder};
use rand::random;
use rayon::prelude::*;
use std::f32::consts;
use std::fs::File;
use std::io::BufWriter;
use std::mem;
use std::time::Instant;

fn main() {
    let size_x = 512;
    let size_y = 512;

    let mut height_map = vec![0.0; size_x * size_y].into_boxed_slice();

    let fbm = Fbm::new()
        .set_seed(random())
        .set_octaves(8)
        .set_lacunarity(2.0)
        .set_persistence(0.5);

    for x in 0..size_x {
        for y in 0..size_y {
            height_map[x + y * size_x] =
                (fbm.get([x as f64 * 0.003, y as f64 * 0.003]) as f32 + 1.0) / 2.0;
        }
    }

    let start = Instant::now();

    let height_map_after = erode_height_map(
        height_map.clone(),
        size_x,
        size_y,
        512,
        200.0 / size_x.max(size_y) as f32,
        0.0004,
        0.0005,
        0.05,
        30.0,
        0.25,
        0.001,
        50.0,
        0.05,
    );

    let dur = Instant::now().duration_since(start);

    debug_assert!(height_map_after.iter().all(|h| h.is_finite()));

    write_height_map("before.png", &height_map, size_x, size_y);
    write_height_map("after.png", &height_map_after, size_x, size_y);

    println!("Erosion took {:.02}s", dur.as_secs_f32());
}

fn write_height_map(path: &str, hm: &[f32], size_x: usize, size_y: usize) {
    let mut encoder = Encoder::new(
        BufWriter::new(File::create(path).unwrap()),
        size_x as u32,
        size_y as u32,
    );
    encoder.set_color(ColorType::Grayscale);
    encoder.set_depth(BitDepth::Eight);
    let mut writer = encoder.write_header().unwrap();
    let data: Box<[u8]> = hm
        .iter()
        .map(|&f| (f.clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    writer.write_image_data(&data).unwrap();
}

fn erode_height_map(
    height_map: Box<[f32]>,
    size_x: usize,
    size_y: usize,
    iterations: u32,
    cell_size: f32,
    rain_rate: f32,
    evaporation_rate: f32,
    min_height_delta: f32,
    gravity: f32,
    dissolve_rate: f32,
    deposition_rate: f32,
    sediment_cap_coeff: f32,
    repose_slope: f32,
) -> Box<[f32]> {
    debug_assert!(height_map.len() == size_x * size_y);

    let mut terrain = height_map;
    let mut terrain2 = vec![0.0; size_x * size_y].into_boxed_slice();
    let mut water = terrain2.clone();
    let mut water2 = terrain2.clone();
    let mut sediment = terrain2.clone();
    let mut sediment2 = terrain2.clone();
    let mut speed = terrain2.clone();
    let mut gradient_x = terrain2.clone();
    let mut gradient_y = terrain2.clone();

    let get = |a: &[f32], x: isize, y: isize| -> f32 {
        a[x.rem_euclid(size_x as isize) as usize + y.rem_euclid(size_y as isize) as usize * size_x]
    };

    let weight_matrix_size = 5_isize;
    let blur_weight_matrix = {
        let mut mat: Box<[f32]> = (0..weight_matrix_size)
            .flat_map(|y| {
                (0..weight_matrix_size).map(move |x| {
                    gauss_blur_fn(
                        (x - weight_matrix_size / 2) as f32,
                        (y - weight_matrix_size / 2) as f32,
                        1.0,
                    )
                })
            })
            .collect();

        let sum: f32 = mat.iter().sum();
        mat.iter_mut().for_each(|v| *v /= sum);
        mat
    };

    for iter in 0..iterations {
        par2d(
            size_x,
            (
                terrain2.par_iter_mut(),
                water.par_iter_mut(),
                sediment.par_iter_mut(),
                speed.par_iter_mut(),
                gradient_x.par_iter_mut(),
                gradient_y.par_iter_mut(),
            ),
            |(terrain2, water, sediment, speed, gradient_x, gradient_y), x, y| {
                *water += rain_rate * cell_size * cell_size;

                let (grad_x, grad_y) = {
                    let dx = get(&terrain, x - 1, y) - get(&terrain, x + 1, y);
                    let dy = get(&terrain, x, y - 1) - get(&terrain, x, y + 1);
                    let mag = f32::hypot(dx, dy);
                    if mag < 1e-10 {
                        let angle = consts::TAU * random::<f32>();
                        (angle.cos(), angle.sin())
                    } else {
                        (dx / mag, dy / mag)
                    }
                };

                *gradient_x = grad_x;
                *gradient_y = grad_y;

                let old_height = get(&terrain, x, y);
                let new_height = {
                    let x = x as f32 + grad_x;
                    let y = y as f32 + grad_y;
                    let ix = x.floor() as isize;
                    let iy = y.floor() as isize;
                    bilerp(
                        get(&terrain, ix, iy),
                        get(&terrain, ix + 1, iy),
                        get(&terrain, ix, iy + 1),
                        get(&terrain, ix + 1, iy + 1),
                        x - x.floor(),
                        y - y.floor(),
                    )
                };

                let height_delta = old_height - new_height;

                let sediment_cap = height_delta.max(min_height_delta) / cell_size
                    * *speed
                    * *water
                    * sediment_cap_coeff;

                let deposited_sediment = {
                    if height_delta < 0.0 {
                        height_delta.min(*sediment)
                    } else if *sediment > sediment_cap {
                        deposition_rate * (*sediment - sediment_cap)
                    } else {
                        dissolve_rate * (*sediment - sediment_cap)
                    }
                }
                .max(-height_delta);

                *sediment -= deposited_sediment;
                *terrain2 = get(&terrain, x, y) + deposited_sediment;
                *speed = gravity * height_delta / cell_size;
            },
        );

        // Transport water and sediment in the Moore neighborhood using the normalized gradient vectors.
        par2d(
            size_x,
            (sediment2.par_iter_mut(), water2.par_iter_mut()),
            |(sediment2, water2), x, y| {
                let wrd = (-get(&gradient_x, x + 1, y + 1)).max(0.0)
                    * (-get(&gradient_y, x + 1, y + 1)).max(0.0);
                let wrm = (-get(&gradient_x, x + 1, y)).max(0.0)
                    * (1.0 - get(&gradient_y, x + 1, y).abs()).max(0.0);
                let wru = (-get(&gradient_x, x + 1, y - 1)).max(0.0)
                    * get(&gradient_y, x + 1, y - 1).max(0.0);
                let wmd = (1.0 - get(&gradient_x, x, y + 1).abs()).max(0.0)
                    * (-get(&gradient_y, x, y + 1)).max(0.0);
                let wmm = (1.0 - get(&gradient_x, x, y).abs()).max(0.0)
                    * (1.0 - get(&gradient_y, x, y).abs()).max(0.0);
                let wmu = (1.0 - get(&gradient_x, x, y - 1).abs()).max(0.0)
                    * get(&gradient_y, x, y - 1).max(0.0);
                let wld = get(&gradient_x, x - 1, y + 1).max(0.0)
                    * (-get(&gradient_y, x - 1, y + 1)).max(0.0);
                let wlm = get(&gradient_x, x - 1, y).max(0.0)
                    * (1.0 - get(&gradient_y, x - 1, y).abs()).max(0.0);
                let wlu = get(&gradient_x, x - 1, y - 1).max(0.0)
                    * get(&gradient_y, x - 1, y - 1).max(0.0);

                *sediment2 = get(&sediment, x + 1, y + 1) * wrd
                    + get(&sediment, x + 1, y) * wrm
                    + get(&sediment, x + 1, y - 1) * wru
                    + get(&sediment, x, y + 1) * wmd
                    + get(&sediment, x, y) * wmm
                    + get(&sediment, x, y - 1) * wmu
                    + get(&sediment, x - 1, y + 1) * wld
                    + get(&sediment, x - 1, y) * wlm
                    + get(&sediment, x - 1, y - 1) * wlu;

                *water2 = get(&water, x + 1, y + 1) * wrd
                    + get(&water, x + 1, y) * wrm
                    + get(&water, x + 1, y - 1) * wru
                    + get(&water, x, y + 1) * wmd
                    + get(&water, x, y) * wmm
                    + get(&water, x, y - 1) * wmu
                    + get(&water, x - 1, y + 1) * wld
                    + get(&water, x - 1, y) * wlm
                    + get(&water, x - 1, y - 1) * wlu;
            },
        );

        par2d(
            size_x,
            ((&mut terrain).par_iter_mut(), (&mut water2).par_iter_mut()),
            |(terrain, water2), x, y| {
                // Evaporate the water while we're here.
                *water2 *= 1.0 - evaporation_rate;

                let slope = f32::hypot(
                    (get(&terrain2, x + 1, y) - get(&terrain2, x - 1, y)) / 2.0 / cell_size,
                    (get(&terrain2, x, y + 1) - get(&terrain2, x, y - 1)) / 2.0 / cell_size,
                );

                // Smooth out slopes that are too steep using gaussian blur.
                if slope > repose_slope {
                    let mut sum = 0.0_f32;
                    for wx in 0..weight_matrix_size {
                        for wy in 0..weight_matrix_size {
                            sum += blur_weight_matrix[(wx + wy * weight_matrix_size) as usize]
                                * get(
                                    &terrain2,
                                    x + wx - weight_matrix_size / 2,
                                    y + wy - weight_matrix_size / 2,
                                );
                        }
                    }
                    *terrain = sum;
                } else {
                    *terrain = get(&terrain2, x, y);
                }
            },
        );

        mem::swap(&mut water, &mut water2);
        mem::swap(&mut sediment, &mut sediment2);

        println!("{} / {}", iter + 1, iterations);
    }

    terrain
}

fn par2d<T, U, V, F>(size_x: usize, t: T, f: F)
where
    T: IntoParallelIterator<Iter = U>,
    U: IndexedParallelIterator<Item = V>,
    F: Fn(V, isize, isize) + Sync,
{
    t.into_par_iter().enumerate().for_each(|(i, e)| {
        f(e, (i % size_x) as isize, (i / size_x) as isize);
    });
}

fn gauss_blur_fn(x: f32, y: f32, sigma: f32) -> f32 {
    let sigma_sq2 = sigma * sigma * 2.0;
    (std::f32::consts::PI * sigma_sq2).recip() * (-(x * x + y * y) / sigma_sq2).exp()
}

/// Bilinear interpolation
fn bilerp(v00: f32, v10: f32, v01: f32, v11: f32, tx: f32, ty: f32) -> f32 {
    lerp(lerp(v00, v10, tx), lerp(v01, v11, tx), ty)
}

/// Linear interpolation
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    debug_assert!(t >= 0.0 && t <= 1.0);
    a * (1.0 - t) + b * t
}
