//! Misc utilities
use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use crate::{
    assign_symbols,
    containers::{Graph, Values},
    dtype, fac,
    linalg::{Matrix3, Matrix6, Vector3},
    noise::GaussianNoise,
    residuals::{BetweenResidual, PriorResidual},
    variables::*,
};

assign_symbols!(X: SE2, SE3);

/// Load a g2o file
///
/// Currently supports only SE2 and SE3 pose graphs. Will autodetect which one
/// it is, so mixed graph type isn't allowed.
pub fn load_g20(file: &str) -> (Graph, Values) {
    _load_g20_impl(file, false)
}

pub fn _load_g20_impl(file_name: &str, is_sqrt_inf: bool) -> (Graph, Values) {
    let file = File::open(file_name).expect("File not found!");

    let mut values: Values = Values::new();
    let mut graph = Graph::new();

    for line in BufReader::new(file).lines() {
        let line = line.expect("Missing line");
        let parts = line.split_whitespace().collect::<Vec<&str>>();
        match parts[0] {
            "VERTEX_SE2" => {
                let id = parts[1].parse::<u32>().expect("Failed to parse g20");
                let x = parts[2].parse::<dtype>().expect("Failed to parse g20");
                let y = parts[3].parse::<dtype>().expect("Failed to parse g20");
                let theta = parts[4].parse::<dtype>().expect("Failed to parse g20");

                let var = SE2::new(theta, x, y);
                let key = X(id);

                // Add prior on whatever the first variable is
                if values.len() == 1 {
                    let factor = fac![PriorResidual::new(var.clone()), key, 1e-6 as cov];
                    graph.add_factor(factor);
                }

                values.insert(key, var);
            }

            "EDGE_SE2" => {
                let id_prev = parts[1].parse::<u32>().expect("Failed to parse g20");
                let id_curr = parts[2].parse::<u32>().expect("Failed to parse g20");
                let x = parts[3].parse::<dtype>().expect("Failed to parse g20");
                let y = parts[4].parse::<dtype>().expect("Failed to parse g20");
                let theta = parts[5].parse::<dtype>().expect("Failed to parse g20");

                let m11 = parts[6].parse::<dtype>().expect("Failed to parse g20");
                let m12 = parts[7].parse::<dtype>().expect("Failed to parse g20");
                let m13 = parts[8].parse::<dtype>().expect("Failed to parse g20");
                let m22 = parts[9].parse::<dtype>().expect("Failed to parse g20");
                let m23 = parts[10].parse::<dtype>().expect("Failed to parse g20");
                let m33 = parts[11].parse::<dtype>().expect("Failed to parse g20");
                // Note have to permute here - g2o stores with translation first, factrs with
                // rotation first

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let var = SE2::new(theta, x, y);

                // If Cholesky fails (non-PD), the file likely stores
                // upper-triangular sqrt information factors instead. Build the
                // info matrix as U^T * U, permute, and decompose.
                let noise = if is_sqrt_inf {
                    #[rustfmt::skip]
                    let sqrt_inf = Matrix3::new(
                        m33, m13, m23,
                        0.0, m11, m12,
                        0.0, 0.0, m22,
                    );
                    GaussianNoise::from_matrix_sqrt_inf(sqrt_inf)
                } else {
                    #[rustfmt::skip]
                    let inf = Matrix3::new(
                        m33, m13, m23,
                        m13, m11, m12,
                        m23, m12, m22,
                    );
                    match GaussianNoise::from_matrix_inf(inf.as_view()) {
                        Some(n) => n,
                        None => {
                            log::warn!(
                                "Information matrix is not positive definite for factor {}. \
                                     Interpreting all factors as upper-triangular sqrt \
                                     information factors.",
                                graph.len()
                            );
                            // Restart loading with sqrt_inf interpretation
                            return _load_g20_impl(file_name, true);
                        }
                    }
                };

                let factor = fac![BetweenResidual::new(var), (key1, key2), noise];
                graph.add_factor(factor);
            }

            "VERTEX_SE3:QUAT" => {
                let id = parts[1].parse::<u32>().expect("Failed to parse g20");
                let x = parts[2].parse::<dtype>().expect("Failed to parse g20");
                let y = parts[3].parse::<dtype>().expect("Failed to parse g20");
                let z = parts[4].parse::<dtype>().expect("Failed to parse g20");
                let qx = parts[5].parse::<dtype>().expect("Failed to parse g20");
                let qy = parts[6].parse::<dtype>().expect("Failed to parse g20");
                let qz = parts[7].parse::<dtype>().expect("Failed to parse g20");
                let qw = parts[8].parse::<dtype>().expect("Failed to parse g20");

                let rot = SO3::from_xyzw(qx, qy, qz, qw);
                let xyz = Vector3::new(x, y, z);
                let var = SE3::from_rot_trans(rot, xyz);
                let key = X(id);

                // Add prior on whatever the first variable is
                if values.is_empty() {
                    let noise =
                        GaussianNoise::<6>::from_diag_covs(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4);
                    let factor = fac![PriorResidual::new(var.clone()), key, noise];
                    graph.add_factor(factor);
                }

                values.insert(key, var);
            }

            "EDGE_SE3:QUAT" => {
                let id_prev = parts[1].parse::<u32>().expect("Failed to parse g20");
                let id_curr = parts[2].parse::<u32>().expect("Failed to parse g20");
                let x = parts[3].parse::<dtype>().expect("Failed to parse g20");
                let y = parts[4].parse::<dtype>().expect("Failed to parse g20");
                let z = parts[5].parse::<dtype>().expect("Failed to parse g20");
                let qx = parts[6].parse::<dtype>().expect("Failed to parse g20");
                let qy = parts[7].parse::<dtype>().expect("Failed to parse g20");
                let qz = parts[8].parse::<dtype>().expect("Failed to parse g20");
                let qw = parts[9].parse::<dtype>().expect("Failed to parse g20");

                let m11 = parts[10].parse::<dtype>().expect("Failed to parse g20");
                let m12 = parts[11].parse::<dtype>().expect("Failed to parse g20");
                let m13 = parts[12].parse::<dtype>().expect("Failed to parse g20");
                let m14 = parts[13].parse::<dtype>().expect("Failed to parse g20");
                let m15 = parts[14].parse::<dtype>().expect("Failed to parse g20");
                let m16 = parts[15].parse::<dtype>().expect("Failed to parse g20");
                let m22 = parts[16].parse::<dtype>().expect("Failed to parse g20");
                let m23 = parts[17].parse::<dtype>().expect("Failed to parse g20");
                let m24 = parts[18].parse::<dtype>().expect("Failed to parse g20");
                let m25 = parts[19].parse::<dtype>().expect("Failed to parse g20");
                let m26 = parts[20].parse::<dtype>().expect("Failed to parse g20");
                let m33 = parts[21].parse::<dtype>().expect("Failed to parse g20");
                let m34 = parts[22].parse::<dtype>().expect("Failed to parse g20");
                let m35 = parts[23].parse::<dtype>().expect("Failed to parse g20");
                let m36 = parts[24].parse::<dtype>().expect("Failed to parse g20");
                let m44 = parts[25].parse::<dtype>().expect("Failed to parse g20");
                let m45 = parts[26].parse::<dtype>().expect("Failed to parse g20");
                let m46 = parts[27].parse::<dtype>().expect("Failed to parse g20");
                let m55 = parts[28].parse::<dtype>().expect("Failed to parse g20");
                let m56 = parts[29].parse::<dtype>().expect("Failed to parse g20");
                let m66 = parts[30].parse::<dtype>().expect("Failed to parse g20");

                // If Cholesky fails (non-PD), the file likely stores
                // upper-triangular sqrt information factors instead. Build the
                // info matrix as U^T * U, permute, and decompose.
                #[rustfmt::skip]
                    let inf = Matrix6::new(
                        m44, m45, m46, m14, m24, m34,
                        m45, m55, m56, m15, m25, m35,
                        m46, m56, m66, m16, m26, m36,
                        m14, m15, m16, m11, m12, m13,
                        m24, m25, m26, m12, m22, m23,
                        m34, m35, m36, m13, m23, m33,
                    );
                let noise = if is_sqrt_inf {
                    #[rustfmt::skip]
                        let sqrt_inf = Matrix6::new(
                            m44, m45, m46, m14, m24, m34,
                            0.0, m55, m56, m15, m25, m35,
                            0.0, 0.0, m66, m16, m26, m36,
                            0.0, 0.0, 0.0, m11, m12, m13,
                            0.0, 0.0, 0.0, 0.0, m22, m23,
                            0.0, 0.0, 0.0, 0.0, 0.0, m33,
                        );
                    GaussianNoise::from_matrix_sqrt_inf(sqrt_inf)
                } else {
                    match GaussianNoise::from_matrix_inf(inf.as_view()) {
                        Some(n) => n,
                        None => {
                            log::warn!(
                                "Information matrix is not positive definite for factor {}. \
                                     Interpreting all factors as upper-triangular sqrt \
                                     information factors.",
                                graph.len()
                            );
                            return _load_g20_impl(file_name, true);
                        }
                    }
                };

                let rot = SO3::from_xyzw(qx, qy, qz, qw);
                let xyz = Vector3::new(x, y, z);
                let var = SE3::from_rot_trans(rot, xyz);

                let key1 = X(id_prev);
                let key2 = X(id_curr);
                let factor = fac![BetweenResidual::new(var), (key1, key2), noise];
                graph.add_factor(factor);
            }

            _ => {
                println!(",Unknown line: {}", parts.join(" "));
            }
        }
    }

    (graph, values)
}
