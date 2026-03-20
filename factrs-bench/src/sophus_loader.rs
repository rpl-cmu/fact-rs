use std::{
    fs::File,
    io::{BufRead, BufReader},
};

use factrs::{core::SO3, traits::Variable};
use nalgebra::{Vector1, Vector2, Vector3};
use sophus_autodiff::linalg::MatF64;
use sophus_lie::{Isometry2F64, Isometry3F64, Rotation2F64, Rotation3F64};
use sophus_opt::{
    nlls::{
        costs::{Isometry2PriorCostTerm, Isometry3PriorCostTerm, PoseGraph2CostTerm},
        CostFn, CostTerms, IsCostFn,
    },
    variables::{VarBuilder, VarFamilies, VarFamily, VarKind},
};

use crate::sophus::PoseGraph3CostTerm;

pub fn load_g2o_2d(file: &str) -> (Vec<Box<dyn IsCostFn>>, VarFamilies) {
    let file = File::open(file).expect("File not found!");

    let mut values = Vec::new();
    let mut betweens = CostTerms::new(["poses", "poses"], vec![]);

    for line in BufReader::new(file).lines() {
        let line = line.expect("Missing line");
        let parts = line.split(" ").collect::<Vec<&str>>();
        match parts[0] {
            "VERTEX_SE2" => {
                // let id = parts[1].parse::<u32>().expect("Failed to parse g20");
                let x = parts[2].parse::<f64>().expect("Failed to parse g20");
                let y = parts[3].parse::<f64>().expect("Failed to parse g20");
                let theta = parts[4].parse::<f64>().expect("Failed to parse g20");

                let var = Isometry2F64::from_translation_and_rotation(
                    Vector2::new(x, y),
                    Rotation2F64::exp(Vector1::new(theta)),
                );
                values.push(var);
            }

            "EDGE_SE2" => {
                let id_prev = parts[1].parse::<usize>().expect("Failed to parse g20");
                let id_curr = parts[2].parse::<usize>().expect("Failed to parse g20");
                let x = parts[3].parse::<f64>().expect("Failed to parse g20");
                let y = parts[4].parse::<f64>().expect("Failed to parse g20");
                let theta = parts[5].parse::<f64>().expect("Failed to parse g20");

                // TODO: Noise models?
                // let m11 = parts[6].parse::<f64>().expect("Failed to parse g20");
                // let m12 = parts[7].parse::<f64>().expect("Failed to parse g20");
                // let m13 = parts[8].parse::<f64>().expect("Failed to parse g20");
                // let m22 = parts[9].parse::<f64>().expect("Failed to parse g20");
                // let m23 = parts[10].parse::<f64>().expect("Failed to parse g20");
                // let m33 = parts[11].parse::<f64>().expect("Failed to parse g20");

                // let inf = Matrix3::new(
                //     m33, m13, m23,
                //     m13, m11, m12,
                //     m23, m12, m22,
                // );

                let meas = Isometry2F64::from_translation_and_rotation(
                    Vector2::new(x, y),
                    Rotation2F64::exp(Vector1::new(theta)),
                );

                betweens.collection.push(PoseGraph2CostTerm {
                    pose_m_from_pose_n: meas,
                    entity_indices: [id_prev, id_curr],
                });
            }

            _ => {}
        }
    }

    let prior = CostTerms::new(
        ["poses"],
        vec![Isometry2PriorCostTerm {
            isometry_prior_mean: Isometry2F64::identity(),
            isometry_prior_precision: MatF64::identity() * 10e2,
            entity_indices: [0],
        }],
    );

    let family = VarFamily::new(VarKind::Free, values);
    let families = VarBuilder::new().add_family("poses", family).build();

    let graph = vec![
        CostFn::new_boxed((), betweens.clone()),
        CostFn::new_boxed((), prior.clone()),
    ];

    (graph, families)
}

pub fn load_g2o_3d(file: &str) -> (Vec<Box<dyn IsCostFn>>, VarFamilies) {
    let file = File::open(file).expect("File not found!");

    let mut values = Vec::new();
    let mut betweens = CostTerms::new(["poses", "poses"], vec![]);

    for line in BufReader::new(file).lines() {
        let line = line.expect("Missing line");
        let parts = line.split(" ").collect::<Vec<&str>>();
        match parts[0] {
            "VERTEX_SE3:QUAT" => {
                // let id = parts[1].parse::<u32>().expect("Failed to parse g20");
                let x = parts[2].parse::<f64>().expect("Failed to parse g20");
                let y = parts[3].parse::<f64>().expect("Failed to parse g20");
                let z = parts[4].parse::<f64>().expect("Failed to parse g20");
                let qx = parts[5].parse::<f64>().expect("Failed to parse g20");
                let qy = parts[6].parse::<f64>().expect("Failed to parse g20");
                let qz = parts[7].parse::<f64>().expect("Failed to parse g20");
                let qw = parts[8].parse::<f64>().expect("Failed to parse g20");

                // TODO: The only way I found to convert this is via a log/exp....
                // Currently no direction constructor from a quat
                let mat = SO3::from_xyzw(qx, qy, qz, qw).log();
                let mat = Vector3::new(mat[0], mat[1], mat[2]);

                let var = Isometry3F64::from_translation_and_rotation(
                    Vector3::new(x, y, z),
                    Rotation3F64::exp(mat),
                );
                values.push(var);
            }

            "EDGE_SE3:QUAT" => {
                let id_prev = parts[1].parse::<usize>().expect("Failed to parse g20");
                let id_curr = parts[2].parse::<usize>().expect("Failed to parse g20");
                let x = parts[3].parse::<f64>().expect("Failed to parse g20");
                let y = parts[4].parse::<f64>().expect("Failed to parse g20");
                let z = parts[5].parse::<f64>().expect("Failed to parse g20");
                let qx = parts[6].parse::<f64>().expect("Failed to parse g20");
                let qy = parts[7].parse::<f64>().expect("Failed to parse g20");
                let qz = parts[8].parse::<f64>().expect("Failed to parse g20");
                let qw = parts[9].parse::<f64>().expect("Failed to parse g20");

                // TODO: Noise models?
                // let m11 = parts[6].parse::<f64>().expect("Failed to parse g20");
                // let m12 = parts[7].parse::<f64>().expect("Failed to parse g20");
                // let m13 = parts[8].parse::<f64>().expect("Failed to parse g20");
                // let m22 = parts[9].parse::<f64>().expect("Failed to parse g20");
                // let m23 = parts[10].parse::<f64>().expect("Failed to parse g20");
                // let m33 = parts[11].parse::<f64>().expect("Failed to parse g20");

                // let inf = Matrix3::new(
                //     m33, m13, m23,
                //     m13, m11, m12,
                //     m23, m12, m22,
                // );

                let mat = SO3::from_xyzw(qx, qy, qz, qw).log();
                let mat = Vector3::new(mat[0], mat[1], mat[2]);

                let meas = Isometry3F64::from_translation_and_rotation(
                    Vector3::new(x, y, z),
                    Rotation3F64::exp(mat),
                );

                betweens.collection.push(PoseGraph3CostTerm {
                    pose_m_from_pose_n: meas,
                    entity_indices: [id_prev, id_curr],
                });
            }

            _ => {}
        }
    }

    let prior = CostTerms::new(
        ["poses"],
        vec![Isometry3PriorCostTerm {
            isometry_prior_mean: Isometry3F64::identity(),
            isometry_prior_precision: MatF64::identity() * 10e2,
            entity_indices: [0],
        }],
    );

    let family = VarFamily::new(VarKind::Free, values);
    let families = VarBuilder::new().add_family("poses", family).build();

    let graph = vec![
        CostFn::new_boxed((), betweens.clone()),
        CostFn::new_boxed((), prior.clone()),
    ];

    (graph, families)
}
