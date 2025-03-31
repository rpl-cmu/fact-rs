use rerun::{
    components::RotationQuat, Arrows2D, Arrows3D, AsComponents, Points2D, Points3D, Quaternion,
    Rotation3D, Transform3D, Vec2D, Vec3D,
};

use crate::{
    containers::Values,
    optimizers::OptObserver,
    variables::{MatrixLieGroup, VariableDtype, VectorVar2, VectorVar3, SE2, SE3, SO2, SO3},
};

// ------------------------- 2D Objects ------------------------- //
// 2D Vectors
#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a VectorVar2> for Vec2D {
    fn from(v: &'a VectorVar2) -> Vec2D {
        Vec2D::new(v[0] as f32, v[1] as f32)
    }
}

impl From<VectorVar2> for Vec2D {
    fn from(v: VectorVar2) -> Vec2D {
        (&v).into()
    }
}

impl<'a> From<&'a VectorVar2> for Points2D {
    fn from(v: &'a VectorVar2) -> Points2D {
        let vec: Vec2D = v.into();
        Points2D::new([vec])
    }
}

impl From<VectorVar2> for Points2D {
    fn from(v: VectorVar2) -> Points2D {
        (&v).into()
    }
}

// 2D Rotations
#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SO2> for Arrows2D {
    fn from(so2: &'a SO2) -> Arrows2D {
        let mat = so2.to_matrix().map(|x| x as f32);
        let x: [f32; 2] = mat
            .column(0)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        let y: [f32; 2] = mat
            .column(1)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        Arrows2D::from_vectors([x, y]).with_colors([[255, 0, 0], [0, 255, 0]])
    }
}

impl From<SO2> for Arrows2D {
    fn from(so2: SO2) -> Arrows2D {
        (&so2).into()
    }
}

// 2D SE2
#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SE2> for Arrows2D {
    fn from(se2: &'a SE2) -> Arrows2D {
        let xy: [f32; 2] = se2
            .xy()
            .map(|x| x as f32)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        let arrows: Arrows2D = se2.rot().into();
        arrows.with_origins([xy, xy])
    }
}

impl From<SE2> for Arrows2D {
    fn from(se2: SE2) -> Arrows2D {
        (&se2).into()
    }
}

#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SE2> for Points2D {
    fn from(se2: &'a SE2) -> Points2D {
        let xy = [se2.x() as f32, se2.y() as f32];
        Points2D::new([xy])
    }
}

impl From<SE2> for Points2D {
    fn from(se2: SE2) -> Points2D {
        (&se2).into()
    }
}

#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SE2> for Vec2D {
    fn from(se2: &'a SE2) -> Vec2D {
        let xy = [se2.x() as f32, se2.y() as f32];
        Vec2D::new(xy[0], xy[1])
    }
}

impl From<SE2> for Vec2D {
    fn from(se2: SE2) -> Vec2D {
        (&se2).into()
    }
}

// ------------------------- 3D Objects ------------------------- //
// 3D Vectors
#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a VectorVar3> for Vec3D {
    fn from(v: &'a VectorVar3) -> Vec3D {
        Vec3D::new(v[0] as f32, v[1] as f32, v[2] as f32)
    }
}

impl From<VectorVar3> for Vec3D {
    fn from(v: VectorVar3) -> Vec3D {
        (&v).into()
    }
}

impl<'a> From<&'a VectorVar3> for Points3D {
    fn from(v: &'a VectorVar3) -> Points3D {
        let vec: Vec3D = v.into();
        Points3D::new([vec])
    }
}

impl From<VectorVar3> for Points3D {
    fn from(v: VectorVar3) -> Points3D {
        (&v).into()
    }
}

// 3D Rotations
#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SO3> for Rotation3D {
    fn from(so3: &'a SO3) -> Rotation3D {
        let xyzw = [
            so3.x() as f32,
            so3.y() as f32,
            so3.z() as f32,
            so3.w() as f32,
        ];
        Rotation3D::Quaternion(RotationQuat(Quaternion::from_xyzw(xyzw)))
    }
}

impl From<SO3> for Rotation3D {
    fn from(so3: SO3) -> Rotation3D {
        (&so3).into()
    }
}

#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SO3> for Arrows3D {
    fn from(so3: &'a SO3) -> Arrows3D {
        let mat = so3.to_matrix().map(|x| x as f32);
        let x: [f32; 3] = mat
            .column(0)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        let y: [f32; 3] = mat
            .column(1)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        let z: [f32; 3] = mat
            .column(2)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        Arrows3D::from_vectors([x, y, z]).with_colors([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
    }
}

impl From<SO3> for Arrows3D {
    fn from(so3: SO3) -> Arrows3D {
        (&so3).into()
    }
}

// 3D Transforms
impl<'a> From<&'a SE3> for Transform3D {
    fn from(se3: &'a SE3) -> Transform3D {
        let xyz: VectorVar3 = se3.xyz().clone_owned().into();
        let xyz: Vec3D = xyz.into();
        let rot: Rotation3D = se3.rot().into();
        Transform3D::from_translation_rotation(xyz, rot)
    }
}

impl From<SE3> for Transform3D {
    fn from(se3: SE3) -> Transform3D {
        (&se3).into()
    }
}

#[allow(clippy::unnecessary_cast)]
impl<'a> From<&'a SE3> for Arrows3D {
    fn from(se3: &'a SE3) -> Arrows3D {
        let arrows: Arrows3D = se3.rot().into();
        let xyz: [f32; 3] = se3
            .xyz()
            .map(|x| x as f32)
            .as_slice()
            .try_into()
            .expect("Failed to convert to slice");
        arrows.with_origins([xyz, xyz, xyz])
    }
}

impl From<SE3> for Arrows3D {
    fn from(se3: SE3) -> Arrows3D {
        (&se3).into()
    }
}

impl<'a> From<&'a SE3> for Vec3D {
    fn from(se3: &'a SE3) -> Vec3D {
        let xyz: VectorVar3 = se3.xyz().clone_owned().into();
        xyz.into()
    }
}

impl From<SE3> for Vec3D {
    fn from(se3: SE3) -> Vec3D {
        (&se3).into()
    }
}

impl<'a> From<&'a SE3> for Points3D {
    fn from(se3: &'a SE3) -> Points3D {
        let xyz: VectorVar3 = se3.xyz().clone_owned().into();
        let xyz: Vec3D = xyz.into();
        Points3D::new([xyz])
    }
}

impl From<SE3> for Points3D {
    fn from(se3: SE3) -> Points3D {
        (&se3).into()
    }
}

// ------------------------- All Together ------------------------- //
// 2D Gatherers
impl<'a> FromIterator<&'a VectorVar2> for Points2D {
    fn from_iter<I: IntoIterator<Item = &'a VectorVar2>>(iter: I) -> Points2D {
        let mut points = Vec::new();

        for v in iter {
            let this: Vec2D = v.into();
            points.push(this);
        }

        Points2D::new(points)
    }
}

impl<'a> FromIterator<&'a SE2> for Arrows2D {
    fn from_iter<I: IntoIterator<Item = &'a SE2>>(iter: I) -> Arrows2D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se2 in iter {
            let this: Arrows2D = se2.into();
            vectors.extend_from_slice(&this.vectors);
            origins.extend_from_slice(
                &this
                    .origins
                    .expect("SE2 missing origins in conversion to rerun::Arrows2D"),
            );
            colors.extend_from_slice(
                &this
                    .colors
                    .expect("SE2 missing colors in conversion to rerun::Arrows2D"),
            );
        }

        Arrows2D::from_vectors(vectors)
            .with_origins(origins)
            .with_colors(colors)
    }
}

impl<'a> FromIterator<&'a SE2> for Points2D {
    fn from_iter<I: IntoIterator<Item = &'a SE2>>(iter: I) -> Points2D {
        let mut points = Vec::new();

        for se2 in iter {
            let this: Vec2D = se2.into();
            points.push(this);
        }

        Points2D::new(points)
    }
}

// 3D Gatherers
impl<'a> FromIterator<&'a VectorVar3> for Points3D {
    fn from_iter<I: IntoIterator<Item = &'a VectorVar3>>(iter: I) -> Points3D {
        let mut points = Vec::new();

        for v in iter {
            let this: Vec3D = v.into();
            points.push(this);
        }

        Points3D::new(points)
    }
}

impl<'a> FromIterator<&'a SE3> for Arrows3D {
    fn from_iter<I: IntoIterator<Item = &'a SE3>>(iter: I) -> Arrows3D {
        let mut vectors = Vec::new();
        let mut origins = Vec::new();
        let mut colors = Vec::new();

        for se3 in iter {
            let this: Arrows3D = se3.into();
            vectors.extend_from_slice(&this.vectors);
            origins.extend_from_slice(
                &this
                    .origins
                    .expect("SE3 missing origins in conversion to rerun::Arrows3D"),
            );
            colors.extend_from_slice(
                &this
                    .colors
                    .expect("SE3 missing colors in conversion to rerun::Arrows3D"),
            );
        }

        Arrows3D::from_vectors(vectors)
            .with_origins(origins)
            .with_colors(colors)
    }
}

impl<'a> FromIterator<&'a SE3> for Points3D {
    fn from_iter<I: IntoIterator<Item = &'a SE3>>(iter: I) -> Points3D {
        let mut points = Vec::new();

        for se3 in iter {
            let this: Vec3D = se3.into();
            points.push(this);
        }

        Points3D::new(points)
    }
}

// ------------------------- Streamer ------------------------- //
/// Rerun optimizer observer
///
/// Observer that can be plugged into an optimizer to record the state of the
/// optimization at each optimization step. Generic V is the variable type and R
/// is the rerun type.
///
/// See the observers field of either
/// [GaussNewton](crate::optimizers::GaussNewton) or
/// [LevenMarquardt](crate::optimizers::LevenMarquardt) for more information.
pub struct RerunObserver<V, R>
where
    V: VariableDtype + 'static,
    R: AsComponents,
    for<'a> R: FromIterator<&'a V>,
{
    rec: rerun::RecordingStream,
    topic: String,
    r_phantom: std::marker::PhantomData<R>,
    v_phantom: std::marker::PhantomData<V>,
}

impl<V, R> RerunObserver<V, R>
where
    V: VariableDtype + 'static,
    R: AsComponents,
    for<'a> R: FromIterator<&'a V>,
{
    pub fn new(rec: rerun::RecordingStream, topic: &str) -> Self {
        Self {
            rec,
            topic: topic.to_string(),
            r_phantom: std::marker::PhantomData,
            v_phantom: std::marker::PhantomData,
        }
    }
}

impl<V, R> OptObserver for RerunObserver<V, R>
where
    V: VariableDtype + 'static,
    R: AsComponents,
    for<'a> R: FromIterator<&'a V>,
{
    fn on_step(&self, values: &Values, idx: f64) {
        self.rec.set_time_seconds("stable_time", idx);
        let sol: R = values.filter::<V>().collect();
        self.rec
            .log(self.topic.clone(), &sol)
            .expect("Failed to log topic");
    }
}
