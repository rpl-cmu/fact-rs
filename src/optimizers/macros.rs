/// Test optimizers
///
/// This macro generates a handful of sanity tests for an optimizer. It tests
/// - Prior optimization for VectorVar3, SO3, and SE3
/// - Between optimization for VectorVar3, SO3, and SE3
#[macro_export]
macro_rules! test_optimizer {
    ($o:ty) => {
        #[test]
        fn priorvector3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_prior::<$o, 3, $crate::variables::VectorVar3>(&f)
        }

        #[test]
        fn priorso3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_prior::<$o, 3, $crate::variables::SO3>(&f);
        }

        #[test]
        fn priorse3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_prior::<$o, 6, $crate::variables::SE3>(&f);
        }

        #[test]
        fn betweenvector3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_between::<$o, 3, 6, $crate::variables::VectorVar3>(
                &f,
            );
        }

        #[test]
        fn betweenso3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_between::<$o, 3, 6, $crate::variables::SO3>(&f);
        }

        #[test]
        fn betweense3() {
            let f = |graph| <$o>::new_default(graph);
            $crate::optimizers::test::optimize_between::<$o, 6, 12, $crate::variables::SE3>(&f);
        }
    };
}
