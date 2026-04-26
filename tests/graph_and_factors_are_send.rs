use factrs::core::{Factor, Graph};

fn assert_send<T: Send>() {}

#[test]
fn graph_is_send() {
    assert_send::<Graph>();
}

#[test]
fn factor_is_send() {
    assert_send::<Factor>();
}
