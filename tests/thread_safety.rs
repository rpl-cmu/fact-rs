use factrs::core::{Factor, Graph, Values};

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

#[test]
fn graph_is_send() {
    assert_send::<Graph>();
}

#[test]
fn factor_is_send() {
    assert_send::<Factor>();
}

#[test]
fn values_are_send() {
    assert_send::<Values>();
}

#[test]
fn graph_is_sync() {
    assert_sync::<Graph>();
}

#[test]
fn factor_is_sync() {
    assert_sync::<Factor>();
}

#[test]
fn values_are_sync() {
    assert_sync::<Values>();
}
