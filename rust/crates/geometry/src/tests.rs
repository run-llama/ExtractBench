use super::union_area;

#[test]
fn overlapping_and_duplicate_rectangles_count_once() {
    let a = [0.0, 0.0, 2.0, 2.0];
    let b = [1.0, 1.0, 3.0, 3.0];
    assert_eq!(
        union_area(&[a, a, b], &[0.0, 1.0, 2.0, 3.0], &[0.0, 1.0, 2.0, 3.0]),
        7.0
    );
}

#[test]
fn touching_and_disjoint_rectangles() {
    let rects = [
        [0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 2.0, 1.0],
        [3.0, 0.0, 4.0, 1.0],
    ];
    assert_eq!(
        union_area(&rects, &[0.0, 1.0, 2.0, 3.0, 4.0], &[0.0, 1.0]),
        3.0
    );
}

#[test]
fn empty_degenerate_and_reversed_rectangles() {
    assert_eq!(union_area(&[], &[], &[]), 0.0);
    assert_eq!(
        union_area(&[[0.0, 0.0, 0.0, 1.0]], &[0.0], &[0.0, 1.0]),
        0.0
    );
    assert_eq!(
        union_area(&[[2.0, 2.0, 0.0, 0.0]], &[0.0, 2.0], &[0.0, 2.0]),
        0.0
    );
}

#[test]
fn nonfinite_comparisons_and_signed_zero_follow_cell_traversal() {
    let nan = f64::NAN;
    assert_eq!(
        union_area(&[[nan, 0.0, 1.0, 1.0]], &[nan, 1.0], &[0.0, 1.0]),
        0.0
    );
    assert_eq!(
        union_area(
            &[[0.0, 0.0, f64::INFINITY, 1.0]],
            &[0.0, f64::INFINITY],
            &[0.0, 1.0]
        ),
        f64::INFINITY
    );
    assert_eq!(
        union_area(&[[-0.0, -0.0, 1.0, 1.0]], &[-0.0, 1.0], &[-0.0, 1.0]),
        1.0
    );
}
