//! Numerical rectangle coverage over host-ordered coordinate boundaries.
#![forbid(unsafe_code)]

pub type Rect = [f64; 4];

/// Union area using the caller's sorted, distinct x and y boundaries.
///
/// The host owns boundary ordering, including its treatment of nonfinite
/// coordinates and signed zero. Cell traversal and accumulation order match
/// the Python scorer so close candidate scores do not acquire new ties.
///
/// ```
/// use extract_bench_geometry::union_area;
/// assert_eq!(union_area(&[[0.0, 0.0, 2.0, 1.0]], &[0.0, 2.0], &[0.0, 1.0]), 2.0);
/// ```
pub fn union_area(rectangles: &[Rect], xs: &[f64], ys: &[f64]) -> f64 {
    let mut total = 0.0;
    for x in xs.windows(2) {
        let (left, right) = (x[0], x[1]);
        if right <= left {
            continue;
        }
        for y in ys.windows(2) {
            let (top, bottom) = (y[0], y[1]);
            if bottom <= top {
                continue;
            }
            if rectangles.iter().any(|rect| {
                rect[0] <= left && rect[2] >= right && rect[1] <= top && rect[3] >= bottom
            }) {
                total += (right - left) * (bottom - top);
            }
        }
    }
    total
}

#[cfg(test)]
mod tests;
