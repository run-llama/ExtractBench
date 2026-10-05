//! Convert Python coordinate batches into owned inputs for the geometry core.
use extract_bench_geometry::{union_area, Rect};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn rect_union_areas(
    py: Python<'_>,
    rectangles: Vec<Vec<Rect>>,
    xs: Vec<Vec<f64>>,
    ys: Vec<Vec<f64>>,
) -> PyResult<Vec<f64>> {
    if rectangles.len() != xs.len() || rectangles.len() != ys.len() {
        return Err(PyValueError::new_err(
            "rectangle and boundary batches must have equal lengths",
        ));
    }
    Ok(py.detach(|| {
        rectangles
            .iter()
            .zip(&xs)
            .zip(&ys)
            .map(|((rects, x), y)| union_area(rects, x, y))
            .collect()
    }))
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(rect_union_areas, module)?)
}
