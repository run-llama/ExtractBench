The geometry crate contains binding-free rectangle coverage calculations.
The PyO3 crate only marshals coordinate batches and releases the GIL while the
core runs. Python retains coordinate boundary ordering, page/group scopes,
text comparison, match selection, thresholds, and metric metadata.

Build the optimized mixed Python/Rust package with `uv sync --extra dev` or
`uv build`. Source builds require Rust and a C linker. Run `cargo test
--manifest-path rust/Cargo.toml --locked`, `cargo clippy --manifest-path
rust/Cargo.toml --all-targets --locked -- -D warnings`, and `cargo fmt
--manifest-path rust/Cargo.toml --all --check`. Python tests compare the native
kernel with the frozen Python geometry implementation, including nonfinite
coordinates, degenerate boxes, and candidate thresholds.

The union algorithm intentionally retains the grid traversal and floating
point accumulation order of the Python scorer. An algorithmic replacement
requires separate parity and performance evidence. Native batches use owned
coordinate arrays; they contain no Python objects or document text.
