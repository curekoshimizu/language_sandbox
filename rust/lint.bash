#!/usr/bin/env bash

cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
# cargo fix
# cargo test
