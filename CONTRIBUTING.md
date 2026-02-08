# Contributing to genviz

Thanks for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/tsangha/genviz-rs.git
cd genviz-rs
cargo build
cargo test
```

## Running Checks

All of these must pass before submitting a PR:

```bash
cargo check
cargo clippy -- -D warnings
cargo test
cargo fmt --check
```

## Pull Requests

1. Fork the repository and create your branch from `main`
2. Add tests for any new functionality
3. Ensure all checks pass
4. Update documentation for any changed public API
5. Update CHANGELOG.md under an `[Unreleased]` section

## Reporting Issues

Open an issue at https://github.com/tsangha/genviz-rs/issues with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Rust version (`rustc --version`)

## Code Style

- Follow standard Rust conventions
- All public items must have doc comments
- Use `thiserror` for error types
- Prefer returning `Result` over panicking

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
