# Security Policy

## Supported Versions

This repository is maintained for research and educational use. Security fixes are considered for the current `master` branch.

## Reporting a Vulnerability

Please do **not** open a public issue for security-sensitive problems.

If you discover a vulnerability, exposed credential, unsafe file, or data leakage risk, please contact the repository owner privately first. Include:

- A short description of the problem
- The affected file, dependency, or workflow
- Steps to reproduce, if applicable
- Suggested mitigation, if available

## Sensitive Content Policy

Do not commit:

- API keys, passwords, private tokens, or credential files
- Private datasets or patient-identifiable data
- Large model checkpoints or experiment outputs
- Local environment files such as `.env`, virtual environments, or IDE settings

Use external storage for datasets and trained weights, and document access instructions in the README instead of committing the files directly.
