# Security Policy

## Supported Versions

Security fixes are applied to the latest mainline version.

## Reporting a Vulnerability

Please do not open public issues for security vulnerabilities.

Instead:
- Open a private security advisory on GitHub, or
- Contact the maintainer directly via GitHub profile contact methods.

Include:
- Description of the issue
- Reproduction steps
- Impact assessment
- Suggested remediation (if available)

You can expect an acknowledgment within 72 hours.

## Secrets Handling

- Never commit API keys, credentials, or private datasets.
- Use environment variables (`GROQ_API_KEY`, etc.).
- Keep `.gitignore` updated for local runtime artifacts.
