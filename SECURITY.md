# Security and data privacy

## Supported versions

Security fixes are made against the latest tagged release. Users should reproduce a
problem on that release before reporting it when possible.

## Reporting a vulnerability

Use the repository's private
[security-advisory form](https://github.com/Sneakypeat/RefGauge/security/advisories/new)
to report a vulnerability. Do not open a public issue containing credentials,
participant identifiers, controlled-access metadata, private download URLs, or an
unpatched exploit. Include the RefGauge and Python versions, operating system,
minimal reproduction, and expected impact.

RefGauge reads local tabular summary statistics and writes local reports. It does
not upload data or make network requests at runtime. Users are responsible for
ensuring that supplied summaries satisfy their data-use agreements.
