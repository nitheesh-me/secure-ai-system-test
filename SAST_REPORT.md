## static analysis security testing (SAST)


## Bandit

```sh
bandit -r .. -x venv # -o ../models/bandit-report-2 -f txt
```

Initial run: [bandit-report-1](./models/bandit-report-1)


1. [CWE-330 | Use of Insufficiently Random Values](https://cwe.mitre.org/data/definitions/330.html)
2. [CWE-502 | Deserialization of Untrusted Data](https://cwe.mitre.org/data/definitions/502.html)

```
Code scanned:
        Total lines of code: 320
        Total lines skipped (#nosec): 0

Run metrics:
        Total issues (by severity):
                Undefined: 0
                Low: 2
                Medium: 2
                High: 0
        Total issues (by confidence):
                Undefined: 0
                Low: 0
                Medium: 0
                High: 4
```

## Semgrep

> Note: This was run after updates to the codebase to address Bandit findings.

```sh
semgrep --config p/ci python
┌──── ○○○ ────┐
│ Semgrep CLI │
└─────────────┘

                                                                                                     
Scanning 6 files (only git-tracked) with 145 Code rules:
            
  CODE RULES
                                                                                                     
  Language      Rules   Files          Origin      Rules                                             
 ─────────────────────────────        ───────────────────                                            
  <multilang>       2       6          Community     145                                             
  python           19       5                                                                        
                           
┌──────────────┐
│ Scan Summary │
└──────────────┘
✅ Scan completed successfully.
 • Findings: 0 (0 blocking)
 • Rules run: 21
 • Targets scanned: 6
 • Parsed lines: ~100.0%
 • Scan was limited to files tracked by git
 • For a detailed list of skipped files and lines, run semgrep with the --verbose flag
Ran 21 rules on 6 files: 0 findings.
```