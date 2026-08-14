# WSDM 2028 — Venue Notes
# Generated: 2026-07-12 | Re-verify every 4 weeks until CFP published

## CFP Status
- URL checked: https://www.wsdm-conference.org/2028/ → **404 Not Found** (2026-07-12)
- WSDM 2028 CFP not yet published (call usually released Aug–Sep of year prior)
- **TODO (check Aug 2026):** Update all dates below once CFP appears.
  Bookmark: https://www.wsdm-conference.org  |  https://dl.acm.org/conference/wsdm

## Known Defaults (from WSDM 2027 precedent — ACM Web Search and Data Mining)
| Item | WSDM 2027 value | WSDM 2028 (estimated) |
|------|----------------|----------------------|
| Abstract deadline | 2026-08-15 23:59 AoE | ~Aug 2027 AoE |
| Full-paper deadline | 2026-08-15 23:59 AoE (same day) | ~Aug 2027 AoE |
| Notification | ~Oct 2026 | ~Oct 2027 |
| Camera-ready | ~Nov 2026 | ~Nov 2027 |
| Conference dates | Feb 15–19, 2027 (Henderson NV) | Feb 2028 (TBD) |
| Page limit (body) | **8 pages** | 8 pages (expected) |
| References | Unlimited (not counted) | Unlimited |
| Camera-ready | 10 pages + refs | 10 pages + refs |
| Format | ACM sigconf, pdfLaTeX | ACM sigconf |
| Review model | **Double-blind** | Double-blind |

## Anonymization Requirements (double-blind)

### Author information
- Remove all author names, affiliations, acknowledgments from submission
- Use `\documentclass[sigconf,review,anonymous]{acmart}` — this removes the author block automatically
- Do NOT put author names in PDF metadata

### Self-citations (WSDM 2027 prior work)
- Reference our WSDM 2027 GNN-IM-RL paper as: "our prior work [ANON]" or "[ANON]"
- BibTeX entry for blind submission:
  ```bibtex
  @misc{anon_wsdm27,
    author    = {Anonymized for double-blind review},
    title     = {Anonymized},
    year      = {2027},
    note      = {To appear, WSDM 2027},
  }
  ```
- In text: `\cite{anon_wsdm27}` — renders as "[ANON]"

### Code repository
- Replace any GitHub link with: https://anonymous.4open.science/ (create anonymous mirror)
- Fallback: "Code available upon acceptance."
- Do NOT mention institution-specific compute clusters or grant numbers that identify authors

### System names
- "GNN-IM-RL" and "GAIL-RL-Rich" from WSDM 2027 may be kept (technical terms, not lab names)
- Avoid phrases like "our previous WSDM paper" (use citation form only)

## acmart Class Invocation for This Submission
```latex
\documentclass[sigconf,review,anonymous]{acmart}
% review  → adds line numbers (required for reviewing)
% anonymous → strips author block for double-blind
```
Switch to `\documentclass[sigconf]{acmart}` for camera-ready (add real author block).

## Supplemental Material
- WSDM typically allows unlimited supplemental appendix after references (not reviewed)
- Mark clearly: "Appendix — not for review"

## Notes for This Paper
- Venue note in abstract: do NOT write "WSDM 2028" anywhere in the blind submission body
- Self-citation count: 1 prior work (WSDM '27) → acceptable (does not reveal identity alone)
- Reviewer pool: WSDM skews Web/IR/RecSys; motivate from network economics first, ML second
