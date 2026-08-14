# Style Notes — Extracted from paper/style_refs/
# Sources: CrossWalk (AAAI-22), Adversarial Graph Embeddings (IJCAI-20/arXiv),
#          Promoting High Consensus News (AAAI-21 + 2026 update)
# All three share authors (Babaei, Weller, Khajehnejad) → consistent house style.

---

## 1. Voice and Person
- **Active, first-person plural** throughout: "We introduce", "We show", "We propose", "We define".
- Never "This paper presents" or "It is shown that".
- Preferred gap-to-contribution pivot: **"Here, we..."** (appears in CrossWalk §1 and Adv.GE §1).
  Not "In this paper, we..." — that phrase is avoided, especially at end of abstract.

## 2. Abstract Structure (6-beat template)
1. **Context sentence** — motivating domain, 1 sentence, no citation required.
2. **Gap sentence** — "However, ..." or direct description of what existing methods miss.
3. **"Here, we..." sentence** — method in one sentence (no jargon overload).
4. **Quantified result** — numbers appear immediately, no hedging ("dramatically reduces disparity while remaining competitive").
5. *(Optional)* **Scope / generality** — "applicable to any...", "extends to...".
6. *(Optional)* **Honest caveat** — one negative or limitation noted plainly.
- Target length: 130–175 words. Never exceeds 200.

## 3. Numbers-First Rule
- Put the number as early in the clause as possible:
  - ✓ "8.6% improvement in revenue at n=2000"
  - ✗ "an improvement of 8.6% in revenue"
- Percentages: one decimal place (8.6%, 40.5%).
- Revenue/score: one decimal place (462.6, 305.6).
- P-values: exact form "p < 0.0001", never "p ≈ 0" or "significant".
- Comparison phrasing: "462.6 vs. 460.0" (not "higher than 460.0 by 2.6").

## 4. Contribution Bullets
- Numbered (1. 2. 3. 4.) or Roman (I. II. III.) — **never dashes or dots**.
- Each bullet starts: **"We [present-tense verb] ..."**
  - Typical verbs: "introduce", "propose", "show", "demonstrate", "provide", "develop".
- Final bullet often states generalizability or applicability.
- Bullets appear under the heading **"Our Contributions"** or inline as `\textbf{Contributions.}`
  followed by an `enumerate` environment.

## 5. Sentence Rhythm
- Alternates: **short declarative** (10–15 words) + **longer explanatory** (20–30 words).
- Maximum sentence: ~35 words before splitting.
- Example pattern (CrossWalk §1):
  "Fairness in machine learning is receiving growing attention [12 words].
   Decisions made by such systems often affect different population subgroups
   disproportionately [15 words]. As a result, fairness — the absence of prejudice
   or favoritism ... — has received much recent interest [25 words]."

## 6. Related Work Style
- "[Author et al., year] showed/proposed/introduced/demonstrated ..."  
  (subject = cited work, not "prior work by X").
- 1–2 sentences per cited method; no separate paragraph per method in intro.
- Gap signaled by: "However, ...", "Yet, ...", "Despite ..., ... remains ...".
- No critiquing tone — "does not consider X" not "fails to account for X" unless framing is clear.

## 7. Notation and Tense
- **Present simple** for definitions and methodology:
  "The GNN encoder maps node features to embeddings."
  "We define the valuation of buyer $i$ as..."
- **Past** for reported experimental results (within Experiments section only):
  "Our model achieved 462.6 revenue..."
- **Conditional present** for stated claims in intro/abstract:
  "the learned policy achieves 305.6" (not "achieved").
- Math notation: use LaTeX `\mathcal`, `\mathbb`, bold for sets; no italic for operators.

## 8. Section and Subsection Openers
- Every section/subsection: **1-sentence framing statement** that tells the reader what they will learn.
- Never "In this section we present/describe/introduce".
- Subsection openers: state the sub-problem, not the structure.
  - ✓ "Budget constraints transform seed selection into a sequential knapsack."
  - ✗ "In this subsection we describe the budget-constrained formulation."

## 9. Citation Style (ACM sigconf / natbib)
- In-text with year: `\citet{key}` → "Babaei et al. (2013)"
- Parenthetical: `\citep{key}` → "(Babaei et al., 2013)"
- Multiple citations: `\citep{key1,key2}` → "(Kempe et al., 2003; Li et al., 2018)"
- Do NOT write "(see [1], [2])" — use author-year form consistently.

## 10. WSDM-Specific Framing (from Promoting High Consensus News)
- Open intro with **domain/application impact** (economics, society, market), not algorithm description.
- Use concrete examples with numbers to motivate: "only 12% of Conservative Fox News viewers..."
- For a WSDM audience: frame as "network economics / viral marketing" first;
  "deep learning as instrument" second—not the other way round.
- Contribution bullets: Roman numerals (I. II. III.) used in this paper,
  but Arabic (1. 2. 3.) also common in CrossWalk. Be consistent within the paper.

---
## Quick Reference: Phrases to Use / Avoid

| Use                                     | Avoid                                      |
|-----------------------------------------|--------------------------------------------|
| "Here, we introduce..."                 | "In this paper, we present..."             |
| "462.6 vs. 460.0 (p < 0.0001)"         | "significantly outperforms (p < 0.0001)"  |
| "We show that X achieves Y"             | "It can be seen that X achieves Y"         |
| "[Babaei et al., 2013] showed..."       | "Previous work showed [1]..."              |
| "However, these methods ignore..."      | "Unfortunately, prior art has failed..."   |
| "Calibrated DP leads at mid-k"          | "Interestingly, calibrated DP..."          |
| "trained on n≤440, zero-shot at n=2000" | "generalized well beyond training range"   |
