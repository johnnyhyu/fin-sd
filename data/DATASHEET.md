# Datasheet: Fin-SD financial reasoning corpus

220 multi-step financial reasoning problems, described in §4.1 and §6 of the
accompanying paper. Released under CC BY 4.0; see [LICENSE-DATA](LICENSE-DATA).

## Motivation

Existing financial reasoning benchmarks largely test information extraction or
the application of a formula whose inputs are stated outright. This corpus was
built to test the step those skip: using financial logic and convention to
*derive* the inputs a formula needs from a realistic scenario.

Each problem is structured in three phases:

1. **Input derivation** — analyze a market, timeline, or balance-sheet scenario
   to compute the exact inputs a target function requires, applying standard
   industry definitions. Key inputs are deliberately not stated directly.
2. **Core function execution** — evaluate the target financial function on those
   derived inputs to get an intermediate metric.
3. **Verifiable closing** — map that metric to a domain-native endpoint, such as
   a comparison against a stated hurdle rate or a precise dollar valuation
   impact, giving a single unambiguous answer.

## Composition

| | |
|---|---|
| `trainingset.json` | **220 problems** — the corpus. Fields: `number`, `problem`, `answer`. |
| `problemset.json` | 1,574 candidate problems across 294 functions, before review. |
| `function-library.json` | The 295 seed functions that survived curation. |

Problems average roughly 1,100 characters. Answers are short and exact —
`$21,600`, `-$1,087,500`, `Exceeds by 43 basis points.` — with rounding stated in
the problem text.

The corpus is split 80/20 into training and validation. **The split is fixed,
not seeded**, so every seed scores the same validation set.

## Collection

1. **Seed.** The 3,133 Python-formatted financial functions released with
   FinanceReasoning (Tang et al., ACL 2025), vendored at
   `benchmarks/FinanceReasoning/data/functions/functions-article-all.json`.
2. **Curate.** Each function graded 1–8 for difficulty by GPT-OSS-120B; those
   scoring ≥ 3 kept, removing trivial operations. A viability judge
   (DeepSeek-V4-Flash) then verified each could support a word problem with a
   single correct answer. **3,133 → 295 functions.**
3. **Generate.** GPT-5.5 wrote context-rich problems against each function in
   the three-phase structure above.
4. **Critique.** A pass with Claude Fable 5 removed contradictory premises and
   quantities that were genuinely underdetermined, focusing the corpus on
   ambiguity a competent analyst could confidently resolve.
5. **Review.** All items reviewed by hand for logical validity and for the
   existence of a unique correct answer under correct convention application.

Steps 1–3 and 5 are reproducible from `build/`; see its README. Because the
generation steps call external models, a rebuild will not match byte for byte.

## Uses

Built for, and used as, the training corpus for Fin-SD and its baselines. Only
problems the student answers incorrectly train, and that set is re-mined every
epoch, so a given run trains on roughly 25 items per epoch out of the ~176
training-split problems.

It is **not** an evaluation set for the paper. All reported accuracy is on the
FinanceReasoning Hard subset, which is held out.

## Limitations

- **Training–evaluation overlap.** 17 of the 295 seed functions appear in the
  solutions of 22 of the 238 FinanceReasoning Hard evaluation items. Results
  reported against that benchmark are therefore in-domain transfer, though the
  problem-writing philosophy differs substantially between the two.
- **Size.** 220 problems is modest, and failure mining means only a subset
  trains in any epoch.
- **No ambiguous items by construction.** Problems admitting multiple
  justifiable assumptions were removed, since grading them fairly needs a rubric
  rather than an exact answer. Real analytical work contains such problems; this
  corpus does not represent them.
- **Synthetic.** Problems are model-generated from a function library, not drawn
  from real filings or transactions. Scenarios are realistic in structure but
  invented. In deployment the intended corpus would be on-premise institutional
  data, which is what makes the method's no-external-teacher property matter.
- **Jurisdiction.** Conventions assumed are those of US GAAP and SEC/FINRA
  practice. Items may be wrong under other accounting standards.
- **Answers are model-generated then human-reviewed.** Review was performed by
  the authors, not by credentialed accountants, and was not independently
  duplicated.

## Distribution and maintenance

Distributed with this repository under CC BY 4.0. The seed function library it
derives from is redistributed under its own upstream terms; see
`benchmarks/FinanceReasoning/NOTICE`.

Maintained by the paper's authors. No update schedule is committed to.
