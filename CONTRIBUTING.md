# Contributing to Archibald

Thank you for taking the time to look at Archibald. It is a sailboat performance and design library for naval architects and yacht designers. **You do not need to write code to help it.** Most of what makes it better comes from people who know the ships:

- a number that looks wrong,
- a paper that says otherwise,
- a tank test to compare with,
- a method that is missing.

This page explains how to send these, so that they can be acted on quickly.

## Contents

- [Ways to help](#ways-to-help)
- [Before you report](#before-you-report)
- [Reporting a wrong result](#reporting-a-wrong-result)
- [Sources: the one rule](#sources-the-one-rule)
- [Confidential designs and data](#confidential-designs-and-data)
- [Suggesting a method or an improvement](#suggesting-a-method-or-an-improvement)
- [Providing validation data](#providing-validation-data)
- [Contributing code](#contributing-code)
- [Using an AI assistant](#using-an-ai-assistant)
- [Conduct and license](#conduct-and-license)

## Ways to help

Each of these is a useful contribution, roughly in order of how much it helps today:

| You have... | Do this |
|---|---|
| A result that looks wrong, or an error message | [Open an issue](https://github.com/julesricheux/Archibald/issues/new): see [Reporting a wrong result](#reporting-a-wrong-result) |
| A paper, a table or a worked example that a model should match | Open an issue with the reference: see [Sources](#sources-the-one-rule) |
| Published or measured data | See [Providing validation data](#providing-validation-data). Some models are waiting for exactly that |
| An idea: a missing method, an awkward input, an unclear output | Open an issue: see [Suggesting a method](#suggesting-a-method-or-an-improvement) |
| A typo or an unclear sentence in a docstring, an example or this page | Edit it directly on GitHub (pencil icon) and propose the change. No installation needed |
| Python experience | See [Contributing code](#contributing-code) |

Questions are welcome as issues too. If something was hard to understand, others will have the same trouble.

## Before you report

A few minutes here saves a round of questions.

1. **Check what is already known.**
   - [`examples/03 - Validation/README.md`](examples/03%20-%20Validation/README.md) says which models have been compared with published data, and which have not.
   - The "Known issues" section of [`CLAUDE.md`](CLAUDE.md) lists confirmed problems. This file is written for AI coding assistants, but it is plain text and readable.
   - Search the [open and closed issues](https://github.com/julesricheux/Archibald/issues?q=is%3Aissue).

   If your problem is already there, a comment with your case is still useful: it tells us it matters.
2. **Note your version.** Run `pip show archibald-vpp`, or `git log -1` if you work from a clone. `archibald.__version__` may print `unknown`; that is a known bug.
3. **Check the conventions.** Many apparent errors are a sign or a unit. Archibald uses:

| Quantity | Convention |
|---|---|
| Axes | x forward, y to port, z up |
| Heel | positive to starboard |
| Trim | positive bow down |
| Leeway | positive to starboard |
| Immersion | `dz` lifts the boat. A hull with its keel at z = 0 floats at draft T with `dz = -T`. Water is at z = 0 |
| Speeds | boat speed `stw` and true wind speed `tws0` are in knots. Lengths are in m, forces in N, moments in N.m |
| Water | check the density actually used, `op_point.environment.water.density`, before comparing displacements or forces. Fresh and sea water differ by 2.6 % |
| Forces and moments | expressed in the axes that move with the boat, with moments about their origin, not about the CoG or the centre of buoyancy |

4. **Check the range of validity.** Most resistance methods are regressions on a family of hulls, with a published range: Froude number, L/B, B/T, prismatic coefficient, and so on. A result outside that range is not a bug in the code, but it is worth saying, and the method should warn about it.

## Reporting a wrong result

The best report lets someone else get your number on their own computer in a few minutes, and says why you expected another one. Please include:

1. **What you computed.** The shortest script that gives the number. If you cannot share your script, describe the model:
   - the method (`dsyhs`, `holtrop`, `savitsky`...),
   - the inputs, with units,
   - the operating point (speed, heel, trim, leeway, immersion).
2. **What you got**, copied from the output, with units.
3. **What you expected, and where it comes from.** For example: a paper (authors, year, page, table or equation number), a tank test, another program with its version, or a hand calculation you can show. *"It looks too high"* is a good start too: say why, for instance in comparison with a similar hull.
4. **The size of the difference.** 1 % and 30 % are different problems. A small, systematic difference can come from the smoothing Archibald adds so that the optimizer works (see [Contributing code](#contributing-code)). Report it anyway, with its size.
5. **Your version**, and your operating system if the problem is a crash.

You can copy this outline into the issue:

```markdown
**Model / method:** e.g. DSYHS residuary resistance, `dsyhs.compute_Rw_dsyhs`
**Archibald version:** output of `pip show archibald-vpp`
**Case:** hull, inputs and operating point, with units (or attach a short script)
**Archibald gives:** ...
**Expected:** ... (source: authors, year, page / table / equation)
**Difference:** ... %
**Anything else:** range of validity, other cases tried, plots...
```

Plots are welcome: a curve against speed or heel often shows the problem at once, such as a hump, a jump or a sign change.

## Sources: the one rule

**Every formula and every coefficient in Archibald must come from a source that someone else can check.** No value is added from memory, from habit, or because "it is usually about that".

What this means in practice:

- When you report or suggest something, give the **full reference**: authors, title, year, journal or conference, and the page, table or equation.
- Primary sources are preferred to secondary ones, such as a textbook quoting a paper. If you only have the secondary source, say so. That is fine, but it is recorded.
- **Do not post full copyrighted papers** in a public issue. Give the reference. If a table of coefficients is needed, the numbers needed for the comparison can be transcribed, with the reference.
- If a value is a placeholder, for example in an example script, it must be labelled as such in the code (`PLACEHOLDER, NOT SOURCED`). If you see an unlabelled value without a source, that is worth an issue in itself.
- Your own measurements are a source too. Say how they were obtained, and whether they can be published.

## Confidential designs and data

Issues on GitHub are public and stay public. Many contributors work on commercial designs, so:

- **Do not post client hull lines, meshes, offsets, weights or tank results** that you are not allowed to publish.
- Try to show the problem on a geometry shipped with Archibald instead. The examples have a cargo sailboat (Molenez), a 49er and a Nacra 15. A simple box or a Wigley hull also works. A scaled or distorted version of your hull is often enough.
- If the problem only shows on confidential data, open an issue that describes it without the data. For anything that cannot be public, contact the maintainer at the address in [`pyproject.toml`](pyproject.toml).

## Suggesting a method or an improvement

New methods are welcome, but each one has to be maintained and trusted, so please include:

- **The reference**, as above, ideally with a worked example or published results to check the implementation against.
- **The range of validity**: hull type, Froude or speed range, form coefficients, and the test basis it was fitted on.
- **What it adds**: a hull type, a regime or a force component that existing methods do not cover, or better accuracy, shown on a case.
- **The inputs it needs**, and whether they can be computed from a hull mesh or must be given by the user.

Suggestions about usability are equally welcome: an input that is hard to get, an output in an inconvenient form, an error message that did not help, or an example that would have saved you time.

## Providing validation data

[`examples/03 - Validation`](examples/03%20-%20Validation/) compares the literature models with published data. Some comparisons are waiting for numbers. In its `data/` folder, a CSV file whose header starts with `# TO FILL` is a blank template, and the matching script prints `[INPUT NEEDED]` until it has rows. The README of that folder lists what is missing.

To fill one, you do not need to run anything:

1. Copy the values from the source into the CSV, keeping the column names and units of the header.
2. Write the full reference in the header, with the page and table.
3. Send it as a pull request (the GitHub web editor is enough), or attach the file to an issue.

Measured data of your own (tank tests, sea trials, heel tests) is very valuable, as long as you may publish it. Please describe how it was measured and its uncertainty.

## Contributing code

You are welcome to fix things yourself. Small, focused pull requests are easiest to review. For a larger change, open an issue first to agree on the approach.

### Setup

```bash
git clone https://github.com/julesricheux/Archibald.git
cd Archibald
git checkout develop
pip install -e ".[test]"
```

A conda environment is described in `environment.yaml`. Optional plotting packages are in the `viz` extra: `pip install -e ".[test,viz]"`.

Work on a branch from `develop`, and open the pull request against `develop`.

### How Archibald is written

Archibald is forked from [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox). Like it, the same code runs either on numbers or inside an optimizer (CasADi and IPOPT), which differentiates through it. This shapes a few rules:

- Import the maths as `import archibald.numpy as np`. Use plain NumPy, imported as `onp`, only for plotting and post-processing.
- Do not branch on values that may be optimization variables (`if V > 0:`). Use `np.where`, or smooth functions such as `np.softmax`, `np.softplus` or the `smooth_clip` of the hydro modules, and keep them close to the published formula. Too much smoothing changes the result: check against the reference.
- Geometry and analysis objects inherit from `ArchibaldObject`. Physics models inherit from `ExplicitAnalysis` or `ImplicitAnalysis`.
- Docstrings give the purpose, every input with its unit in brackets, the returns, and **the source of the formulas**. Match the style of the file you edit.
- No mutable default arguments, and keyword arguments in long calls.

### Tests

```bash
pytest archibald -q
```

Some tests are known to fail: they use the API inherited from AeroSandbox, which has been removed. [`CLAUDE.md`](CLAUDE.md) gives the current baseline. Compare against it rather than expecting all tests to pass. The CI excludes those tests by name: if you fix one, remove it from `.github/workflows/run-pytest.yml`.

For a method from the literature, the most useful test reproduces a published number, with the reference in the test. Put it in `archibald/dynamics/hydro/test_validation_published.py` or a file like it. Numbers checked there must not move without a reason.

### Before opening the pull request

- [ ] The change has a source, cited in the docstring.
- [ ] A test covers it, ideally against published values.
- [ ] The tests that passed before still pass.
- [ ] The examples you touched run from their own folder.
- [ ] New dependencies, if any, are declared in `pyproject.toml`, `requirements.txt` and `environment.yaml`.
- [ ] The description says what changed in the results, with numbers before and after, if any result moved.

## Using an AI assistant

You may use an AI assistant to write a script, a fix or a report. [`CLAUDE.md`](CLAUDE.md) describes the conventions of the code for such tools. The same rule applies to its output: **an AI assistant is not a source.** Coefficients, formulas and references it suggests must be checked against the original document before they go into a report or a pull request, because invented but plausible values are a real risk. Please say in the pull request if an assistant wrote a significant part of it.

## Conduct and license

Be kind and specific. Disagreement about a method or a result is normal and useful: argue it with sources and numbers, not with people. Contributors come from naval architecture, yacht design and software, with different vocabularies. Explain your terms, and assume good faith.

Archibald is distributed under the [MIT License](LICENSE). By contributing, you agree that your contribution is distributed under the same license. Data you contribute must be data you have the right to publish.
