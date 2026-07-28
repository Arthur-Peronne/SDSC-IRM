# NN — <Report title>

> <1–2 sentences: what this stage does and why it exists in the pipeline.
> Keep it to the final, working idea — no exploration history.>

## 1. Objective

<2–3 sentences: the scientific question. What are we trying to measure, learn,
or check, and where does this step sit in the overall pipeline?>

## 2. Method — what the code does

**Entry point:** [`scripts/run_<x>.py`](../../scripts/run_<x>.py)
**Config:** [`config_files/<x>.yaml`](config_files/<x>.yaml)
**Core functions:** [`src/.../<module>.py`](../../src/.../<module>.py)

Pipeline, in the order the script runs it:

1. **Load** <input data: image type, frame ED/ES, source folder>.
2. **<Transform>** <what happens to the data, with the function that does it>.
3. **<Model / computation>** <the core step>.
4. **<Outputs>** <what is produced and where it is logged/saved>.

Key parameters (from `config_files/<x>.yaml`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `...`     | `...` | ...     |

<If relevant, add a short note on run modes (e.g. CALC vs LOAD) or MLflow
tracking — one or two sentences.>

## 3. Results

<Show the figures and state the numbers plainly. One short paragraph per figure
or per idea. Point to the exact filenames the code produces.>

![<caption>](figures/<figure>.png)

*<What this figure shows and the key takeaway.>*

## 4. Conclusion

<1 short paragraph: the scientific take-away, and what this stage enables /
validates for later reports.>

## 5. Reproduce

- Take the chosen `.yaml` in `config_files/` from this folder, rename it to
  `<x>.yaml`, and put it in the general `configs/` folder.
- Run `scripts/run_<x>.py`.
- Expected outputs: the figures in `figures/`.

## 6. Notes & limitations

- <Gotchas a reader/reproducer must know: data prerequisites, split
  compatibility, naming/normalization conventions, silent skips, etc.>

<!--
Folder layout for each report:
  NN_topic/
  ├── report_<topic>.md   ← this file
  ├── config_files/       ← the YAML(s) that produced the results
  └── figures/            ← the plots, regenerated from the current code
-->