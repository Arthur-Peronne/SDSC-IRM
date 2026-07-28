# <NN — Report title>

> One-line summary of what this stage does and why it exists in the pipeline.

## 1. Objective

<2–3 sentences: the scientific question. What are we trying to measure or learn,
and where does this step sit in the overall pipeline?>

## 2. Method — what the code does

**Entry point:** [`scripts/run_<x>.py`](../../scripts/run_<x>.py)
**Config:** [`configs/<x>.yaml`](configs/<x>.yaml)
**Core functions:** [`src/.../<module>.py`](../../src/.../<module>.py)

<Describe the pipeline in the order the code runs it: input data (image type,
frame ED/ES, split), the transformation(s), the model, and the outputs. Keep it
to what the code actually does — no exploration history. Name the key
parameters and what they control.>

Key parameters (from `configs/<x>.yaml`):

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `...`     | `...` | ...     |

## 3. Reproduce

```bash
python scripts/run_<x>.py
```

- **Config used:** [`configs/<x>.yaml`](configs/<x>.yaml) (frozen copy in this folder)
- **MLflow:** experiment `<experiment_name>`, run_id(s) in [`run_ids.md`](run_ids.md)
- **Expected outputs:** the figures in [`figures/`](figures/)

## 4. Results

<Show the figures and state the numbers plainly. One short paragraph per figure.>

![<caption>](figures/<figure>.png)

*<What this figure shows and the key takeaway.>*

## 5. Conclusion

<1 short paragraph: the scientific take-away. What did this stage establish?>

## 6. Notes & limitations

<Optional. Gotchas a reader/reproducer must know: split compatibility,
normalization conventions, data prerequisites, etc.>