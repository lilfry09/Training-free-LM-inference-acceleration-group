# Cached PPL Sanity Check

- Dataset: `wikitext` `test`
- Slice: 512 context tokens + 256 continuation tokens

| metric | ppl | nll_sum | tokens |
| --- | --- | --- | --- |
| Exact teacher-forced PPL | 63.718854 | 17012.597656 | 4095 |
| Same-slice teacher-forced PPL | 13.088025 | 658.354614 | 256 |
| Same-slice cached PPL | 13.088739 | 658.368575 | 256 |

The same-slice teacher-forced PPL uses the same continuation tokens as cached PPL.
Therefore this check isolates slice choice from cache-based scoring.
