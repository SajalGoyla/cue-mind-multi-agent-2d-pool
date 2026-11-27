# v3 physics-hint validation (noise-free pooltool, dev/floor states)

- states: 120 max, pot variants evaluated: 1962
- real scratch rate: 13.7%; warned: 17.3%
- scratch warning precision: 37%  recall: 47%
- cue-ball stop error: median 0.43 m, mean 0.52 m (table 0.99 x 1.98 m); naive baseline (cue ball stays at the contact point): median 0.56 m
- same half of the table (upper/lower): 75%; same side (left/right): 65%

| spin | speed | n | real scratch | warned | precision | recall | median stop error (m) |
|---|---|---|---|---|---|---|---|
| stun | soft | 218 | 11% | 8% | 44% | 32% | 0.36 |
| stun | medium | 218 | 11% | 12% | 46% | 50% | 0.42 |
| stun | firm | 218 | 17% | 12% | 65% | 47% | 0.47 |
| follow | soft | 218 | 15% | 27% | 33% | 59% | 0.35 |
| follow | medium | 218 | 17% | 30% | 31% | 56% | 0.55 |
| follow | firm | 218 | 19% | 32% | 33% | 55% | 0.66 |
| draw | soft | 218 | 4% | 7% | 25% | 44% | 0.30 |
| draw | medium | 218 | 11% | 13% | 34% | 42% | 0.40 |
| draw | firm | 218 | 19% | 15% | 41% | 32% | 0.56 |
