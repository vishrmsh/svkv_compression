# Frozen redundancy closure check

The original protocols are reused; these are descriptive comparisons, not a fresh validation set.
Clinical evidence is aggregate only. All non-full methods retain the same count within each context/stay.

## Synthetic

Completed 60/60 attempts; statuses `{'complete': 60}`.

| Method | Mean | SVDD − method | Paired 95% interval |
|---|---:|---:|---:|
| svdd | 0.861111 | 0.000000 | [0.000000, 0.000000] |
| mass_oracle | 0.319444 | 0.541667 | [0.486111, 0.597222] |
| mass_warm | 0.316667 | 0.544444 | [0.488889, 0.600000] |
| recency | 0.000000 | 0.861111 | [0.825000, 0.894444] |
| random | 0.472222 | 0.388889 | [0.344444, 0.433333] |
| full | 0.722222 | 0.138889 | [0.100000, 0.180556] |
| keydiff | 0.663889 | 0.197222 | [0.155556, 0.241667] |
| linear_leverage | 0.694444 | 0.166667 | [0.113889, 0.222222] |
| kernel_leverage | 0.836111 | 0.025000 | [0.005556, 0.044444] |

## Clinical

Completed 718/1465 attempts; statuses `{'complete': 718, 'no_event': 747}`.

| Method | Mean | SVDD − method | Paired 95% interval |
|---|---:|---:|---:|
| svdd | 0.464374 | 0.000000 | [0.000000, 0.000000] |
| density | 0.225375 | 0.238999 | [0.194388, 0.283529] |
| recency | 0.408778 | 0.055597 | [0.015772, 0.096418] |
| random | 0.332606 | 0.131768 | [0.094243, 0.170001] |
| full | 1.000000 | -0.535626 | [-0.563195, -0.507638] |
| keydiff | 0.407557 | 0.056817 | [0.027638, 0.087196] |
| linear_leverage | 0.484897 | -0.020522 | [-0.041662, -0.000082] |
| kernel_leverage | 0.479547 | -0.015172 | [-0.037696, 0.007088] |

