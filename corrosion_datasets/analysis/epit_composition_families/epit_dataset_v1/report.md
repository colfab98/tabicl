# EPIT composition-family analysis

This target-free report analyzes the 403 exact composition templates in `epit_dataset_v1`, matching the static empirical composition profile documented in `main2.tex`.

The reported core range is P05--P95 among templates where the element is positive. Presence must be used alongside that range; the range alone does not describe sparsity.

## Family coverage

| Family | Templates | Support | Expected base dominant | Median active (17) | Median largest share |
|---|---:|---|---:|---:|---:|
| fe_alloy | 298 | supported | 98.7% | 4.0 | 66.7% |
| al_alloy | 56 | supported | 87.5% | 2.0 | 76.5% |
| hea | 19 | limited | n/a | 4.0 | 25.5% |
| nicrmo_alloy | 17 | limited | 100.0% | 3.0 | 66.7% |
| other | 13 | insufficient | n/a | 2.0 | 72.0% |

## fe_alloy

Dominant elements: Fe 294/298, Ni 2/298, Cr 1/298, Mo 1/298.

Visible 17-element sum P05/median/P95: 85.96 / 99.27 / 100.00 wt.%.

| Element | Presence | Positive P05 | Positive median | Positive P95 |
|---|---:|---:|---:|---:|
| Fe | 100.0% | 47.113 | 66.707 | 83.026 |
| Cr | 96.3% | 13.188 | 18.100 | 27.210 |
| Ni | 80.9% | 0.330 | 13.200 | 25.580 |
| Mo | 73.2% | 0.019 | 2.170 | 6.153 |

Strongest positive-only log correlations:

- Ni--Nb: r=0.861, n=16
- Cr--Nb: r=-0.855, n=24
- Cr--Al: r=-0.845, n=15
- Ni--W: r=-0.625, n=13
- Mo--Al: r=-0.589, n=12

## al_alloy

Dominant elements: Al 49/56, Mo 3/56, Cr 2/56, Nb 2/56.

Visible 17-element sum P05/median/P95: 96.75 / 100.00 / 100.00 wt.%.

| Element | Presence | Positive P05 | Positive median | Positive P95 |
|---|---:|---:|---:|---:|
| Al | 100.0% | 43.980 | 76.495 | 99.500 |
| Mo | 32.1% | 9.008 | 26.215 | 57.700 |
| Cr | 26.8% | 0.008 | 17.640 | 51.504 |
| Fe | 14.3% | 0.001 | 0.325 | 9.153 |

Strongest positive-only log correlations:

- Mo--Al: r=-0.927, n=18
- Fe--Al: r=-0.698, n=8
- Cr--Al: r=-0.625, n=15

## hea

Dominant elements: Ni 10/19, Co 4/19, Fe 2/19, Cr 1/19, Cu 1/19, Ti 1/19.

Visible 17-element sum P05/median/P95: 75.27 / 99.89 / 100.14 wt.%.

| Element | Presence | Positive P05 | Positive median | Positive P95 |
|---|---:|---:|---:|---:|
| Fe | 94.7% | 16.390 | 22.950 | 36.883 |
| Ni | 89.5% | 11.088 | 24.600 | 38.416 |
| Cr | 89.5% | 16.434 | 21.300 | 25.024 |
| Co | 73.7% | 18.944 | 24.000 | 26.000 |
| Al | 52.6% | 3.522 | 8.815 | 21.700 |

Strongest positive-only log correlations:

- Fe--Co: r=0.936, n=14
- Cr--Co: r=0.792, n=14
- Fe--Cr: r=0.747, n=16
- Fe--Al: r=-0.714, n=9
- Ni--Al: r=-0.503, n=8

## nicrmo_alloy

Dominant elements: Ni 17/17.

Visible 17-element sum P05/median/P95: 99.99 / 100.00 / 100.00 wt.%.

| Element | Presence | Positive P05 | Positive median | Positive P95 |
|---|---:|---:|---:|---:|
| Ni | 100.0% | 55.734 | 66.670 | 76.002 |
| Cr | 100.0% | 10.328 | 19.040 | 29.820 |
| Mo | 100.0% | 5.112 | 11.800 | 21.452 |

Strongest positive-only log correlations:

- Cr--Ni: r=-0.668, n=17
- Ni--Mo: r=-0.435, n=17
- Cr--Mo: r=-0.334, n=17

## other

Dominant elements: Ni 5/13, Cr 3/13, Mo 3/13, Fe 2/13.

Visible 17-element sum P05/median/P95: 98.74 / 100.00 / 100.00 wt.%.

| Element | Presence | Positive P05 | Positive median | Positive P95 |
|---|---:|---:|---:|---:|
| Cr | 61.5% | 15.175 | 20.005 | 89.500 |
| Ni | 46.2% | 39.100 | 66.900 | 95.000 |
| Fe | 46.2% | 13.554 | 32.986 | 84.787 |
| Mo | 38.5% | 0.990 | 62.060 | 94.406 |

## Generator interpretation

- Fe and Al have enough distinct templates for dataset-derived family ranges.
- HEA and Ni--Cr--Mo can be included as limited-data pilot families, but their ranges should not be treated as precise population bounds.
- `other` is too small and chemically mixed for one learned family distribution; exclude it initially or split it only after adding data.
- A future generator should model element presence separately from positive amount. Sampling only continuous ranges would make alloys too dense.
- Dedicated Al/HEA EPIT target rules are not justified by this composition-only analysis; this report supports composition generation only.
