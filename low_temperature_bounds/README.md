# Bounds on energy cells at low temperature

`low_temperature_bounds.py` evaluates the bounds of Section 4.4 of the article
that are quoted in Remark 4.13 and in Campaign E. It prints them and writes no
files.

```bash
python low_temperature_bounds.py
```

For the orthogonal instance of Campaigns A-C and G (dimension 64, four
patterns of norm 8, inverse temperature 0.15), it prints the radius `rbar` of
the inscribed ball, the lower bound on the escape gap of Proposition 4.11(c),
the largest escape gap certified by Corollary 4.6, and the upper bound given by
the energy of a pair of patterns at the midpoint of their segment. For each
minimum of the planar landscape of Campaign E (inverse temperature 1.5), it
prints the escape gap given by the interval-certified escape level, the lower
bound of Proposition 4.11(c), the excess of the maximum of the energy on the
segment of Proposition 4.11(d) over the certified level, and the first-order
level of Remark 4.13. The certified levels are those enclosed by
`campaign_E_audit/certify_planar_critical_points.py`.

Tested with CPython 3.12.3 and NumPy 2.3.5.
