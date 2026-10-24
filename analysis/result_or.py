# -*- coding: utf-8 -*-
"""Read the adjusted OR from the result tables by scanning each cell for the
'number (number-number)' pattern, instead of trusting a column index or a
header string. The headers are Chinese and their encoding differs between
readers, so both are unreliable.
"""
import pathlib
import re

import pandas as pd


def _or(txt):
    # the CI separator is an en dash in these tables, but accept any dash,
    # an HTML entity, or a plain hyphen
    m = re.search(r"([0-9]+\.?[0-9]*)\s*\(?\s*([0-9]+\.?[0-9]*)\s*"
                  r"(?:&ndash;|&#8211;|[-–—])\s*([0-9]+\.?[0-9]*)\s*\)?",
                  str(txt))
    if not m:
        return None
    return (float(m.group(1)), float(m.group(2)), float(m.group(3)))


def adjusted_or(path, db=None, contrast=None):
    """Adjusted OR (95% CI) for one row, located by content.

    db matches the first column exactly; contrast is searched across all
    columns, because the pooled row is identified by its label in one table and
    by its contrast in another.
    """
    r = pd.read_csv(path)
    c0 = r.columns[0]
    hit = r
    if db is not None:
        hit = hit[hit[c0].astype(str).str.strip() == db]
    if contrast is not None:
        mask = pd.Series(False, index=hit.index)
        for c in hit.columns:
            mask |= hit[c].astype(str).str.contains(contrast, regex=False)
        hit = hit[mask]
    if hit.empty:
        raise SystemExit("no row for db=%r contrast=%r in %s"
                         % (db, contrast, path))
    row = hit.iloc[0]
    # the adjusted OR is the only cell that carries a full OR (95% CI) triple
    for v in row.values:
        got = _or(v)
        if got:
            return got
    raise SystemExit("no OR cell in row %r of %s" % (db, path))


def _pooled_label(path):
    """The label pandas shows for the pooled row is mojibake under some
    readers, so it is identified by the trailing 'k=2' instead of by name."""
    r = pd.read_csv(path)
    c0 = r.columns[0]
    for v in r[c0].astype(str):
        if "k=2" in v:
            return v.strip()
    raise SystemExit("no pooled row in %s" % path)


if __name__ == "__main__":
    OUT = pathlib.Path(__file__).resolve().parent.parent / "out"
    t22 = OUT / "t22a_first_sepsis.csv"
    t25 = OUT / "t25a_tier_with_pooled.csv"
    print("pooled sepsis   :", adjusted_or(t22, contrast="k=2"))
    # the pooled row is the one whose first column carries the pooled label;
    # matching both the database and the contrast keeps it from picking the
    # first per-database row
    print("pooled Tier C   :",
          adjusted_or(t25, db=_pooled_label(t25), contrast="Tier C"))
    for db in ("MIMIC-IV", "eICU-CRD"):
        print("%-9s Tier C:" % db, adjusted_or(t25, db=db, contrast="Tier C"))
