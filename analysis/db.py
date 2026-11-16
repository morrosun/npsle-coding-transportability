# -*- coding: utf-8 -*-
"""PostgreSQL connection parameters for the extraction and analysis scripts.

No credential is stored in this repository.  Set the connection details in the
environment before running any script that reads a database:

    NPSLE_PG_HOST      default "localhost"
    NPSLE_PG_PORT      default "5432"
    NPSLE_PG_USER      default "postgres"
    NPSLE_PG_PASSWORD  no default, must be set

Example (PowerShell):
    $env:NPSLE_PG_PASSWORD = "***"
    python analysis/extract_cohort.py mimiciv eicu nwicu

Example (bash):
    export NPSLE_PG_PASSWORD=***
    python analysis/extract_cohort.py mimiciv eicu nwicu

`params()` returns host/port/user/password only, so it can be splatted into a
call that names the database itself: `psycopg2.connect(dbname="eicu", **db.params())`.
"""
import os


def params():
    """Connection keyword arguments except `dbname`."""
    return dict(
        host=os.environ.get("NPSLE_PG_HOST", "localhost"),
        port=int(os.environ.get("NPSLE_PG_PORT", "5432")),
        user=os.environ.get("NPSLE_PG_USER", "postgres"),
        password=os.environ.get("NPSLE_PG_PASSWORD", ""),
    )


def params_for(dbname):
    """Connection keyword arguments including `dbname`."""
    p = params()
    p["dbname"] = dbname
    return p
