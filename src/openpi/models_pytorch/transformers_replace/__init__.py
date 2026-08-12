"""Pi0.5 model code, rebased onto transformers 4.57.

These modules used to be copied over the installed ``transformers`` package
at install time. Importing them instead means nothing shadows
``transformers/``, so this can share an environment with projects that need
a different transformers version.
"""
