"""Enable ``python -m cli`` as a synonym for ``python -m cli.main``.

The interactive ``ask`` / ``chat`` / ``personal`` surface is defined in
``cli.main``; this shim lets the documented and natural ``python -m cli``
invocation reach it without forcing every user to know about the inner
module path.
"""

from cli.main import main

if __name__ == "__main__":
    main()
