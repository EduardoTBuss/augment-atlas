import sys

from .cli import main

# Guarded: derive workers use the "spawn" start method, which re-imports the main module
# as ``__mp_main__``; an unguarded call would re-run the CLI inside every worker.
if __name__ == "__main__":
    sys.exit(main())
