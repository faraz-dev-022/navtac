#!/usr/bin/env python3
"""Verify the database connection using the project's DATABASE_URL."""

import os
import sys

# Use the exact connection string the user requested
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg2://navtac:navtac@localhost:5432/navtac",
)

def main() -> int:
    try:
        from sqlalchemy import create_engine
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            value = result.scalar()
            if value != 1:
                print(f"Unexpected result: {value}")
                return 1
            print("connected")
            return 0
    except ImportError as exc:
        print(f"Missing dependency: {exc}", file=sys.stderr)
        print("Install requirements first: pip install -r requirements.txt", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"connection failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
