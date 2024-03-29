#!/usr/bin/env python3

import sys
import pandas as pd
from describe import describe


def main():
    try:
        df = pd.read_csv(sys.argv[1])
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")
    print(df.describe())
    print(describe(df))


if __name__ == "__main__":
    main()
