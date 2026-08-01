import numpy as np
import pandas as pd
import sys

for fname in sys.argv[1:]:
    df = pd.read_parquet(fname)
    print(df)
    for c in df.columns:
        print(c, df[c].unique())
