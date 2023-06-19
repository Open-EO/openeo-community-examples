import pandas as pd
from openeo.udf.debug import inspect

def udf_apply_feature_dataframe(df: pd.DataFrame) -> float:
    inspect(message="logging from UDF!!",data=df)
    return 123.456