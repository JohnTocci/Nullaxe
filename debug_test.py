import pandas as pd
import polars as pl
import re
from typing import Union, List

# Test the function directly
def test_standardize_units():
    df = pd.DataFrame({
        'measurements': ['temperature 32 F']
    })

    unit_mappings = {'f': 'C'}
    unit_mappings_lower = {k.lower(): v for k, v in unit_mappings.items()}
    print("Unit mappings:", unit_mappings)
    print("Unit mappings lower:", unit_mappings_lower)

    # Create regex pattern
    unit_pattern = re.compile(r'\b(' + '|'.join(re.escape(unit) for unit in unit_mappings.keys()) + r')\b', re.IGNORECASE)
    print("Pattern:", unit_pattern.pattern)

    text = 'temperature 32 F'
    print("Original text:", text)
    print("Matches found:", unit_pattern.findall(text))

    # Test the replace function
    def replace_units(text):
        if pd.isna(text) or text is None:
            return text
        result = unit_pattern.sub(lambda match: unit_mappings_lower[match.group(0).lower()], str(text))
        print(f"Replacing '{text}' -> '{result}'")
        return result

    # Apply to DataFrame
    df_copy = df.copy()
    for col in ['measurements']:
        if col in df_copy.columns and df_copy[col].dtype in ['object', 'string']:
            df_copy[col] = df_copy[col].apply(replace_units)

    print("Final result:", df_copy['measurements'].iloc[0])
    return df_copy

if __name__ == "__main__":
    test_standardize_units()
