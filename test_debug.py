import sys
sys.path.insert(0, 'src')

import pandas as pd
from nullaxe.functions._standardize_units import standardize_units

# Recreate the ACTUAL test scenario with degree symbols
df = pd.DataFrame({
    'measurements': [
        '5 ft 10 in tall',
        'weighs 150 lbs',
        'distance of 10 miles',
        'speed 60 mph',
        'temperature 32 °F',  # This has degree symbol!
        'volume 1 gallon'
    ]
})

unit_mappings = {
    'ft': 'meters',
    'in': 'centimeters',
    'lbs': 'kilograms',
    'miles': 'kilometers',
    'mph': 'km/h',
    '°f': '°C',  # This also has degree symbol!
    'gallon': 'liters'
}

print("Input DataFrame:")
print(df)
print("\nUnit mappings:")
print(unit_mappings)

result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)
measurements = result['measurements'].to_list()

print("\nResults:")
for i, measurement in enumerate(measurements):
    print(f"{i}: '{measurement}'")

print(f"\nProblem case:")
print(f"Expected: 'temperature 32 °C'")
print(f"Actual:   '{measurements[4]}'")
print(f"Match: {measurements[4] == 'temperature 32 °C'}")

# Check if °F is still there
print(f"Contains °F: {'°F' in measurements[4]}")
print(f"Contains °C: {'°C' in measurements[4]}")

# Test the regex pattern directly
import re
text = 'temperature 32 °F'
pattern = re.compile(r'\b(' + '|'.join(re.escape(unit) for unit in unit_mappings.keys()) + r')\b', re.IGNORECASE)
print(f"\nRegex pattern: {pattern.pattern}")
print(f"Matches found: {pattern.findall(text)}")
