import re

# Test the pattern construction step by step
unit_mappings = {'ft': 'meters', 'in': 'centimeters', 'lbs': 'kilograms'}

escaped_units = [re.escape(unit) for unit in unit_mappings.keys()]
print('Escaped units:', escaped_units)

pattern_string = escaped_units[0]
print('Initial pattern_string:', repr(pattern_string))

for unit in escaped_units[1:]:
    old_pattern = pattern_string
    pattern_string = pattern_string + '|' + unit
    print(f'Adding {unit}: {repr(old_pattern)} + "|" + {repr(unit)} = {repr(pattern_string)}')

print('Final pattern_string:', repr(pattern_string))
full_pattern = r'\b(' + pattern_string + r')\b'
print('Full pattern:', repr(full_pattern))

# Test the compiled pattern
compiled_pattern = re.compile(full_pattern, re.IGNORECASE)
print('Compiled pattern:', compiled_pattern.pattern)

# Test matching
test_text = "5 ft 10 in tall"
matches = compiled_pattern.findall(test_text)
print('Test text:', repr(test_text))
print('Matches:', matches)
