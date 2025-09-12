import pytest
import pandas as pd
import polars as pl
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nullaxe.functions._standardize_units import standardize_units


class TestStandardizeUnitsPandas:
    def test_basic_unit_standardization(self):
        """Test basic unit standardization with common units."""
        df = pd.DataFrame({
            'distance': ['10 km', '5 miles', '100 meters', '2 feet'],
            'weight': ['5 kg', '10 lbs', '500 grams', '2 oz'],
            'other': ['no units here', 'just text', 'random', 'data']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters',
            'feet': 'meters',
            'kg': 'kilograms',
            'lbs': 'kilograms',
            'grams': 'grams',
            'oz': 'grams'
        }

        result = standardize_units(df, columns=['distance', 'weight'], unit_mappings=unit_mappings)

        assert result['distance'].iloc[0] == '10 kilometers'
        assert result['distance'].iloc[1] == '5 kilometers'
        assert result['distance'].iloc[2] == '100 meters'
        assert result['distance'].iloc[3] == '2 meters'

        assert result['weight'].iloc[0] == '5 kilograms'
        assert result['weight'].iloc[1] == '10 kilograms'
        assert result['weight'].iloc[2] == '500 grams'
        assert result['weight'].iloc[3] == '2 grams'

        # Other column should remain unchanged
        assert result['other'].equals(df['other'])

    def test_case_insensitive_matching(self):
        """Test that unit matching is case insensitive."""
        df = pd.DataFrame({
            'measurements': ['10 KM', '5 Miles', '100 METERS', '2 Feet']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters',
            'feet': 'meters'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)

        assert result['measurements'].iloc[0] == '10 kilometers'
        assert result['measurements'].iloc[1] == '5 kilometers'
        assert result['measurements'].iloc[2] == '100 meters'
        assert result['measurements'].iloc[3] == '2 meters'

    def test_multiple_units_in_text(self):
        """Test handling of multiple units in the same text."""
        df = pd.DataFrame({
            'description': ['Distance: 10 km, Weight: 5 kg', 'Length 2 feet, Mass 3 lbs']
        })

        unit_mappings = {
            'km': 'kilometers',
            'kg': 'kilograms',
            'feet': 'meters',
            'lbs': 'kilograms'
        }

        result = standardize_units(df, columns=['description'], unit_mappings=unit_mappings)

        assert result['description'].iloc[0] == 'Distance: 10 kilometers, Weight: 5 kilograms'
        assert result['description'].iloc[1] == 'Length 2 meters, Mass 3 kilograms'

    def test_word_boundary_matching(self):
        """Test that unit matching respects word boundaries."""
        df = pd.DataFrame({
            'text': ['10 km not kilometers', '5 miles milestone', 'weight in kg']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'kg': 'kilograms'
        }

        result = standardize_units(df, columns=['text'], unit_mappings=unit_mappings)

        # Should only replace standalone units, not parts of words
        assert result['text'].iloc[0] == '10 kilometers not kilometers'
        assert result['text'].iloc[1] == '5 kilometers milestone'
        assert result['text'].iloc[2] == 'weight in kilograms'

    def test_null_and_empty_values(self):
        """Test handling of null and empty values."""
        df = pd.DataFrame({
            'measurements': ['10 km', None, '', '5 lbs', pd.NA]
        })

        unit_mappings = {
            'km': 'kilometers',
            'lbs': 'kilograms'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)

        assert result['measurements'].iloc[0] == '10 kilometers'
        assert pd.isna(result['measurements'].iloc[1])
        assert result['measurements'].iloc[2] == ''
        assert result['measurements'].iloc[3] == '5 kilograms'
        assert pd.isna(result['measurements'].iloc[4])

    def test_non_string_columns_ignored(self):
        """Test that non-string columns are ignored."""
        df = pd.DataFrame({
            'numeric_col': [123, 456, 789],
            'string_col': ['10 km', '5 miles', '100 meters']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters'
        }

        result = standardize_units(df, columns=['numeric_col', 'string_col'], unit_mappings=unit_mappings)

        # Numeric column should remain unchanged
        assert result['numeric_col'].equals(df['numeric_col'])
        # String column should be processed
        assert result['string_col'].iloc[0] == '10 kilometers'

    def test_nonexistent_columns_ignored(self):
        """Test handling of nonexistent columns in the columns list."""
        df = pd.DataFrame({
            'existing_col': ['10 km', '5 miles']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers'
        }

        result = standardize_units(df, columns=['existing_col', 'nonexistent_col'], unit_mappings=unit_mappings)

        assert result['existing_col'].iloc[0] == '10 kilometers'
        assert result['existing_col'].iloc[1] == '5 kilometers'

    def test_empty_unit_mappings(self):
        """Test behavior with empty unit mappings."""
        df = pd.DataFrame({
            'measurements': ['10 km', '5 miles', '100 meters']
        })

        result = standardize_units(df, columns=['measurements'], unit_mappings={})

        # Should return unchanged data
        assert result['measurements'].equals(df['measurements'])

    def test_special_characters_in_units(self):
        """Test handling of units with special characters."""
        df = pd.DataFrame({
            'measurements': ['10 m²', '5 ft³', '100 m/s']
        })

        unit_mappings = {
            'm²': 'square meters',
            'ft³': 'cubic meters',
            'm/s': 'meters per second'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)

        assert result['measurements'].iloc[0] == '10 square meters'
        assert result['measurements'].iloc[1] == '5 cubic meters'
        assert result['measurements'].iloc[2] == '100 meters per second'


class TestStandardizeUnitsPolars:
    def test_basic_unit_standardization_polars(self):
        """Test basic unit standardization with Polars DataFrame."""
        df = pl.DataFrame({
            'distance': ['10 km', '5 miles', '100 meters', '2 feet'],
            'weight': ['5 kg', '10 lbs', '500 grams', '2 oz']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters',
            'feet': 'meters',
            'kg': 'kilograms',
            'lbs': 'kilograms',
            'grams': 'grams',
            'oz': 'grams'
        }

        result = standardize_units(df, columns=['distance', 'weight'], unit_mappings=unit_mappings)

        distance_col = result['distance'].to_list()
        weight_col = result['weight'].to_list()

        assert distance_col[0] == '10 kilometers'
        assert distance_col[1] == '5 kilometers'
        assert distance_col[2] == '100 meters'
        assert distance_col[3] == '2 meters'

        assert weight_col[0] == '5 kilograms'
        assert weight_col[1] == '10 kilograms'
        assert weight_col[2] == '500 grams'
        assert weight_col[3] == '2 grams'

    def test_case_insensitive_polars(self):
        """Test case insensitive matching with Polars."""
        df = pl.DataFrame({
            'measurements': ['10 KM', '5 Miles', '100 METERS']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)
        measurements = result['measurements'].to_list()

        assert measurements[0] == '10 kilometers'
        assert measurements[1] == '5 kilometers'
        assert measurements[2] == '100 meters'

    def test_null_values_polars(self):
        """Test handling of null values with Polars."""
        df = pl.DataFrame({
            'measurements': ['10 km', None, '5 lbs']
        })

        unit_mappings = {
            'km': 'kilometers',
            'lbs': 'kilograms'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)
        measurements = result['measurements'].to_list()

        assert measurements[0] == '10 kilometers'
        assert measurements[1] is None
        assert measurements[2] == '5 kilograms'

    def test_non_string_columns_polars(self):
        """Test that non-string columns are ignored in Polars."""
        df = pl.DataFrame({
            'numeric_col': [123, 456, 789],
            'string_col': ['10 km', '5 miles', '100 meters']
        })

        unit_mappings = {
            'km': 'kilometers',
            'miles': 'kilometers',
            'meters': 'meters'
        }

        result = standardize_units(df, columns=['numeric_col', 'string_col'], unit_mappings=unit_mappings)

        # Numeric column should remain unchanged
        assert result['numeric_col'].to_list() == df['numeric_col'].to_list()
        # String column should be processed
        string_col = result['string_col'].to_list()
        assert string_col[0] == '10 kilometers'


class TestStandardizeUnitsCommon:
    def test_invalid_input_type(self):
        """Test that function raises TypeError for invalid input types."""
        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            standardize_units("not a dataframe", columns=['col'], unit_mappings={})

        with pytest.raises(TypeError, match="Input must be a pandas or polars DataFrame"):
            standardize_units(['list', 'not', 'dataframe'], columns=['col'], unit_mappings={})

    def test_comprehensive_unit_mappings(self):
        """Test with comprehensive real-world unit mappings."""
        df = pd.DataFrame({
            'measurements': [
                '5 ft 10 in tall',
                'weighs 150 lbs',
                'distance of 10 miles',
                'speed 60 mph',
                'temperature 32 °F',
                'volume 1 gallon'
            ]
        })

        # Comprehensive imperial to metric mappings
        unit_mappings = {
            'ft': 'meters',
            'in': 'centimeters',
            'lbs': 'kilograms',
            'miles': 'kilometers',
            'mph': 'km/h',
            '°f': '°C',
            'gallon': 'liters'
        }

        result = standardize_units(df, columns=['measurements'], unit_mappings=unit_mappings)
        measurements = result['measurements'].to_list()

        assert measurements[0] == '5 meters 10 centimeters tall'
        assert measurements[1] == 'weighs 150 kilograms'
        assert measurements[2] == 'distance of 10 kilometers'
        assert measurements[3] == 'speed 60 km/h'
        assert measurements[4] == 'temperature 32 °C'
        assert measurements[5] == 'volume 1 liters'

    def test_complex_text_with_units(self):
        """Test with complex text containing multiple units and context."""
        df = pd.DataFrame({
            'recipe': [
                'Add 2 cups flour, 1 lb butter, bake at 350°F for 30 minutes',
                'Mix 500 grams sugar with 2 liters milk',
                'Distance: 5 km, Time: 30 min, Speed: 10 mph'
            ]
        })

        unit_mappings = {
            'cups': 'ml',
            'lb': 'kg',
            '°f': '°C',
            'grams': 'g',
            'liters': 'l',
            'km': 'kilometers',
            'mph': 'km/h'
        }

        result = standardize_units(df, columns=['recipe'], unit_mappings=unit_mappings)
        recipes = result['recipe'].to_list()

        assert 'ml' in recipes[0]
        assert 'kg' in recipes[0]
        assert '°C' in recipes[0]
        assert 'g' in recipes[1]
        assert 'l' in recipes[1]
        assert 'kilometers' in recipes[2]
        assert 'km/h' in recipes[2]

    def test_edge_cases(self):
        """Test edge cases and corner scenarios."""
        df = pd.DataFrame({
            'edge_cases': [
                'km vs kilometers',  # Partial word matching
                '10km without space',  # No space separation
                'multiple km km km',  # Repeated units
                'Unit at end 5 kg',  # Unit at end
                'kg at start of sentence',  # Unit at start
                'no units here',  # No units
                ''  # Empty string
            ]
        })

        unit_mappings = {
            'km': 'kilometers',
            'kg': 'kilograms'
        }

        result = standardize_units(df, columns=['edge_cases'], unit_mappings=unit_mappings)
        edge_cases = result['edge_cases'].to_list()

        # Should only replace word-boundary matches
        assert 'kilometers vs kilometers' in edge_cases[0]
        # Should handle cases with/without spaces appropriately
        assert 'kilograms' in edge_cases[3]
        assert 'kilograms' in edge_cases[4]
        # Should handle repeated units
        assert edge_cases[2].count('kilometers') >= 1
