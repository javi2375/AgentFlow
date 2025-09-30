import pandas as pd
import os


def validate_result_column(df, column_name='result'):
    """
    Validates the 'result' column in a DataFrame:
    - Must be either a string or a list of strings.
    - Reports how many are lists, and how many lists have multiple elements.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in DataFrame. Available columns: {list(df.columns)}")

    # Initialize counters
    total_rows = len(df)
    string_count = 0
    list_of_strings_count = 0
    list_with_multiple_elements = 0
    invalid_type_count = 0
    non_string_in_list_count = 0

    print(f"Validating '{column_name}' column in DataFrame with {total_rows} rows...\n")

    for idx, value in df[column_name].items():
        if isinstance(value, str):
            string_count += 1
        elif isinstance(value, list):
            list_of_strings_count += 1
            if len(value) > 1:
                list_with_multiple_elements += 1

            # Check if all elements in the list are strings
            non_strings = [item for item in value if not isinstance(item, str)]
            if len(non_strings) > 0:
                print(f"❌ Row {idx}: List contains non-string elements: {non_strings}")
                non_string_in_list_count += 1
        else:
            print(f"❌ Row {idx}: Invalid type {type(value).__name__}: {value}")
            invalid_type_count += 1

    # Summary
    print("=" * 60)
    print("🔍 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total rows: {total_rows}")
    print(f"✅ String values: {string_count}")
    print(f"✅ List of strings: {list_of_strings_count}")
    print(f"   ➕ Lists with multiple elements: {list_with_multiple_elements}")
    print(f"⚠️  Lists with non-string elements: {non_string_in_list_count}")
    print(f"❌ Invalid types (not str/list): {invalid_type_count}")

    # Final result
    if invalid_type_count == 0 and non_string_in_list_count == 0:
        print("🎉 All entries in the 'result' column are valid.")
    else:
        print("❌ Some entries in the 'result' column are invalid. See above for details.")

    return {
        'total_rows': total_rows,
        'string_count': string_count,
        'list_of_strings_count': list_of_strings_count,
        'list_with_multiple_elements': list_with_multiple_elements,
        'invalid_type_count': invalid_type_count,
        'non_string_in_list_count': non_string_in_list_count,
        'is_valid': (invalid_type_count == 0 and non_string_in_list_count == 0)
    }


def check_parquet_file_fully(file_path, num_rows=5):
    """
    Reads a parquet file, prints first few rows fully, and validates the 'result' column.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please make sure you have run the data generation script first.")
        return None

    print(f"📄 Reading parquet file: {file_path}")
    try:
        df = pd.read_parquet(file_path)
        print(f"✅ Loaded DataFrame with {len(df)} rows and {len(df.columns)} columns.")
        print(f"📊 Columns: {list(df.columns)}")

        # Display first few rows fully
        print(f"\n--- First {num_rows} rows (full content) ---")
        print(df.head(num_rows).to_string())

        # Validate the 'result' column
        print(f"\n--- Validating 'result' column ---")
        result = validate_result_column(df, 'result')

        return df, result

    except Exception as e:
        print(f"❌ An error occurred while reading the file: {e}")
        return None


if __name__ == '__main__':
    # Path to your parquet file
    parquet_file_path = 'data/val/aime24.parquet'

    # Run full check
    df, summary = check_parquet_file_fully(parquet_file_path, num_rows=10)