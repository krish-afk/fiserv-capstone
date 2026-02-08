# pull_data.py

import os
import time
from pathlib import Path
from typing import Dict
import yaml
from dotenv import load_dotenv
from fredapi import Fred
import pandas as pd


def load_config(config_path: str = "config_local.yaml") -> dict:
    """Load configuration from YAML file."""
    # If config_path is relative, look for it relative to the project root
    if not os.path.isabs(config_path):
        # Get the directory containing this script
        script_dir = Path(__file__).parent
        # Go up one level to project root
        project_root = script_dir.parent
        config_path = project_root / config_path

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_api_key(env_var: str = "FRED_API_KEY") -> str:
    """Load API key from .env file."""
    load_dotenv()
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} not found in .env file")
    return api_key


def fetch_series_data(
    api_key: str,
    series_code: str,
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    sleep_between: float = 0.7
) -> pd.DataFrame:
    """
    Fetch time series data from FRED API.

    Args:
        api_key: FRED API key
        series_code: FRED series code to fetch
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Number of retry attempts on failure
        sleep_between: Seconds to wait between retries

    Returns:
        DataFrame with Date and Value columns, or empty DataFrame on failure
    """
    fred = Fred(api_key=api_key)

    exc = None
    for attempt in range(1, max_retries + 1):
        try:
            series = fred.get_series(
                series_code,
                observation_start=start_date,
                observation_end=end_date
            )
            series.index = pd.to_datetime(series.index)

            # Convert to DataFrame
            df = series.to_frame('Value')
            df = df.reset_index().rename(columns={'index': 'Date'})

            return df

        except Exception as e:
            exc = e
            if attempt < max_retries:
                time.sleep(sleep_between)
            else:
                print(f"[SKIP] {series_code} after {max_retries} attempts: {e}")
                return pd.DataFrame(columns=['Date', 'Value'])

    return pd.DataFrame(columns=['Date', 'Value'])


def fetch_multiple_series(
    api_key: str,
    series_dict: Dict[str, Dict[str, str]],
    start_date: str,
    end_date: str,
    max_retries: int = 3,
    sleep_between: float = 0.7
) -> Dict[str, Dict[str, any]]:
    """
    Fetch multiple time series from FRED API.

    Args:
        api_key: FRED API key
        series_dict: Dictionary mapping series codes to config dicts with 'name' and 'series_id'
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_retries: Number of retry attempts on failure
        sleep_between: Seconds to wait between retries

    Returns:
        Dictionary mapping series codes to dicts containing 'data', 'name', and 'series_id'
    """
    results = {}

    for series_code, series_config in series_dict.items():
        column_name = series_config['name']
        series_id = series_config['series_id']

        print(f"Fetching {series_code} -> {column_name}...")
        df = fetch_series_data(
            api_key=api_key,
            series_code=series_code,
            start_date=start_date,
            end_date=end_date,
            max_retries=max_retries,
            sleep_between=sleep_between
        )

        if not df.empty:
            # Rename Value column to the specified column name
            df = df.rename(columns={'Value': column_name})
            results[series_code] = {
                'data': df,
                'name': column_name,
                'series_id': series_id
            }

    return results


def generate_filename(data_source: str, series_name: str, date_accessed: str) -> str:
    """
    Generate filename following the convention: DataSource_DataSeries_DateAccessed.csv

    Args:
        data_source: Data source identifier (e.g., 'fred')
        series_id: Series identifier (e.g., 'GS30')
        date_accessed: Date in YYYYMMDD format

    Returns:
        Filename string (e.g., 'fred_GS30_20260206.csv')
    """
    return f"{data_source}_{series_name}_{date_accessed}.csv"


def write_data_to_csv(
    data: pd.DataFrame,
    filename: str,
    data_folder: str = "data",
    index: bool = False
) -> str:
    """
    Write DataFrame to CSV file in specified folder.

    Args:
        data: DataFrame to write
        filename: Name of the output file
        data_folder: Folder to save the file (relative to project root)
        index: Whether to include index in CSV

    Returns:
        Full path to the saved file
    """
    # If data_folder is relative, resolve it relative to project root
    if not os.path.isabs(data_folder):
        # Get the directory containing this script
        script_dir = Path(__file__).parent
        # Go up one level to project root
        project_root = script_dir.parent
        data_path = project_root / data_folder
    else:
        data_path = Path(data_folder)

    # Create data folder if it doesn't exist
    data_path.mkdir(parents=True, exist_ok=True)

    # Construct full file path
    file_path = data_path / filename

    # Write to CSV
    data.to_csv(file_path, index=index)
    print(f"Data saved to {file_path}")

    return str(file_path)


def main():
    """Main function to fetch FRED data and save to CSV."""
    # Load configuration
    config = load_config()

    # Load API key from .env
    api_key = load_api_key("FRED_API_KEY")

    # Extract FRED configuration
    fred_config = config['fred']
    output_config = config['output']

    # Get data source and date accessed for filename generation
    data_source = fred_config.get('data_source', 'fred')
    date_accessed = fred_config.get('date_accessed', time.strftime('%Y%m%d'))

    # Fetch all series
    series_data = fetch_multiple_series(
        api_key=api_key,
        series_dict=fred_config['series'],
        start_date=fred_config['start_date'],
        end_date=fred_config['end_date'],
        max_retries=fred_config['max_retries'],
        sleep_between=fred_config['sleep_between_retries']
    )

    if series_data:
        # Save individual series files if configured
        if output_config.get('save_individual_files', True):
            print("\nSaving individual series files...")
            for series_info in series_data.values():
                filename = generate_filename(
                    data_source=data_source,
                    series_name=series_info['name'],
                    date_accessed=date_accessed
                )
                write_data_to_csv(
                    data=series_info['data'],
                    filename=filename,
                    data_folder=output_config['data_folder']
                )

        # Save merged file if configured
        if output_config.get('save_merged_file', False):
            print("\nSaving merged file...")
            merged_data = None
            for series_info in series_data.values():
                if merged_data is None:
                    merged_data = series_info['data']
                else:
                    merged_data = merged_data.merge(
                        series_info['data'],
                        on='Date',
                        how='outer'
                    )

            # Sort by date
            merged_data = merged_data.sort_values('Date').reset_index(drop=True)

            # Save to CSV
            write_data_to_csv(
                data=merged_data,
                filename=output_config['merged_filename'],
                data_folder=output_config['data_folder']
            )

            print(f"Date range: {merged_data['Date'].min()} to {merged_data['Date'].max()}")
            print(f"Total rows: {len(merged_data)}")

        print(f"\nSuccessfully fetched and saved {len(series_data)} series")
    else:
        print("No data was successfully fetched.")


if __name__ == "__main__":
    main()
