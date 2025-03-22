I was inspired by sam willison post on mistral amnd its impressive capabilities so i decided to make valuation analyst processing these documents



# Financial Valuation System

## CLI Usage

The system provides a command-line interface to process financial statements and generate valuation reports.

### Basic Commands

```bash
# Process a PDF and save report in the same directory
python -m backend.cli -i your_file.pdf

# Specify custom output location
python -m backend.cli -i your_file.pdf -o path/to/output/report.docx

# Get help
python -m backend.cli --help
```

### Example Usage

```bash
# Process a specific financial statement
python -m backend.cli -i vz_2023_isotra.pdf
# The report will be saved as 'vz_2023_isotra_report.docx' in the same directory
```

### Output Behavior

- If no output path (-o) is specified, the report will be saved in the same directory as the input file with '_report.docx' suffix
- If output path is specified, the report will be saved at that location
- The system will create any necessary directories in the output path

### Requirements

- Python 3.8+
- Required packages (specified in pyproject.toml)
- PDF file containing financial statements

### Notes

- Make sure you run the commands from the project root directory (where the 'backend' folder is located)
- The input PDF file should be accessible from your current working directory
- The system will log all processing steps to both console and a log file

This README provides:
* Basic command syntax
* Real example that worked for you
* Clear explanation of where files are saved
* Requirements and important notes
* Context for running the commands correctly










add fallbacks when no income statement tr something