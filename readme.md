# Financial Statement Analyzer & Company Valuator

A Python-based tool that analyzes company financial statements from annual reports and performs company valuations using the multiples approach (EV/EBITDA and EV/EBIT).

Recent breakthroughs in Optical Character Recognition (OCR) and Large Language Models have revolutionized how we can process and analyze complex documents. Inspired by these advancements, this project aims to automate the challenging task of financial statement analysis and company valuation - a process traditionally done manually by financial analysts and investors.

This tool leverages state-of-the-art OCR technology to extract financial data from annual reports, combining it with intelligent processing to perform accurate company valuations. By automating this traditionally time-consuming process, we're making financial analysis more accessible and efficient for investors, analysts, and financial professionals.


## Overview

This project automates the process of:
1. Extracting financial data from annual reports (PDF format)
2. Processing and analyzing financial statements
3. Calculating company valuations using industry multiples
4. Generating detailed valuation reports

Currently implements:
- Multiple-based valuation approach
- CLI interface
- Financial data extraction from PDFs
- Industry-specific multiple calculations based on Damodaran data

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd [repo-name]
```

2. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

### Basic Command

Process a financial statement PDF:
```bash
python -m backend.cli -i your_financial_statement.pdf
```

### Output

The system generates several files in the `temp_cli_results` folder:
- Extracted financial data (JSON format)
- OCR processing results (HTML)
- Valuation calculations
- Final report (DOCX format)

### Options

```bash
# Process with custom output location
python -m backend.cli -i input.pdf -o path/to/output.docx


### Example Usage

```bash
# Process a specific financial statement
python -m backend.cli -i vz_2023_isotra.pdf
# The report will be saved as 'vz_2023_isotra_report.docx' in the same directory


# Display help
python -m backend.cli --help
```


### Output Behavior

- If no output path (-o) is specified, the report will be saved in the same directory as the input file with '_report.docx' suffix
- If output path is specified, the report will be saved at that location
- The system will create any necessary directories in the output path



## Project Structure

```
backend/
├── src/
│   ├── valuation/    # Valuation logic
│   ├── financials/   # Financial data processing
│   └── ocr/          # Document processing
├── temp_cli_results/ # Temporary processing files
└── requirements.txt
```

## Features

- PDF financial statement processing
- Financial data extraction
- Industry-specific multiple calculations
- Valuation report generation
- Temporary results storage for analysis

## Limitations

- Currently supports only multiples-based valuation (no DCF)
- CLI interface only (no frontend)
- Focused on standard financial statement formats

## Future Development

Planned features:
- DCF valuation implementation
- Web interface
- Additional valuation methods
- Enhanced report customization
- API endpoints

