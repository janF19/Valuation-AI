import json
import requests
import os

from typing import Dict, Any, Optional, List, Tuple, Union

import re
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from difflib import SequenceMatcher
import openai
from backend.config.settings import settings
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FinancialExtractor:
    """
    Extract financial data from HTML reports.
    
    Attributes:
        financial_data (Dict[str, Any]): Extracted financial data
        client (Optional[openai.OpenAI]): OpenAI client instance
    
    Raises:
        ValueError: If invalid input is provided
        RuntimeError: If critical extraction fails
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.financial_data = {}
        # Initialize OpenAI client
        if openai_api_key:
            self.client = openai.OpenAI(api_key=openai_api_key)
        else:
            # Use settings.OPENAI_API_KEY instead of direct import
            if settings.OPENAI_API_KEY:
                self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            else:
                self.client = None
                logger.warning("No OpenAI API key provided - LLM fallback strategies will be disabled")
    def extract_from_html(self, html_content: str) -> Dict[str, Any]:
        """Extract financial data from HTML content."""
        if not html_content or not isinstance(html_content, str):
            raise ValueError("Invalid HTML content provided")
        
        if len(html_content.strip()) == 0:
            raise ValueError("Empty HTML content provided")
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize with empty structures to prevent KeyErrors
            self.financial_data = {
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow': {},
                'information': {}
            }
            
            # Extract data from different financial statements
            try:
                self._extract_income_statement(soup)
            except Exception as e:
                logger.error(f"Error extracting income statement: {e}", exc_info=True)
            
            try:
                self._extract_balance_sheet(soup)
            except Exception as e:
                logger.error(f"Error extracting balance sheet: {e}", exc_info=True)
            
            try:
                self._extract_cash_flow(soup)
            except Exception as e:
                logger.error(f"Error extracting cash flow: {e}", exc_info=True)
            
            # Add company information extraction
            try:
                self.financial_data['information'] = self._extract_company_information(soup)
            except Exception as e:
                logger.error(f"Error extracting company information: {e}", exc_info=True)
                self.financial_data['information'] = {
                    'company_name': "Unknown Company",
                    'accounting_period': str(datetime.now().year)
                }
            
            # Validate required fields and apply fallbacks
            try:
                self._validate_and_fix_required_fields(soup)
            except Exception as e:
                logger.error(f"Error in validation and fixing: {e}", exc_info=True)
            
            # Perform final validation check
            try:
                self._perform_final_validation()
            except Exception as e:
                logger.error(f"Error in final validation: {e}", exc_info=True)
            
            return self.financial_data
            
        except Exception as e:
            logger.error(f"Critical error extracting financial data: {e}", exc_info=True)
            raise RuntimeError(f"Failed to extract financial data: {str(e)}")
    
    def _extract_income_statement(self, soup: BeautifulSoup) -> None:
        """Extract data from income statement using multiple fallback strategies."""
        # Strategy 1: Find by header (existing approach)
        income_header = soup.find('h1', string=re.compile('VÝKAZ ZISKU A ZTRÁT', re.IGNORECASE))
        income_table = None
        
        if income_header:
            income_table = self._find_table_after_header(income_header)
        
        # Strategy 2: Find by table header/caption if Strategy 1 fails
        if not income_table:
            for table in soup.find_all('table'):
                # Check table caption or first row
                caption = table.find('caption')
                if caption and re.search(r'VÝKAZ\s+ZISKU\s+A\s+ZTR[ÁA]T', caption.get_text(), re.IGNORECASE):
                    income_table = table
                    break
                    
                # Check first few rows for income statement identifier
                first_rows = table.find_all('tr')[:3]  # Check first 3 rows
                for row in first_rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        if re.search(r'VÝKAZ\s+ZISKU\s+A\s+ZTR[ÁA]T', cell.get_text(), re.IGNORECASE):
                            income_table = table
                            break
                    if income_table:
                        break
                if income_table:
                    break
        
        # Strategy 3: Find by key income statement line items
        if not income_table:
            key_items = [
                r'Tržby z prodeje výrobk[ůu]',
                r'Výkonová spotřeba',
                r'Osobní náklady',
                r'Provozní výsledek'
            ]
            
            for table in soup.find_all('table'):
                matches = 0
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        cell_text = cell.get_text().strip()
                        if any(re.search(pattern, cell_text, re.IGNORECASE) for pattern in key_items):
                            matches += 1
                
                # If we find at least 2 key items, this is likely the income statement
                if matches >= 2:
                    income_table = table
                    break
        
        # Strategy 4: Use table structure analysis
        if not income_table:
            for table in soup.find_all('table'):
                rows = table.find_all('tr')
                if len(rows) > 5:  # Income statement usually has many rows
                    # Check if table has typical income statement structure
                    # (item description followed by numbers)
                    first_row = rows[0].find_all(['td', 'th'])
                    if len(first_row) >= 3:  # Usually has description and at least 2 number columns
                        # Check if last columns contain numeric data
                        last_cells = [row.find_all(['td', 'th'])[-1] for row in rows[1:5]]
                        numeric_cells = sum(1 for cell in last_cells if 
                                         re.match(r'^[\s\d\-\(\)]+$', cell.get_text().strip()))
                        if numeric_cells >= 3:  # Most cells should be numeric
                            income_table = table
                            break
        
        if not income_table:
            logger.warning("Income statement table not found using any strategy")
            return
        
        # Extract the data using existing method
        income_data = self._extract_table_data(income_table)
        
        # Process the data even if validation fails, but log the warning
        if not self._validate_income_statement_data(income_data):
            logger.warning("Extracted income statement data appears invalid, but proceeding with processing")
        
        # Always try to process and store key metrics
        processed_data = self._process_income_statement(income_data)
        
        # Only store if we got some data
        if processed_data:
            self.financial_data['income_statement'] = processed_data
        else:
            logger.error("Failed to process income statement data")
        
       
            
            logger.error("All income statement extraction strategies failed LLM will be tried later")
    
    def _validate_income_statement_data(self, data: List[Dict[str, str]]) -> bool:
        """Validate that extracted data appears to be from income statement."""
        if not data:
            return False
        
        # Check for key income statement items
        key_terms = [
            (r'tržby|výnosy|revenue', 0.7),
            (r'spotřeba|náklady|costs', 0.7),
            (r'výsledek|zisk|profit', 0.7)
        ]
        
        matches = 0
        for row in data:
            text = row.get('TEXT', '').lower()
            for pattern, threshold in key_terms:
                if any(SequenceMatcher(None, text, term).ratio() > threshold 
                      for term in pattern.split('|')):
                    matches += 1
                    break
        
        return matches >= 2  # At least 2 key terms should be present
    
    def _extract_balance_sheet(self, soup: BeautifulSoup) -> None:
        """Extract data from balance sheet."""
        # Find the balance sheet section using fuzzy matching
        balance_header = self._find_header(soup, 'ROZVAHA')
        
        if not balance_header:
            logger.warning("Balance sheet not found in the document")
            return
            
        # Get the table after the header
        balance_table = self._find_table_after_header(balance_header)
        
        if not balance_table:
            logger.warning("Balance sheet table not found")
            return
            
        # Extract the data
        balance_data = self._extract_table_data(balance_table)
        
        # Process and store key metrics
        self.financial_data['balance_sheet'] = self._process_balance_sheet(balance_data)
    
    def _extract_cash_flow(self, soup: BeautifulSoup) -> None:
        """Extract data from cash flow statement."""
        # Find the cash flow section using fuzzy matching
        cash_flow_header = self._find_header(soup, 'PŘEHLED O PĚNĚŽNICH TOCÍCH')
        
        if not cash_flow_header:
            logger.warning("Cash flow statement not found in the document")
            return
            
        # Get the table after the header
        cash_flow_table = self._find_table_after_header(cash_flow_header)
        
        if not cash_flow_table:
            logger.warning("Cash flow table not found")
            return
            
        # Extract the data
        cash_flow_data = self._extract_table_data(cash_flow_table)
        
        # Process and store key metrics
        self.financial_data['cash_flow'] = self._process_cash_flow(cash_flow_data)
    
    def _find_table_after_header(self, header_element) -> Optional[Any]:
        """Find the first table after a given header element."""
        # First try direct next siblings
        element = header_element
        
        # Search through next 5 elements to find a table
        for _ in range(5):
            element = element.find_next()
            if element and element.name == 'table':
                return element
        
        # If no table found, try finding any table that follows this header
        next_tables = header_element.find_all_next('table', limit=1)
        if next_tables:
            return next_tables[0]
            
        return None
    
    def _extract_table_data(self, table) -> List[Dict[str, str]]:
        """Extract data from a table element."""
        rows = []
        
        try:
            # Get header row
            header_row = table.find('thead').find_all('th')
            headers = [header.text.strip() for header in header_row]
            
            # Check if this is a balance sheet table (has AKTIVA or PASIVA column)
            is_balance_sheet = any('AKTIVA' in h or 'PASIVA' in h for h in headers)
            
            if is_balance_sheet:
                # For balance sheet, use AKTIVA/PASIVA column and Netto columns
                text_col_index = next((i for i, h in enumerate(headers) if 'AKTIVA' in h or 'PASIVA' in h), 1)
                current_value_index = 5  # Netto (3) column
                previous_value_index = 6  # Netto (4) column
            else:
                # For other statements, use TEXT column and last two columns
                text_col_index = next((i for i, h in enumerate(headers) if 'TEXT' in h), 3)
                current_value_index = -2
                previous_value_index = -1
            
            # Get data rows
            data_rows = table.find('tbody').find_all('tr')
            
            for row in data_rows:
                cells = row.find_all('td')
                
                # Skip rows without enough cells
                if len(cells) < len(headers):
                    continue
                
                # Get the text and values
                text_value = cells[text_col_index].text.strip()
                
                try:
                    # Clean up the values
                    def clean_value(val: str) -> str:
                        # Remove special characters and spaces
                        val = val.strip().replace('$', '').replace(' ', '')
                        # Handle negative numbers in parentheses
                        if val.startswith('(') and val.endswith(')'):
                            val = '-' + val[1:-1]
                        # Handle other special characters
                        val = val.replace('−', '-')  # Replace minus sign
                        return val

                    current_value = clean_value(cells[current_value_index].text)
                    previous_value = clean_value(cells[previous_value_index].text)
                except IndexError:
                    continue
                
                if text_value and (current_value or previous_value):
                    row_data = {
                        'TEXT': text_value,
                        'Běžné účetní období': current_value,
                        'Minulé účetní období': previous_value
                    }
                    rows.append(row_data)
                    
        except Exception as e:
            logger.error(f"Error processing table: {str(e)}")
            logger.error(f"Headers found: {headers if 'headers' in locals() else 'No headers'}")
        
        return rows
    
    def _process_income_statement(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process and extract key metrics from income statement."""
        result = {}
        
        # Add debug logging
        logger.debug(f"Processing income statement data: {data[:2]}...")  # Show first 2 rows
        
        def similar(a: str, b: str, threshold: float = 0.75) -> bool:  # Lower threshold
            """Check if two strings are similar using sequence matcher."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
        
        # Map Czech terms to English financial terms with more variations
        term_mapping = {
            'Tržby z prodeje výrobkủ a služeb|Tržby z prodeje vlastních výrobků|Výnosy': 'revenue_from_products_and_services',
            'Tržby za prodej zboží|Tržby zboží': 'revenue_from_goods',
            'Výkonová spotřeba|Náklady výkonová': 'production_consumption',
            'Osobní náklady|Mzdové náklady': 'personnel_costs',
            'Mzdové náklady': 'wage_costs',
            'Provozní výsledek hospodaření|Provozní výsledek|Výsledek hospodaření provozní': 'operating_profit',
            'Výsledek hospodaření před zdaněním|Výsledek před zdaněním': 'profit_before_tax',
            'Výsledek hospodaření za účetní období|Výsledek za období|Čistý zisk': 'net_profit',
            'Úpravy hodnot v provozní oblasti|Odpisy': 'depreciation'
        }
        
        # Find the relevant columns for current and previous years
        current_year_column = None
        previous_year_column = None
        
        if data and len(data) > 0:
            sample_row = data[0]
            logger.debug(f"Sample row keys: {sample_row.keys()}")
            
            # Try multiple patterns for column identification
            current_patterns = ['běžné', 'běžném', 'current', '2023']
            previous_patterns = ['minulé', 'předchozí', 'previous', '2022']
            
            for key in sample_row.keys():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in current_patterns):
                    current_year_column = key
                elif any(pattern in key_lower for pattern in previous_patterns):
                    previous_year_column = key
        
        # Fallback to positional logic if needed
        if not current_year_column or not previous_year_column:
            numeric_columns = []
            for row in data[:5]:  # Check first 5 rows
                for key, value in row.items():
                    if value and re.match(r'^[\s\d\-\(\)]+$', str(value).strip()):
                        numeric_columns.append(key)
            
            numeric_columns = list(set(numeric_columns))  # Remove duplicates
            if len(numeric_columns) >= 2:
                current_year_column = numeric_columns[-2]
                previous_year_column = numeric_columns[-1]
        
        logger.debug(f"Selected columns - Current: {current_year_column}, Previous: {previous_year_column}")

        # Extract values for each financial metric
        for row in data:
            text_value = row.get('TEXT', '')
            if not text_value:
                continue
            
            # Try to match with our known terms using fuzzy matching
            for czech_terms, english_term in term_mapping.items():
                if any(similar(term, text_value) for term in czech_terms.split('|')):
                    for year_suffix, column in [('current', current_year_column), 
                                              ('previous', previous_year_column)]:
                        if column and column in row:
                            try:
                                value = row[column].replace(' ', '')
                                # Handle parentheses for negative numbers
                                if value.startswith('(') and value.endswith(')'):
                                    value = '-' + value[1:-1]
                                # Remove any remaining non-numeric characters except minus
                                value = re.sub(r'[^\d\-]', '', value)
                                if value:
                                    result[f"{english_term}_{year_suffix}"] = int(value)
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Could not convert value for {english_term}_{year_suffix}: {e}")
        
        # Log the result for debugging
        logger.debug(f"Processed income statement data: {result}")
        
        # Always set EBIT equal to operating profit if available
        if 'operating_profit_current' in result:
            result['ebit_current'] = result['operating_profit_current']
        if 'operating_profit_previous' in result:
            result['ebit_previous'] = result['operating_profit_previous']
        
        return result
    
    def _process_balance_sheet(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process and extract key metrics from balance sheet."""
        result = {}
        
        def similar(a: str, b: str, threshold: float = 0.80) -> bool:
            """Check if two strings are similar using sequence matcher."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
        
        # Map Czech terms to English financial terms
        term_mapping = {
            'AKTIVA CELKEM': 'total_assets',
            'Stálá aktiva': 'fixed_assets',
            'Dlouhodobý nehmotný majetek': 'intangible_assets',
            'Dlouhodobý hmotný majetek': 'tangible_assets',
            'Oběžná aktiva': 'current_assets',
            'Zásoby': 'inventory',
            'Pohledávky': 'receivables',
            'Peněžní prostředky': 'cash',
            'PASIVA CELKEM': 'total_liabilities_equity',
            'Vlastní kapitál': 'equity',
            'Základní kapitál': 'registered_capital',
            'Cizí zdroje': 'liabilities',
            'Rezervy': 'provisions',
            'Závazky': 'payables',
            'Dlouhodobé závazky': 'long_term_payables',
            'Krátkodobé závazky': 'short_term_payables'
        }
        
        # Extract values for each financial metric
        for row in data:
            text_value = row.get('TEXT', '')
            if not text_value:
                continue
            
            # Try to match with our known terms using fuzzy matching
            for czech_term, english_term in term_mapping.items():
                if similar(czech_term, text_value):
                    # Try to extract current year value
                    current_value = row.get('Běžné účetní období', '').replace(' ', '')
                    if current_value:
                        try:
                            result[f"{english_term}_current"] = int(current_value) if current_value.isdigit() else current_value
                        except (ValueError, KeyError):
                            logger.warning(f"Could not convert value for {english_term}_current")
                    
                    # Try to extract previous year value
                    previous_value = row.get('Minulé účetní období', '').replace(' ', '')
                    if previous_value:
                        try:
                            result[f"{english_term}_previous"] = int(previous_value) if previous_value.isdigit() else previous_value
                        except (ValueError, KeyError):
                            logger.warning(f"Could not convert value for {english_term}_previous")
                    
                    break
        
        return result
    
    def _process_cash_flow(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process and extract key metrics from cash flow statement."""
        result = {}
        
        def similar(a: str, b: str, threshold: float = 0.75) -> bool:
            """Check if two strings are similar using sequence matcher."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
        
        # Map Czech terms to English financial terms
        term_mapping = {
            'Počáteční stav peněžnich prostředkô': 'initial_cash_balance',
            'Účetní zisk nebo ztráta před zdaněním': 'profit_before_tax',
            'Čistý penĕżní tok z provozní činnosti': 'net_operating_cash_flow',
            'Výdaje spojené s nabytím stálých aktiv': 'capex',
            'Příjmy z prodeje stálých aktiv': 'proceeds_from_sale_of_fixed_assets',
            'Čistý penĕžní tok z investiční činnosti': 'net_investment_cash_flow',
            'Čistý penĕžní tok z finančnĭ çinnosti': 'net_financial_cash_flow',
            'Čistá zmĕna peněžních prostředků': 'net_change_in_cash',
            'Konečný stav peněžních prostředků': 'ending_cash_balance'
        }
        
        current_year_column = None
        previous_year_column = None
        
        # Find the relevant columns
        if data and len(data) > 0:
            sample_row = data[0]
            for key in sample_row.keys():
                if 'běžné' in key.lower() or 'bĕžné' in key.lower():
                    current_year_column = key
                elif 'minulé' in key.lower() or 'minulé' in key.lower():
                    previous_year_column = key
        
        # If we couldn't find the columns by name, try using positional logic
        if not current_year_column or not previous_year_column:
            keys = list(data[0].keys())
            if len(keys) >= 7:  # Cash flow table format
                current_year_column = keys[-2]  # Second to last column
                previous_year_column = keys[-1]  # Last column
        
        # Extract values for each financial metric
        for row in data:
            text_column = None
            # Find the column containing TEXT
            for key in row.keys():
                if key == 'TEXT' or 'text' in key.lower():
                    text_column = key
                    break
                    
            if not text_column or not row.get(text_column):
                continue
            
            text_value = row[text_column]
                
            # Try to match with our known terms using fuzzy matching
            for czech_term, english_term in term_mapping.items():
                if similar(czech_term, text_value):
                    # Try to extract current year value
                    if current_year_column and row.get(current_year_column):
                        try:
                            value = row[current_year_column].replace(' ', '')
                            # Handle negative numbers in parentheses
                            if value.startswith('(') and value.endswith(')'):
                                value = '-' + value[1:-1]
                            result[f"{english_term}_current"] = int(value) if value.replace('-', '').isdigit() else value
                        except (ValueError, KeyError):
                            logger.warning(f"Could not convert value for {english_term}_current")
                    
                    # Try to extract previous year value
                    if previous_year_column and row.get(previous_year_column):
                        try:
                            value = row[previous_year_column].replace(' ', '')
                            # Handle negative numbers in parentheses
                            if value.startswith('(') and value.endswith(')'):
                                value = '-' + value[1:-1]
                            result[f"{english_term}_previous"] = int(value) if value.replace('-', '').isdigit() else value
                        except (ValueError, KeyError):
                            logger.warning(f"Could not convert value for {english_term}_previous")
                    
                    break
        
        return result

    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the extracted financial data for consistency.
        Returns validation results and any detected issues.
        """
        validation_results = {
            'is_valid': True,
            'issues': []
        }
        
        # Check total assets = total liabilities + equity
        if all(k in self.financial_data.get('balance_sheet', {}) for k in 
               ['total_assets_current', 'total_liabilities_equity_current']):
            assets = self.financial_data['balance_sheet']['total_assets_current']
            liabilities_equity = self.financial_data['balance_sheet']['total_liabilities_equity_current']
            if assets != liabilities_equity:
                validation_results['is_valid'] = False
                validation_results['issues'].append(
                    f"Balance sheet doesn't balance: assets ({assets}) != liabilities+equity ({liabilities_equity})"
                )
        
        # Check cash flow consistency
        if 'cash_flow' in self.financial_data:
            cf = self.financial_data['cash_flow']
            if all(k in cf for k in ['initial_cash_balance_current', 'net_change_in_cash_current', 
                                   'ending_cash_balance_current']):
                initial = cf['initial_cash_balance_current']
                change = cf['net_change_in_cash_current']
                ending = cf['ending_cash_balance_current']
                if initial + change != ending:
                    validation_results['is_valid'] = False
                    validation_results['issues'].append(
                        f"Cash flow inconsistency: initial ({initial}) + change ({change}) != ending ({ending})"
                    )
        
        return validation_results

    def to_dataframe(self) -> Dict[str, pd.DataFrame]:
        """
        Convert extracted financial data to pandas DataFrames.
        """
        dataframes = {}
        
        for statement_type in ['income_statement', 'balance_sheet', 'cash_flow']:
            if statement_type in self.financial_data:
                # Separate current and previous year data
                current_data = {k.replace('_current', ''): v 
                              for k, v in self.financial_data[statement_type].items() 
                              if k.endswith('_current')}
                previous_data = {k.replace('_previous', ''): v 
                               for k, v in self.financial_data[statement_type].items() 
                               if k.endswith('_previous')}
                
                # Create DataFrame with both years
                df = pd.DataFrame({
                    'Current_Year': pd.Series(current_data),
                    'Previous_Year': pd.Series(previous_data)
                })
                dataframes[statement_type] = df
        
        return dataframes

    def _find_header(self, soup: BeautifulSoup, search_text: str, threshold: float = 0.8) -> Optional[Any]:
        """Find header using fuzzy matching."""
        for header in soup.find_all(['h1', 'h2', 'h3']):
            if SequenceMatcher(None, search_text.lower(), header.text.strip().lower()).ratio() > threshold:
                return header
        return None
    
    
    
    
            
            

    def _extract_company_information(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract company information using pure LLM approach with the first 250 lines."""
        # Initialize with default values as a fallback
        result = {
            "accounting_period": None,
            "company_name": None,
            "legal_form": None,
            "main_activities": [],
            "employees": None,
            "established": None,
            "headquarters": None,
            "industry": None
        }
        
        if not self.client:
            logger.error("OpenAI client not available for LLM extraction")
            return result
        
        try:
            # Get the first 250 lines of text
            all_text = soup.get_text()
            lines = all_text.split('\n')
            first_150_lines = '\n'.join(lines[:150])
            
            # Prepare the LLM prompt with strict requirements
            prompt = f"""
            Extract the following company information from this Czech financial report text. The text is from the first 250 lines of an annual report and may contain OCR errors. Return ONLY a JSON object with the exact fields specified below. Do not add extra fields or explanations.

            Fields to extract:
            1. "accounting_period": The year of the report (e.g., "2023") from the title or early context.
            2. "company_name": The full company name (e.g., "ISOTRA a.s.").
            3. "legal_form": The legal form (e.g., "akciová společnost").
            4. "main_activities": A list of primary business activities as listed in the "Předmět podnikání" section (e.g., "Výroba elektřiny", "Izolatérství"), not marketing activities.
            5. "employees": The number of employees as of the report's end date (e.g., 534 from "k 31. 12. 2023 v trvalém pracovním poměru 534"), or null if not found.
            6. "established": The establishment date (e.g., "14. září 1992").
            7. "headquarters": The company address (e.g., "Bílovecká 2411/1, 74601 Opava").
            8. "industry": The exact industry from this list ONLY: Advertising, Aerospace/Defense, Apparel, Auto & Truck, Auto Parts, Beverage (Alcoholic), Beverage (Soft), Broadcasting, Building Materials, Business & Consumer Services, Cable TV, Chemical (Basic), Chemical (Diversified), Chemical (Specialty), Coal & Related Energy, Computer Services, Computers/Peripherals, Construction Supplies, Diversified, Drugs (Pharmaceutical), Education, Electrical Equipment, Electronics (General), Engineering/Construction, Farming/Agriculture, Food Processing, Food Wholesalers, Furn/Home Furnishings, Homebuilding, Hotel/Gaming, Household Products, Information Services, Machinery, Metals & Mining, Office Equipment & Services, Paper/Forest Products, Power, Real Estate (Operations & Services), Recreation, Restaurant/Dining, Retail (Automotive), Retail (Building Supply), Retail (Distributors), Retail (General), Retail (Special Lines), Rubber& Tires, Semiconductor, Semiconductor Equip, Software (System & Application), Steel, Transportation, Trucking
            Text from document:
            {first_150_lines}

            Return ONLY this JSON structure:
            {{
                "accounting_period": "string",
                "company_name": "string",
                "legal_form": "string",
                "main_activities": ["string", "string", ...],
                "employees": number or null,
                "established": "string",
                "headquarters": "string",
                "industry": "string"
            }}

            If a value cannot be found, use null for strings and numbers, and an empty list for main_activities. For industry even if 
            you dont find exact match try to infer it from what company does based on description. Use fallback Manufacturing if nothing found.
            """
            
            # Call the LLM
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Follow instructions exactly."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for precision
                max_tokens=500   # Enough for a detailed JSON response
            )
            
            # Parse the LLM response
            try:
                llm_data = json.loads(response.choices[0].message.content)
                # Ensure all required fields are present, even if null
                for key in result.keys():
                    if key in llm_data:
                        result[key] = llm_data[key]
                    else:
                        logger.warning(f"LLM did not return required field: {key}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response: {e}")
                return result
        
        except Exception as e:
            logger.error(f"Error in LLM company information extraction: {e}")
            return result
                
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    def _extract_industry_with_fallbacks(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract industry information using multiple fallback strategies."""
        # Strategy 1: Look for industry keywords in the document
        industry_keywords = [
            r'(?:Odvětví|Sektor|Industry):\s*([^\n\.]+)',
            r'(?:NACE|CZ-NACE|Obor podnikání):\s*([^\n\.]+)',
            r'(?:Oblast podnikání|Klasifikace):\s*([^\n\.]+)'
        ]
        
        all_text = soup.get_text()
        
        for pattern in industry_keywords:
            match = re.search(pattern, all_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Strategy 2: Use company description to infer industry
        company_info = self.financial_data['information']
        if 'main_activities' in company_info and company_info['main_activities']:
            # Mapping of keywords to industries
            industry_map = {
                'stínicí': 'Building Materials',
                'žaluzií': 'Building Materials',
                'stavební': 'Construction',
                'výroba': 'Manufacturing',
                'software': 'Technology',
                'IT': 'Technology',
                'potravin': 'Food Production',
                'retail': 'Retail',
                'obchod': 'Retail',
                'finanční': 'Financial Services',
                'bankov': 'Banking',
                'energeti': 'Energy',
                'doprav': 'Transportation'
            }
            
            for activity in company_info['main_activities']:
                for keyword, industry in industry_map.items():
                    if keyword.lower() in activity.lower():
                        return industry
        
        # Strategy 3: Use LLM (most comprehensive)
        if self.client:
            try:
                # Get document excerpts that might contain industry information
                # Look for headers, company description, or summary sections
                potential_sections = []
                
                # Find company description or "about us" section
                description_headers = ['O společnosti', 'Profil společnosti', 'Základní údaje']
                for header in description_headers:
                    header_elem = soup.find(string=re.compile(header, re.IGNORECASE))
                    if header_elem:
                        # Extract next few paragraphs
                        parent = header_elem.parent
                        context = header_elem.get_text() + "\n"
                        for sibling in parent.find_next_siblings():
                            if sibling.name in ['p', 'div'] and len(context) < 2000:
                                context += sibling.get_text() + "\n"
                        potential_sections.append(context)
                
                # Try to extract from any preamble or introduction
                intro_section = soup.find_all(['p', 'div'], limit=10)
                intro_text = "\n".join([section.get_text() for section in intro_section])
                if len(intro_text) > 100:  # Only if it's substantial
                    potential_sections.append(intro_text[:2000])
                
                # If no sections, use first 2000 characters of document
                if not potential_sections:
                    potential_sections.append(all_text[:2000])
                
                # Use company name to help identification
                company_name = company_info.get('company_name', 'Unknown Company')
                
                # Prepare the context
                context = f"Company name: {company_name}\n\n"
                context += "Document excerpts:\n" + "\n---\n".join(potential_sections[:3])  # Limit to 3 sections
                
                prompt = f"""
                Based on the following information from a Czech financial report, determine the industry of the company.
                
                {context}
                
                Identify the industry and return ONLY one of the following industries that best matches:
                - Agriculture
                - Automotive
                - Banking
                - Building Materials
                - Chemicals
                - Construction
                - Consumer Goods
                - Energy
                - Financial Services
                - Food Production
                - Healthcare
                - Hospitality
                - Information Technology
                - Insurance
                - Manufacturing
                - Media
                - Mining
                - Pharmaceuticals
                - Real Estate
                - Retail
                - Technology
                - Telecommunications
                - Transportation
                - Utilities
                
                If absolutely none of these match, return "Other Manufacturing".
                Return just the industry name, nothing else.
                """
                
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # Using 3.5 for cost efficiency, adjust as needed
                    messages=[
                        {"role": "system", "content": "You analyze financial documents to determine the company's industry."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=20  # Just need a short answer
                )
                
                industry = response.choices[0].message.content.strip()
                # Check if the response contains only the industry name
                if industry in [
                    "Agriculture", "Automotive", "Banking", "Building Materials", 
                    "Chemicals", "Construction", "Consumer Goods", "Energy", 
                    "Financial Services", "Food Production", "Healthcare", 
                    "Hospitality", "Information Technology", "Insurance", 
                    "Manufacturing", "Media", "Mining", "Pharmaceuticals", 
                    "Real Estate", "Retail", "Technology", "Telecommunications", 
                    "Transportation", "Utilities", "Other Manufacturing"
                ]:
                    return industry
                else:
                    logger.warning(f"LLM returned invalid industry format: {industry}")
                
            except Exception as e:
                logger.error(f"Error in industry LLM extraction: {e}")
        
        # If all strategies fail, return a default
        return "Manufacturing"  # Most common default
    
    
    
    def _validate_and_fix_required_fields(self, soup: BeautifulSoup) -> None:
        """Validate and fix required fields using fallback mechanisms."""
        # Define required fields for each section
        required_fields = {
            'income_statement': [
                'depreciation_current',
                'operating_profit_current'
            ],
            'balance_sheet': [
                'total_assets_current'
            ],
            'information': [
                'company_name',
                'accounting_period',
                'industry'  # This field is critical as mentioned
            ]
        }
        
        # Check income statement required fields
        missing_income = [field for field in required_fields['income_statement'] 
                        if field not in self.financial_data['income_statement']]
        if missing_income and self.client:
            logger.warning(f"Missing required income statement fields: {missing_income}. Trying LLM fallback.")
            self._extract_income_statement_llm(soup)
        
        # Check balance sheet required fields
        missing_balance = [field for field in required_fields['balance_sheet'] 
                        if field not in self.financial_data['balance_sheet']]
        if missing_balance and self.client:
            logger.warning(f"Missing required balance sheet fields: {missing_balance}. Trying LLM fallback.")
            self._extract_balance_sheet_llm(soup)
        
        # Check company information required fields - special focus on industry
        info = self.financial_data['information']
        if 'industry' not in info or not info['industry']:
            logger.warning("Missing critical field: industry. Attempting special extraction.")
            industry = self._extract_industry_with_fallbacks(soup)
            if industry:
                self.financial_data['information']['industry'] = industry
    
    
    def _perform_final_validation(self) -> None:
        """Perform final validation and log issues instead of raising exceptions."""
        critical_fields = [
            ('income_statement', ['revenue_from_products_and_services_current', 'operating_profit_current']),
            ('balance_sheet', ['total_assets_current']),
            ('information', ['company_name', 'accounting_period', 'industry'])
        ]
        
        missing_fields = []
        
        for section, fields in critical_fields:
            for field in fields:
                if section not in self.financial_data or field not in self.financial_data[section] or self.financial_data[section][field] is None:
                    missing_fields.append(f"{section}.{field}")
        
        if missing_fields:
            logger.warning(f"Some critical financial data fields missing: {', '.join(missing_fields)}. Proceeding with available data.")
        else:
            logger.info("All critical financial data fields present.")
            
    
    
    
    
    def _extract_income_statement_llm(self, soup: BeautifulSoup) -> None:
        """Extract income statement data using LLM as a fallback."""
        if not self.client:
            logger.error("OpenAI client not available for LLM fallback")
            return
        
        try:
            # Find potential income statement sections
            all_text = soup.get_text()
            text_blocks = all_text.split('\n')
            
            income_keywords = ['VÝKAZ ZISKU A ZTRÁT', 'Výkaz zisku a ztráty', 'Income Statement', 'Výsledovka']
            
            # Find sections that contain income statement keywords
            potential_sections = []
            for i, block in enumerate(text_blocks):
                if any(keyword.lower() in block.lower() for keyword in income_keywords):
                    # Take a larger context (220 lines) to ensure we capture the full statement
                    context = '\n'.join(text_blocks[i:i+220])
                    potential_sections.append(context)
            
            if not potential_sections:
                logger.warning("No income statement sections found for LLM extraction")
                return
            
            # Use the first potential section (or combine multiple if they're short)
            combined_context = '\n'.join(potential_sections[:2])  # Use up to 2 sections
            
            prompt = f"""
            Extract key financial metrics from this income statement (Výkaz zisku a ztráty).
            The text is in Czech and might contain OCR errors.
            
            Please extract these specific values for the CURRENT year only:
            1. Revenue from products and services (Tržby z prodeje výrobků a služeb)
            2. Revenue from goods (Tržby za prodej zboží)
            3. Production consumption (Výkonová spotřeba)
            4. Personnel costs (Osobní náklady)
            5. Wage costs (Mzdové náklady)
            6. Depreciation (Odpisy)
            7. Operating profit (Provozní výsledek hospodaření)
            8. EBIT (same as operating profit)
            
            Text from document:
            {combined_context}
            
            Return ONLY a JSON object with these keys:
            {{
                "revenue_from_products_and_services_current": number,
                "revenue_from_goods_current": number,
                "production_consumption_current": number,
                "personnel_costs_current": number,
                "wage_costs_current": number,
                "depreciation_current": number,
                "operating_profit_current": number,
                "ebit_current": number
            }}
            
            If a value cannot be found, use null. Remove any thousands separators and convert to integers.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            try:
                llm_extracted_data = json.loads(response.choices[0].message.content)
                
                # Merge with existing income statement data, prioritizing existing values
                current_data = self.financial_data['income_statement']
                for key, value in llm_extracted_data.items():
                    if value is not None and key not in current_data:
                        current_data[key] = value
                
                # Make sure EBIT is same as operating profit if one is missing
                if 'operating_profit_current' in current_data and 'ebit_current' not in current_data:
                    current_data['ebit_current'] = current_data['operating_profit_current']
                elif 'ebit_current' in current_data and 'operating_profit_current' not in current_data:
                    current_data['operating_profit_current'] = current_data['ebit_current']
                
                logger.info("Successfully extracted missing income statement data using LLM")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM income statement response: {e}")
            
        except Exception as e:
            logger.error(f"Error in income statement LLM extraction: {e}")

    def _extract_balance_sheet_llm(self, soup: BeautifulSoup) -> None:
        """Extract balance sheet data using LLM as a fallback."""
        if not self.client:
            logger.error("OpenAI client not available for LLM fallback")
            return
        
        try:
            # Find potential balance sheet sections
            all_text = soup.get_text()
            text_blocks = all_text.split('\n')
            
            balance_keywords = ['ROZVAHA', 'Rozvaha', 'Balance Sheet', 'AKTIVA', 'PASIVA']
            
            # Find sections that contain balance sheet keywords
            potential_sections = []
            for i, block in enumerate(text_blocks):
                if any(keyword.lower() in block.lower() for keyword in balance_keywords):
                    # Take a larger context to ensure we capture the full statement
                    context = '\n'.join(text_blocks[i:i+120])
                    potential_sections.append(context)
            
            if not potential_sections:
                logger.warning("No balance sheet sections found for LLM extraction")
                return
            
            # Use the first potential section (or combine multiple if needed)
            combined_context = '\n'.join(potential_sections[:2])
            
            prompt = f"""
            Extract key financial metrics from this balance sheet (Rozvaha).
            The text is in Czech and might contain OCR errors.
            
            Please extract these specific values for both the CURRENT and PREVIOUS years:
            1. Total assets (AKTIVA CELKEM)
            2. Intangible assets (Dlouhodobý nehmotný majetek)
            3. Tangible assets (Dlouhodobý hmotný majetek)
            4. Current assets (Oběžná aktiva)
            
            Text from document:
            {combined_context}
            
            Return ONLY a JSON object with these keys:
            {{
                "total_assets_current": number,
                "total_assets_previous": number,
                "intangible_assets_current": number,
                "intangible_assets_previous": number,
                "tangible_assets_current": number,
                "tangible_assets_previous": number,
                "current_assets_current": number,
                "current_assets_previous": number
            }}
            
            If a value cannot be found, use null. Remove any thousands separators and convert to integers.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a financial data extraction assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            
            try:
                llm_extracted_data = json.loads(response.choices[0].message.content)
                
                # Merge with existing balance sheet data, prioritizing existing values
                current_data = self.financial_data['balance_sheet']
                for key, value in llm_extracted_data.items():
                    if value is not None and key not in current_data:
                        current_data[key] = value
                
                logger.info("Successfully extracted missing balance sheet data using LLM")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM balance sheet response: {e}")
            
        except Exception as e:
            logger.error(f"Error in balance sheet LLM extraction: {e}")
                    
                
                
                

# Example usage
if __name__ == "__main__":
    # Use settings instead of direct env var
    extractor = FinancialExtractor(openai_api_key=settings.OPENAI_API_KEY)
    
    try:
        # Read the HTML file
        with open('output_vz_2023_isotra.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Extract and validate data
        financial_data = extractor.extract_from_html(html_content)
        validation = extractor.validate_data()
        dataframes = extractor.to_dataframe()
        
        # Save dictionary output as JSON
        with open('financial_data.json', 'w', encoding='utf-8') as f:
            json.dump(financial_data, f, indent=4, ensure_ascii=False)
            
        # Save validation results
        with open('validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(validation, f, indent=4, ensure_ascii=False)
        
        # Save DataFrames to Excel (all in one file, different sheets)
        with pd.ExcelWriter('financial_data.xlsx') as writer:
            for statement_type, df in dataframes.items():
                df.to_excel(writer, sheet_name=statement_type)
        
        print("Files saved successfully:")
        print("1. financial_data.json - Raw extracted data")
        print("2. validation_results.json - Validation results")
        print("3. financial_data.xlsx - Formatted statements in Excel")
            
    except FileNotFoundError:
        print("Error: Please provide a financial_report.html file in the same directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")