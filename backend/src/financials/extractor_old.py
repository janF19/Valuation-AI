import json
import requests
import os
import time

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
        """Extract data from income statement using multiple fallback strategies with fuzzy matching for OCR text."""
        income_table = None
        
        # Strategy 1: Find by fuzzy header matching in any element
        income_keywords = ['VÝKAZ ZISKU', 'VYKAZ ZISKU', 'VÝKAZ ZISKU A ZTRÁTY', 'VYKAZ ZISKU A ZTRATY', 'VÝSLEDOVKA']
        
        # Log the search process for debugging
        logger.info("Starting income statement extraction with multiple strategies")
        
        # First, try to find tables that have income statement structure directly
        all_tables = soup.find_all('table')
        logger.info(f"Found {len(all_tables)} tables in document")
        
        # Add direct table content analysis as first strategy
        for table_idx, table in enumerate(all_tables):
            rows = table.find_all('tr')
            if len(rows) < 5:  # Too small to be income statement
                continue
            
            # Check for numeric patterns in columns
            numeric_columns = 0
            text_column_found = False
            
            # Sample a few rows to check structure
            sample_rows = rows[1:min(10, len(rows))]
            for row in sample_rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:  # Need at least description + number
                    # First cell should be text (description)
                    first_cell_text = cells[0].get_text().strip()
                    if len(first_cell_text) > 3 and not first_cell_text.isdigit():
                        text_column_found = True
                    
                    # Other cells should contain numbers
                    for cell in cells[1:]:
                        cell_text = cell.get_text().strip()
                        # Check if cell contains a number (allowing for formatting)
                        if re.search(r'\d', cell_text):
                            numeric_columns += 1
            
            # If we have both text descriptions and numeric data, this might be a financial table
            if text_column_found and numeric_columns > 10:
                # Extract and try to process this table
                table_data = self._extract_table_data(table)
                if table_data and len(table_data) > 5:
                    # Try to find key financial terms in the data
                    financial_terms = ['tržby', 'výnos', 'náklad', 'zisk', 'výsledek', 'spotřeba', 'osobní']
                    matches = 0
                    for row in table_data:
                        text = row.get('TEXT', '')
                        if not text:
                            continue
                        if any(term in text.lower() for term in financial_terms):
                            matches += 1
                    
                    if matches >= 2:
                        logger.info(f"Found potential income statement table #{table_idx+1} with {matches} financial terms")
                        processed_data = self._process_income_statement(table_data)
                        if processed_data and ('revenue_from_products_and_services_current' in processed_data or 
                                              'operating_profit_current' in processed_data):
                            self.financial_data['income_statement'] = processed_data
                            logger.info(f"Successfully processed income statement with {len(processed_data)} metrics")
                            return  # Successfully found and processed
        
        # Continue with existing strategies if direct analysis didn't work
        if not income_table:
            for element in soup.find_all(['h1', 'h2', 'h3', 'p', 'th', 'td', 'div', 'span']):
                element_text = element.get_text().strip()
                # Use fuzzy matching with a lower threshold for OCR text
                for keyword in income_keywords:
                    if SequenceMatcher(None, keyword.lower(), element_text.lower()).ratio() > 0.6:
                        logger.info(f"Found income statement header with fuzzy match: '{element_text}'")
                        # Try to find table after this element
                        income_table = self._find_table_after_header(element)
                        if income_table:
                            break
                if income_table:
                    break
        
        # Strategy 2: Search in full text for income statement keywords, then find nearby tables
        if not income_table:
            all_text = soup.get_text()
            for keyword in income_keywords:
                # Find approximate position of keyword in text
                for i in range(len(all_text) - 5):
                    text_segment = all_text[i:i+len(keyword)+5].lower()
                    if SequenceMatcher(None, keyword.lower(), text_segment).ratio() > 0.6:
                        # Found a potential match, look for tables nearby in the HTML
                        position = all_text.find(text_segment)
                        if position > 0:
                            # Get all tables and find the closest one after this position
                            for table in all_tables:
                                table_pos = all_text.find(table.get_text())
                                if table_pos > position and table_pos - position < 2000:  # Within reasonable distance
                                    income_table = table
                                    logger.info(f"Found income statement table near keyword match at position {position}")
                                    break
                        if income_table:
                            break
                if income_table:
                    break
        
        # Strategy 3: Find by key income statement line items with fuzzy matching
        if not income_table:
            key_items = [
                'Tržby z prodeje výrobků', 'Trzby z prodeje vyrobku',
                'Výkonová spotřeba', 'Vykonova spotreba',
                'Provozní výsledek', 'Provozni vysledek'
            ]
            
            for table in all_tables:
                matches = 0
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    for cell in cells:
                        cell_text = cell.get_text().strip()
                        for item in key_items:
                            if SequenceMatcher(None, item.lower(), cell_text.lower()).ratio() > 0.7:
                                matches += 1
            
                # If we find at least 2 key items, this is likely the income statement
                if matches >= 2:
                    income_table = table
                    logger.info(f"Found income statement table by key items with {matches} matches")
                    break
        
        # Strategy 4: Use table structure analysis with more flexible pattern matching
        if not income_table:
            for table in all_tables:
                rows = table.find_all('tr')
                if len(rows) > 5:  # Income statement usually has many rows
                    # Check for numeric patterns in the table
                    numeric_pattern = re.compile(r'^\s*[\d\s\-\(\)]+\s*$')
                    numeric_cells = 0
                    total_cells = 0
                    
                    for row in rows[1:10]:  # Check a sample of rows
                        cells = row.find_all(['td', 'th'])
                        if len(cells) >= 3:  # Need at least description + numbers
                            for cell in cells[1:]:  # Skip first cell which is usually text
                                cell_text = cell.get_text().strip()
                                total_cells += 1
                                if numeric_pattern.match(cell_text) or cell_text.isdigit():
                                    numeric_cells += 1
                    
                    # If more than 40% of cells contain numeric data, this might be a financial table
                    if total_cells > 0 and numeric_cells / total_cells > 0.4:
                        # Now check if it has income statement keywords
                        table_text = table.get_text().lower()
                        income_indicators = ['tržby', 'trzby', 'výnos', 'vynos', 'náklad', 'naklad', 
                                            'zisk', 'ztráta', 'ztrata', 'hospodaření', 'hospodareni']
                        
                        indicator_matches = sum(1 for indicator in income_indicators if indicator in table_text)
                        if indicator_matches >= 2:
                            income_table = table
                            logger.info(f"Found income statement table by structure analysis with {indicator_matches} keyword matches")
                            break
        
        if not income_table:
            logger.warning("Income statement table not found using any strategy")
            # Immediately try LLM fallback if available
            if self.client:
                logger.info("Attempting LLM fallback for income statement extraction")
                self._extract_income_statement_llm(soup)
            return
        
        # Extract the data using existing method
        income_data = self._extract_table_data(income_table)
        
        # Debug the extracted data
        logger.info(f"Extracted {len(income_data)} rows from income statement table")
        if income_data and len(income_data) > 0:
            logger.debug(f"Sample row: {income_data[0]}")
        
        # Process the data even if validation fails, but log the warning
        if not self._validate_income_statement_data(income_data):
            logger.warning("Extracted income statement data appears invalid, but proceeding with processing")
        
        # Always try to process and store key metrics
        processed_data = self._process_income_statement(income_data)
        
        # Only store if we got some data
        if processed_data:
            self.financial_data['income_statement'] = processed_data
            logger.info(f"Successfully processed income statement with {len(processed_data)} metrics")
        else:
            logger.error("Failed to process income statement data")
            # Try LLM fallback if we couldn't process the data
            if self.client:
                logger.info("Attempting LLM fallback for income statement extraction")
                self._extract_income_statement_llm(soup)
    
    def _validate_income_statement_data(self, data: List[Dict[str, str]]) -> bool:
        """Validate that extracted data appears to be from income statement with fuzzy matching."""
        if not data:
            return False
        
        # Check for key income statement items with lower threshold for OCR text
        key_terms = [
            (r'tržby|výnosy|revenue|trzby|vynosy', 0.6),
            (r'spotřeba|náklady|costs|spotreba|naklady', 0.6),
            (r'výsledek|zisk|profit|vysledek', 0.6)
        ]
        
        matches = 0
        for row in data:
            text = row.get('TEXT', '')
            if not text:  # Try alternative keys if TEXT is not found
                for key in row.keys():
                    if 'text' in key.lower() or 'název' in key.lower() or 'nazev' in key.lower():
                        text = row[key].lower()
                        break
            
            if not text:
                continue
            
            for pattern, threshold in key_terms:
                if any(SequenceMatcher(None, text.lower(), term).ratio() > threshold 
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
            # Get all rows first
            all_rows = table.find_all('tr')
            if not all_rows:
                logger.warning("No rows found in table")
                return rows
            
            logger.info(f"Found {len(all_rows)} rows in table")
            
            # Try to find header row - it could be in thead or just the first row
            header_row = None
            headers = []
            
            # First check if there's a thead
            thead = table.find('thead')
            if thead:
                header_rows = thead.find_all('tr')
                if header_rows:
                    header_row = header_rows[-1]  # Use the last row in thead
                    headers = [header.get_text().strip() for header in header_row.find_all(['th', 'td'])]
            
            # If no headers found in thead, use the first row
            if not headers:
                if all_rows:
                    header_row = all_rows[0]
                    headers = [header.get_text().strip() for header in header_row.find_all(['th', 'td'])]
            
            logger.debug(f"Headers found: {headers}")
            
            # If still no headers, create default ones
            if not headers:
                logger.warning("No headers found in table, using default headers")
                headers = ["TEXT", "Current", "Previous"]
            
            # Determine which columns to use for text and values
            text_col_index = 0  # Default to first column
            current_value_index = 1  # Default to second column
            previous_value_index = 2  # Default to third column
            
            # Try to identify columns by name
            for i, header in enumerate(headers):
                header_lower = header.lower()
                if any(term in header_lower for term in ['text', 'název', 'nazev', 'položka', 'polozka']):
                    text_col_index = i
                elif any(term in header_lower for term in ['běžné', 'bezne', 'current']):
                    current_value_index = i
                elif any(term in header_lower for term in ['minulé', 'minule', 'previous']):
                    previous_value_index = i
            
            # Get data rows - skip header row if it was identified
            data_rows = all_rows[1:] if header_row in all_rows else all_rows
            
            for row in data_rows:
                cells = row.find_all(['td', 'th'])
                
                # Skip rows without enough cells
                if len(cells) <= max(text_col_index, current_value_index, previous_value_index):
                    continue
                
                # Get the text and values
                text_value = cells[text_col_index].get_text().strip()
                
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

                    current_value = clean_value(cells[current_value_index].get_text())
                    previous_value = clean_value(cells[previous_value_index].get_text())
                except IndexError:
                    continue
                
                if text_value:  # Accept row even if values are empty
                    row_data = {
                        'TEXT': text_value,
                        'Běžné účetní období': current_value,
                        'Minulé účetní období': previous_value
                    }
                    rows.append(row_data)
                    
        except Exception as e:
            logger.error(f"Error processing table: {str(e)}", exc_info=True)
        
        return rows
    
    def _process_income_statement(self, data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Process and extract key metrics from income statement."""
        result = {}
        
        # Add debug logging
        logger.debug(f"Processing income statement data: {data[:2]}...")  # Show first 2 rows
        
        # Improved fuzzy matching with lower threshold for OCR text
        def similar(a: str, b: str, threshold: float = 0.6) -> bool:
            """Check if two strings are similar using sequence matcher."""
            return SequenceMatcher(None, a.lower(), b.lower()).ratio() > threshold
        
        # Enhanced term mapping with more variations and common OCR errors
        term_mapping = {
            'Tržby z prodeje výrobkủ a služeb|Tržby z prodeje vlastních výrobků|Výnosy|Tržby|Trzby|Výkony|Vykony': 'revenue_from_products_and_services',
            'Tržby za prodej zboží|Tržby zboží|Trzby za zbozi': 'revenue_from_goods',
            'Výkonová spotřeba|Náklady výkonová|Spotřeba|Vykonova spotreba|Spotreba materialu': 'production_consumption',
            'Osobní náklady|Mzdové náklady|Osobni naklady|Naklady na zamestnance': 'personnel_costs',
            'Mzdové náklady|Mzdy|Mzdove naklady': 'wage_costs',
            'Provozní výsledek hospodaření|Provozní výsledek|Výsledek hospodaření provozní|Provozni vysledek|Vysledek hospodareni': 'operating_profit',
            'Výsledek hospodaření před zdaněním|Výsledek před zdaněním|Zisk před zdaněním|Vysledek pred zdanenim': 'profit_before_tax',
            'Výsledek hospodaření za účetní období|Výsledek za období|Čistý zisk|Zisk po zdanění|Cisty zisk': 'net_profit',
            'Úpravy hodnot v provozní oblasti|Odpisy|Odpisy dlouhodobého majetku|Odpisy dlouhodobeho majetku|Upravy hodnot': 'depreciation'
        }
        
        # Find the relevant columns for current and previous years
        current_year_column = None
        previous_year_column = None
        
        if data and len(data) > 0:
            sample_row = data[0]
            logger.debug(f"Sample row keys: {sample_row.keys()}")
            
            # Try multiple patterns for column identification
            current_patterns = ['běžné', 'běžném', 'current', '2023', 'běžn']
            previous_patterns = ['minulé', 'předchozí', 'previous', '2022', 'minul']
            
            for key in sample_row.keys():
                key_lower = key.lower()
                if any(pattern in key_lower for pattern in current_patterns):
                    current_year_column = key
                elif any(pattern in key_lower for pattern in previous_patterns):
                    previous_year_column = key
        
        # Fallback to positional logic if needed
        if not current_year_column or not previous_year_column:
            logger.info("Using positional logic for column identification")
            if 'Běžné účetní období' in data[0]:
                current_year_column = 'Běžné účetní období'
            if 'Minulé účetní období' in data[0]:
                previous_year_column = 'Minulé účetní období'
        
        # Last resort fallback
        if not current_year_column and len(data[0]) >= 2:
            keys = list(data[0].keys())
            if 'TEXT' in keys and len(keys) >= 2:
                text_index = keys.index('TEXT')
                if text_index + 1 < len(keys):
                    current_year_column = keys[text_index + 1]
                if text_index + 2 < len(keys):
                    previous_year_column = keys[text_index + 2]
        
        logger.debug(f"Selected columns - Current: {current_year_column}, Previous: {previous_year_column}")

        # Add more aggressive numeric value extraction
        for row in data:
            text_value = row.get('TEXT', '')
            if not text_value:
                # Try alternative keys if TEXT is not found
                for key in row.keys():
                    if 'text' in key.lower() or 'název' in key.lower() or 'nazev' in key.lower() or 'popis' in key.lower():
                        text_value = row[key]
                        break
            
            if not text_value:
                continue
            
            # Try to match with our known terms using fuzzy matching
            for czech_terms, english_term in term_mapping.items():
                if any(similar(term, text_value) for term in czech_terms.split('|')):
                    logger.debug(f"Matched '{text_value}' to '{english_term}'")
                    
                    # Look for numeric values in any column
                    for col_key, col_value in row.items():
                        if col_key == 'TEXT' or 'text' in col_key.lower():
                            continue  # Skip text columns
                        
                        # Try to extract a numeric value
                        if col_value:
                            # Clean and extract numeric value
                            numeric_str = re.sub(r'[^\d\-]', '', str(col_value).replace(' ', ''))
                            if numeric_str and numeric_str != '-':
                                try:
                                    numeric_value = int(numeric_str)
                                    # Determine if this is current or previous year
                                    if 'běžné' in col_key.lower() or 'bezne' in col_key.lower() or 'current' in col_key.lower():
                                        result[f"{english_term}_current"] = numeric_value
                                    elif 'minulé' in col_key.lower() or 'minule' in col_key.lower() or 'previous' in col_key.lower():
                                        result[f"{english_term}_previous"] = numeric_value
                                except ValueError:
                                    pass  # Not a valid number
        
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
        """Extract company information using pure LLM approach with the first 100 lines."""
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
            
            # Call the LLM with a smaller model
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # Use a smaller model with lower token limits
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
            industry = "Not found" #self._extract_industry_with_fallbacks(soup)
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
        """Extract income statement data using LLM as a fallback with fuzzy matching for OCR text."""
        if not self.client:
            logger.error("OpenAI client not available for LLM fallback")
            return
        
        try:
            # Find potential income statement sections using fuzzy matching
            all_text = soup.get_text()
            text_blocks = all_text.split('\n')
            
            income_keywords = ['VÝKAZ ZISKU A ZTRÁT', 'VYKAZ ZISKU A ZTRAT', 'Výkaz zisku a ztráty', 'Výsledovka']
            
            # Find sections that contain income statement keywords with fuzzy matching
            potential_sections = []
            for i, block in enumerate(text_blocks):
                for keyword in income_keywords:
                    if SequenceMatcher(None, keyword.lower(), block.lower()).ratio() > 0.6:
                        # Take a larger context (220 lines) to ensure we capture the full statement
                        context = '\n'.join(text_blocks[i:i+220])
                        potential_sections.append((context, i))
                        logger.info(f"Found potential income statement section at line {i} with fuzzy match. Content: '{block[:50]}...'")
                        break
            
            # If no sections found with fuzzy matching, try searching in the full text
            if not potential_sections:
                logger.info("No income statement sections found with fuzzy matching, trying full text search")
                for keyword in income_keywords:
                    for i in range(len(all_text) - 10):
                        text_segment = all_text[i:i+len(keyword)+10].lower()
                        if SequenceMatcher(None, keyword.lower(), text_segment).ratio() > 0.6:
                            # Find the line number for this position
                            line_num = all_text[:i].count('\n')
                            if line_num < len(text_blocks):
                                context = '\n'.join(text_blocks[line_num:line_num+220])
                                potential_sections.append((context, line_num))
                                logger.info(f"Found potential income statement section at position {i} (line {line_num}) with full text search. Content: '{text_segment[:50]}...'")
                                break
                    if potential_sections:
                        break
            
            # If still no sections, try looking for key income statement items
            if not potential_sections:
                logger.info("No income statement sections found with keywords, looking for key items")
                key_items = [
                    'Tržby z prodeje výrobků', 'Trzby z prodeje vyrobku',
                    'Výkonová spotřeba', 'Vykonova spotreba',
                    'Provozní výsledek', 'Provozni vysledek'
                ]
                
                for i, block in enumerate(text_blocks):
                    for item in key_items:
                        if SequenceMatcher(None, item.lower(), block.lower()).ratio() > 0.7:
                            # If we find a key item, this might be part of the income statement
                            # Take context from 20 lines before to 200 lines after
                            start = max(0, i-20)
                            context = '\n'.join(text_blocks[start:i+200])
                            potential_sections.append((context, start))
                            logger.info(f"Found potential income statement section at line {i} with key item match. Content: '{block[:50]}...'")
                            break
                    if len(potential_sections) >= 2:  # Limit to 2 sections
                        break
            
            if not potential_sections:
                logger.warning("No income statement sections found for LLM extraction")
                return
            
            # Sort sections by line number and use the first 2 (or combine if they're short)
            potential_sections.sort(key=lambda x: x[1])
            contexts = [section[0] for section in potential_sections[:2]]
            combined_context = '\n'.join(contexts)
            
            # Limit context size to avoid token limits
            if len(combined_context) > 12000:
                combined_context = combined_context[:12000]
                logger.info("Truncated context to 12000 characters for LLM processing")
            
            # Log a sample of the context being sent to the LLM
            logger.debug(f"Sample of context being sent to LLM: {combined_context[:500]}...")
            
            prompt = f"""
            Extract key financial metrics from this income statement (Výkaz zisku a ztráty).
            The text is in Czech and contains OCR errors, so look for approximate matches.
            
            Please extract these specific values for the CURRENT year only:
            1. Revenue from products and services (Tržby z prodeje výrobků a služeb)
            2. Revenue from goods (Tržby za prodej zboží)
            3. Production consumption (Výkonová spotřeba)
            4. Personnel costs (Osobní náklady)
            5. Wage costs (Mzdové náklady)
            6. Depreciation (Odpisy or Úpravy hodnot)
            7. Operating profit (Provozní výsledek hospodaření)
            8. EBIT (same as operating profit)
            
            Text from document:
            {combined_context}
            
            Return ONLY a valid JSON object with these keys. If you cannot find any values, return an empty JSON object {{}}.
            If you find some values but not others, include only the ones you find.
            
            Example of valid response:
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
            Be flexible with OCR errors - look for similar terms and numbers that appear in the right context.
            DO NOT include any explanatory text before or after the JSON.
            """
            
            try:
                # Add exponential backoff retry logic
                max_retries = 5
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",  # Use a smaller model to reduce rate limiting
                            messages=[
                                {"role": "system", "content": "You are a financial data extraction assistant specialized in handling OCR errors."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1
                        )
                        break  # If successful, break out of retry loop
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.info(f"Rate limited, retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise  # Re-raise if we've exhausted our retries
                        else:
                            raise  # Re-raise if it's not a rate limit error
                
                response_content = response.choices[0].message.content
                # Fix: Don't try to access usage stats with get() method
                logger.info("LLM request completed successfully")
                logger.debug(f"LLM response: {response_content}")
                
                try:
                    llm_extracted_data = json.loads(response_content)
                    
                    # Merge with existing income statement data, prioritizing existing values
                    current_data = self.financial_data.get('income_statement', {})
                    if 'income_statement' not in self.financial_data:
                        self.financial_data['income_statement'] = {}
                    
                    for key, value in llm_extracted_data.items():
                        if value is not None and key not in current_data:
                            self.financial_data['income_statement'][key] = value
                    
                    # Make sure EBIT is same as operating profit if one is missing
                    if 'operating_profit_current' in self.financial_data['income_statement'] and 'ebit_current' not in self.financial_data['income_statement']:
                        self.financial_data['income_statement']['ebit_current'] = self.financial_data['income_statement']['operating_profit_current']
                    elif 'ebit_current' in self.financial_data['income_statement'] and 'operating_profit_current' not in self.financial_data['income_statement']:
                        self.financial_data['income_statement']['operating_profit_current'] = self.financial_data['income_statement']['ebit_current']
                    
                    logger.info("Successfully extracted missing income statement data using LLM")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM income statement response: {e}. Response: {response_content[:100]}")
                    # Add fallback parsing - try to extract JSON from text
                    try:
                        # Look for JSON-like content between curly braces
                        match = re.search(r'\{.*\}', response_content, re.DOTALL)
                        if match:
                            potential_json = match.group(0)
                            llm_extracted_data = json.loads(potential_json)
                            logger.info("Successfully extracted JSON using regex fallback")
                            # Continue with the existing logic to process the data
                            # ... (copy the existing code that processes llm_extracted_data)
                        else:
                            # If no JSON found, create empty dict with default values
                            logger.warning("No JSON-like content found in response, using defaults")
                            llm_extracted_data = {
                                "operating_profit_current": None,
                                "depreciation_current": None
                            }
                    except Exception as nested_e:
                        logger.error(f"Fallback JSON parsing also failed: {nested_e}")
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
        
        except Exception as e:
            logger.error(f"Error in income statement LLM extraction: {e}")

    def _extract_balance_sheet_llm(self, soup: BeautifulSoup) -> None:
        """Extract balance sheet data using LLM as a fallback with fuzzy matching for OCR text."""
        if not self.client:
            logger.error("OpenAI client not available for LLM fallback")
            return
        
        try:
            # Find potential balance sheet sections using fuzzy matching
            all_text = soup.get_text()
            text_blocks = all_text.split('\n')
            
            balance_keywords = ['ROZVAHA', 'Rozvaha', 'Balance Sheet', 'AKTIVA CELKEM', 'PASIVA CELKEM']
            
            # Find sections that contain balance sheet keywords with fuzzy matching
            potential_sections = []
            for i, block in enumerate(text_blocks):
                for keyword in balance_keywords:
                    if SequenceMatcher(None, keyword.lower(), block.lower()).ratio() > 0.7:
                        # Take a larger context to ensure we capture the full statement
                        context = '\n'.join(text_blocks[i:i+120])
                        potential_sections.append((context, i))
                        logger.info(f"Found potential balance sheet section at line {i} with fuzzy match. Content: '{block[:50]}...'")
                        break
            
            # If no sections found with fuzzy matching, try searching in the full text
            if not potential_sections:
                logger.info("No balance sheet sections found with fuzzy matching, trying full text search")
                found_section = False
                for keyword in balance_keywords:
                    if found_section:
                        break
                    for i in range(len(all_text) - 10):
                        text_segment = all_text[i:i+len(keyword)+10].lower()
                        if SequenceMatcher(None, keyword.lower(), text_segment).ratio() > 0.7:
                            # Find the line number for this position
                            line_num = all_text[:i].count('\n')
                            if line_num < len(text_blocks):
                                context = '\n'.join(text_blocks[line_num:line_num+120])
                                potential_sections.append((context, line_num))
                                logger.info(f"Found potential balance sheet section at position {i} (line {line_num}) with full text search. Content: '{text_segment[:50]}...'")
                                found_section = True
                                break
            
            # If still no sections, try looking for key balance sheet items
            if not potential_sections:
                logger.info("No balance sheet sections found with keywords, looking for key items")
                key_items = [
                    'Dlouhodobý majetek', 'Dlouhodoby majetek',
                    'Oběžná aktiva', 'Obezna aktiva',
                    'Vlastní kapitál', 'Vlastni kapital',
                    'Cizí zdroje', 'Cizi zdroje'
                ]
                
                found_items = False
                for i, block in enumerate(text_blocks):
                    if found_items:
                        break
                    for item in key_items:
                        if SequenceMatcher(None, item.lower(), block.lower()).ratio() > 0.7:
                            # If we find a key item, this might be part of the balance sheet
                            # Take context from 20 lines before to 100 lines after
                            start = max(0, i-20)
                            context = '\n'.join(text_blocks[start:i+100])
                            potential_sections.append((context, start))
                            logger.info(f"Found potential balance sheet section at line {i} with key item match. Content: '{block[:50]}...'")
                            found_items = True
                            break
            
            if not potential_sections:
                logger.warning("No balance sheet sections found for LLM extraction")
                return
            
            # Sort sections by line number and use the first 2 (or combine if they're short)
            potential_sections.sort(key=lambda x: x[1])
            contexts = [section[0] for section in potential_sections[:2]]
            combined_context = '\n'.join(contexts)
            
            # Limit context size to avoid token limits
            if len(combined_context) > 10000:
                combined_context = combined_context[:10000]
                logger.info("Truncated context to 10000 characters for LLM processing")
            
            # Log a sample of the context being sent to the LLM
            logger.debug(f"Sample of balance sheet context being sent to LLM: {combined_context[:500]}...")
            
            prompt = f"""
            Extract key financial metrics from this balance sheet (Rozvaha).
            The text is in Czech and contains OCR errors, so look for approximate matches.
            
            Please extract these specific values for both the CURRENT and PREVIOUS years:
            1. Total assets (AKTIVA CELKEM)
            2. Intangible assets (Dlouhodobý nehmotný majetek)
            3. Tangible assets (Dlouhodobý hmotný majetek)
            4. Current assets (Oběžná aktiva)
            5. Total equity and liabilities (PASIVA CELKEM)
            6. Equity (Vlastní kapitál)
            7. Liabilities (Cizí zdroje)
            
            Text from document:
            {combined_context}
            
            Return ONLY a valid JSON object with these keys. If you cannot find any values, return an empty JSON object {{}}.
            If you find some values but not others, include only the ones you find.
            
            Example of valid response:
            {{
                "total_assets_current": 1000000,
                "total_assets_previous": 900000,
                "intangible_assets_current": 500000,
                "intangible_assets_previous": 400000,
                "tangible_assets_current": 500000,
                "tangible_assets_previous": 400000,
                "current_assets_current": 1000000,
                "current_assets_previous": 900000,
                "total_liabilities_equity_current": 1000000,
                "total_liabilities_equity_previous": 900000,
                "equity_current": 500000,
                "equity_previous": 400000
            }}
            
            If a value cannot be found, use null. Remove any thousands separators and convert to integers.
            Be flexible with OCR errors - look for similar terms and numbers that appear in the right context.
            DO NOT include any explanatory text before or after the JSON.
            """
            
            try:
                # Add exponential backoff retry logic
                max_retries = 5
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        response = self.client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are a financial data extraction assistant specialized in handling OCR errors."},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.1
                        )
                        break  # If successful, break out of retry loop
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                logger.info(f"Rate limited, retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                            else:
                                raise  # Re-raise if we've exhausted our retries
                        else:
                            raise  # Re-raise if it's not a rate limit error
                
                response_content = response.choices[0].message.content
                # Fix: Don't try to access usage stats with get() method
                logger.info("LLM request completed successfully")
                logger.debug(f"LLM balance sheet response: {response_content}")
                
                try:
                    llm_extracted_data = json.loads(response_content)
                    
                    # Merge with existing balance sheet data, prioritizing existing values
                    if 'balance_sheet' not in self.financial_data:
                        self.financial_data['balance_sheet'] = {}
                    
                    current_data = self.financial_data['balance_sheet']
                    for key, value in llm_extracted_data.items():
                        if value is not None and key not in current_data:
                            current_data[key] = value
                    
                    # Ensure total assets equals total liabilities + equity if both are present
                    if ('total_assets_current' in current_data and 
                        'total_liabilities_equity_current' in current_data and
                        current_data['total_assets_current'] != current_data['total_liabilities_equity_current']):
                        logger.warning("Balance sheet doesn't balance in LLM extraction, but proceeding with data")
                    
                    logger.info("Successfully extracted missing balance sheet data using LLM")
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM balance sheet response: {e}. Response: {response_content[:100]}")
            except Exception as e:
                logger.error(f"Error calling LLM API for balance sheet: {e}")
        
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