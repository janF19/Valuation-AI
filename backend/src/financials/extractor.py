import json
import requests
import os

from typing import Dict, Any, Optional

import re
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from difflib import SequenceMatcher
import openai
from backend.config.settings import settings
from datetime import datetime

logger = logging.getLogger(__name__)

class FinancialExtractor:
    """Extract financial data from HTML reports."""
    
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
            self._extract_income_statement(soup)
            self._extract_balance_sheet(soup)
            self._extract_cash_flow(soup)
            
            # Add company information extraction
            self.financial_data['information'] = self._extract_company_information(soup)
            
            # Ensure required fields exist
            if not self.financial_data['information'].get('company_name'):
                self.financial_data['information']['company_name'] = "Unknown Company"
            if not self.financial_data['information'].get('accounting_period'):
                self.financial_data['information']['accounting_period'] = str(datetime.now().year)
            
            return self.financial_data
            
        except Exception as e:
            logger.error(f"Error extracting financial data: {e}")
            # Return minimal valid structure instead of raising
            return {
                'income_statement': {},
                'balance_sheet': {},
                'cash_flow': {},
                'information': {
                    'company_name': "Unknown Company",
                    'accounting_period': str(datetime.now().year)
                }
            }
    
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
        
        # Strategy 5 (Last Resort): Use fuzzy search + OpenAI
        if (not income_table or not processed_data) and self.client:
            logger.info("Attempting last resort strategy with fuzzy search and LLM")
            try:
                # Find any text containing "VÝKAZ ZISKU A ZTRÁT" with fuzzy matching
                all_text = soup.get_text()
                text_blocks = all_text.split('\n')
                
                # Store all potential income statement sections
                potential_sections = []
                
                for i, block in enumerate(text_blocks):
                    if SequenceMatcher(None, "VÝKAZ ZISKU A ZTRÁT", block.upper()).ratio() > 0.7:
                        # Take next 130 lines after finding the header (increased from 30 to capture full statement)
                        context = '\n'.join(text_blocks[i:i+130])
                        
                        # Simple validation to check if this looks like an income statement
                        validation_terms = ['Tržby', 'Výkonová', 'Výsledek', 'náklady']
                        matches = sum(1 for term in validation_terms if term.lower() in context.lower())
                        
                        if matches >= 2:  # At least 2 key terms should be present
                            potential_sections.append({
                                'context': context,
                                'score': matches,  # Store the number of matches as a relevance score
                                'position': i  # Store the position in document
                            })
                
                if not potential_sections:
                    logger.warning("No potential income statement sections found")
                    return
                
                # Sort sections by score (most relevant first) and position (earlier in document first)
                potential_sections.sort(key=lambda x: (-x['score'], x['position']))
                
                # Combine contexts if they're not too far apart (within 200 lines)
                combined_context = potential_sections[0]['context']
                for i in range(1, len(potential_sections)):
                    if (potential_sections[i]['position'] - 
                        (potential_sections[i-1]['position'] + 130)) < 200:
                        combined_context += "\n...\n" + potential_sections[i]['context']
                
                # Prepare prompt for OpenAI
                prompt = f"""
                Extract key financial metrics from this income statement (Výkaz zisku a ztrát).
                The text is in Czech and might contain OCR errors.
                
                Please extract these specific values:
                1. Revenue from products and services (Tržby z prodeje výrobků a služeb)
                2. Revenue from goods (Tržby za prodej zboží)
                3. Production consumption (Výkonová spotřeba)
                4. Personnel costs (Osobní náklady)
                5. Operating profit (Provozní výsledek hospodaření)
                6. Profit before tax (Výsledek hospodaření před zdaněním)
                7. Net profit (Výsledek hospodaření za účetní období)
                
                Text from document:
                {combined_context}
                
                Return ONLY a JSON object with these keys:
                {
                    "revenue_from_products_and_services_current": number,
                    "revenue_from_goods_current": number,
                    "production_consumption_current": number,
                    "personnel_costs_current": number,
                    "operating_profit_current": number,
                    "profit_before_tax_current": number,
                    "net_profit_current": number
                }
                
                If a value cannot be found, use null. Remove any thousands separators and convert to integers.
                If you find multiple values for the same metric, use the most recent or most complete one.
                """
                
                # Call OpenAI API using the client
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a financial data extraction assistant. Extract only the requested metrics and return them in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )
                
                try:
                    # Parse the response
                    llm_extracted_data = json.loads(response.choices[0].message.content)
                    
                    # Validate the extracted data
                    if any(value is not None for value in llm_extracted_data.values()):
                        logger.info("Successfully extracted income statement data using LLM")
                        self.financial_data['income_statement'] = llm_extracted_data
                        return
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                except Exception as e:
                    logger.error(f"Error processing LLM response: {e}")
                    
            except Exception as e:
                logger.error(f"Error in LLM fallback strategy: {e}")
            
            logger.error("All income statement extraction strategies failed")
    
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
        """Extract comprehensive company information."""
        # Initialize with default values to ensure required fields exist
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
        
        try:
            # Extract all text from the document
            all_text = soup.get_text()
            
            # First try direct extraction before using OpenAI
            # Look for company name in typical locations
            company_patterns = [
                r'(?:Obchodní firma|Název společnosti|Company name):\s*([^\n]+)',
                r'(?:Společnost|Company):\s*([^\n]+)',
            ]
            
            for pattern in company_patterns:
                match = re.search(pattern, all_text, re.IGNORECASE)
                if match:
                    result['company_name'] = match.group(1).strip()
                    break
            
            # Extract year/accounting period
            year_pattern = r'(?:rok|year|období|period)\s*(20\d{2})'
            year_match = re.search(year_pattern, all_text, re.IGNORECASE)
            if year_match:
                result['accounting_period'] = year_match.group(1)
            
            # Only proceed with OpenAI if we have a client and are missing information
            if self.client and (not result['company_name'] or not result['accounting_period']):
                # ... existing OpenAI code ...
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                
                try:
                    api_response = json.loads(response.choices[0].message.content)
                    # Update result only for non-None values from API
                    for key, value in api_response.items():
                        if value is not None:
                            result[key] = value
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OpenAI response: {e}")
                    # Continue with what we have from direct extraction
            
            # Fallback values if still missing required fields
            if not result['company_name']:
                # Try to find any company-like name in the text
                company_candidates = re.findall(r'([A-Z][A-Za-z\s\.]+(?:a\.s\.|s\.r\.o\.|spol\. s r\.o\.))', all_text)
                if company_candidates:
                    result['company_name'] = company_candidates[0].strip()
                else:
                    result['company_name'] = "Unknown Company"
            
            if not result['accounting_period']:
                # Default to current year
                from datetime import datetime
                result['accounting_period'] = str(datetime.now().year)
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting company information: {e}")
            # Return default values instead of raising
            return {
                "accounting_period": str(datetime.now().year),
                "company_name": "Unknown Company",
                "legal_form": "Unknown",
                "main_activities": [],
                "employees": None,
                "established": None,
                "headquarters": None,
                "industry": "Unknown"
            }

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