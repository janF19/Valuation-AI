import json
import re
import logging
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, List, Tuple

import openai
from bs4 import BeautifulSoup
# Assuming settings holds your OpenAI key correctly
# If not, fallback logic using os.environ is included
try:
    from backend.config.settings import settings
    OPENAI_API_KEY_SOURCE = "settings"
except ImportError:
    settings = None
    OPENAI_API_KEY_SOURCE = "environment"
import os


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Add DEBUG level logging specifically for the extractor's fuzzy matching
# logging.getLogger('backend.src.financials.extractor2').setLevel(logging.DEBUG)
logger = logging.getLogger(__name__) # Get logger for the current module

# --- Constants ---
# Added more variations, including the exact header from your HTML
INCOME_KEYWORDS = ['VÝKAZ ZISKU A ZTRÁTY', 'VÝKAZ ZISKU A ZTRÁTY, druhové členění', 'VYKAZ ZISKU A ZTRATY', 'VÝKAZ ZISKU', 'VYKAZ ZISKU', 'VÝSLEDOVKA', 'Income Statement']
BALANCE_KEYWORDS = ['ROZVAHA', 'Rozvaha', 'Balance Sheet', 'BILANCE', 'AKTIVA', 'PASIVA']
CASHFLOW_KEYWORDS = ['PŘEHLED O PENĚŽNÍCH TOCÍCH', 'PREHLED O PENEZNICH TOCICH', 'CASH FLOW', 'PENĚŽNÍ TOKY', 'PENEZNI TOKY', 'Přehled o peněžních tocích']

# LLM Model Configuration
LLM_MODEL = "gpt-4o"
MAX_CONTEXT_CHARS = 12000 # Keep increased context
CONTEXT_LINES_BEFORE = 2
CONTEXT_LINES_AFTER = 90
INFO_CONTEXT_LINES = 90

class FinancialExtractor:
    """
    Extracts financial data from HTML reports using an LLM-first approach.
    Focuses on robust section finding and detailed LLM prompts for extraction.
    Handles cases where LLM returns nulls by storing empty dicts.
    """

    def __init__(self, openai_api_key: Optional[str] = None):
        # ... (Initialization logic - unchanged) ...
        self.financial_data = self._initialize_financial_data()
        api_key_to_use = openai_api_key

        if not api_key_to_use:
            if OPENAI_API_KEY_SOURCE == "settings" and settings and hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                api_key_to_use = settings.OPENAI_API_KEY
                logger.info("Loaded OpenAI API key from settings.")
            elif os.environ.get("OPENAI_API_KEY"):
                 api_key_to_use = os.environ.get("OPENAI_API_KEY")
                 logger.info(f"Loaded OpenAI API key from environment variable (Source: {OPENAI_API_KEY_SOURCE}).")

        if api_key_to_use:
            try:
                self.client = openai.OpenAI(api_key=api_key_to_use)
                self.client.models.list() # Verify key
                logger.info(f"OpenAI client initialized and key verified using model: {LLM_MODEL}")
            except openai.AuthenticationError:
                 logger.error("OpenAI Authentication Error: The provided API key is invalid or expired.")
                 self.client = None
            except Exception as e:
                 logger.error(f"Failed to initialize OpenAI client: {e}")
                 self.client = None
        else:
            self.client = None
            logger.warning("No OpenAI API key provided or found. LLM extraction will not be possible.")


    def _initialize_financial_data(self) -> Dict[str, Dict]:
        # ... (Unchanged) ...
        return {
            'income_statement': {},
            'balance_sheet': {},
            'cash_flow': {},
            'information': {}
        }

    def _clean_text(self, text: str) -> str:
        """More aggressive cleaning for comparison."""
        text = ' '.join(text.split()) # Normalize whitespace
        text = text.replace('\xa0', ' ') # Replace non-breaking spaces
        return text.lower().strip()

    def _fuzzy_match_score(self, s1_clean: str, s2_clean: str) -> float:
        """Calculates similarity score between two already cleaned lowercase strings."""
        # Removed cleaning from here, expects pre-cleaned input
        return SequenceMatcher(None, s1_clean, s2_clean).ratio()

    def _find_section_context(self,
                              text_lines: List[str],
                              keywords: List[str],
                              threshold: float = 0.85) -> Optional[Tuple[str, int]]: # Raised threshold slightly, rely on better matching
        """
        Refined section finding: Cleans text better, checks for containment,
        and prioritizes strong matches.
        """
        best_match_score = 0.0
        best_match_index = -1
        best_match_line_original = ""
        cleaned_keywords = [self._clean_text(kw) for kw in keywords]

        logger.debug(f"Searching for keywords like '{cleaned_keywords[0]}' with threshold {threshold}")

        for i, line in enumerate(text_lines):
            line_original = line.strip() # Keep original for context
            if not line_original:
                 continue

            line_clean = self._clean_text(line_original)
            if not line_clean: # Skip if cleaning results in empty string
                 continue

            # --- Debug specific lines if needed ---
            # if 15 < i < 30 or 100 < i < 115: # Example line ranges
            #      logger.debug(f"Line {i} Clean: '{line_clean}'")
            # ------------------------------------

            current_line_best_score_for_any_keyword = 0.0
            for kw_clean in cleaned_keywords:
                # 1. Check for high-similarity containment (keyword inside the line)
                score_contained = 0.0
                if kw_clean in line_clean: # Quick check
                     # Use SequenceMatcher on the *substring* for better fuzzy containment
                     matcher = SequenceMatcher(None, kw_clean, line_clean)
                     # Find best block match within the line
                     match = matcher.find_longest_match(0, len(kw_clean), 0, len(line_clean))
                     if match and match.size >= len(kw_clean) * 0.95: # Needs high similarity block
                         # Calculate score based on how well the best block matches keyword
                         block_similarity = SequenceMatcher(None, kw_clean, line_clean[match.b:match.b + match.size]).ratio()
                         score_contained = block_similarity * 0.98 # Slightly penalize just being contained
                         # logger.debug(f"  Contained Match: '{kw_clean}' in '{line_clean}' -> Block Sim: {block_similarity:.2f}, Score: {score_contained:.2f}")


                # 2. Check for similarity if line *starts with* keyword (or is very close)
                score_prefix = 0.0
                # Compare against a segment slightly larger than the keyword
                line_segment_clean = line_clean[:len(kw_clean) + 5]
                prefix_sim = self._fuzzy_match_score(kw_clean, line_segment_clean)
                if prefix_sim > 0.8: # Only consider strong prefix matches
                     score_prefix = prefix_sim
                     # logger.debug(f"  Prefix Match: '{kw_clean}' vs '{line_segment_clean}' -> Score: {score_prefix:.2f}")


                # 3. Direct comparison (whole line vs keyword) - less likely but possible
                score_direct = self._fuzzy_match_score(kw_clean, line_clean)
                # logger.debug(f"  Direct Match: '{kw_clean}' vs '{line_clean}' -> Score: {score_direct:.2f}")


                # Combine: Take the highest score found for this keyword on this line
                line_keyword_best_score = max(score_contained, score_prefix, score_direct)

                # Update the best score found *for this specific line* across all keywords
                current_line_best_score_for_any_keyword = max(current_line_best_score_for_any_keyword, line_keyword_best_score)


            # Update the overall best score if this line's best score is higher
            if current_line_best_score_for_any_keyword > best_match_score:
                best_match_score = current_line_best_score_for_any_keyword
                best_match_index = i
                best_match_line_original = line_original # Store the original non-cleaned line
                # logger.debug(f"==> New Best Overall Score: {best_match_score:.2f} at Line {i}: '{line_original}'")


        # Final Check against threshold
        if best_match_score >= threshold:
            logger.info(f"Found potential section header '{best_match_line_original}' (Best Score: {best_match_score:.2f}) at line {best_match_index} for keywords starting with '{keywords[0]}'.")

            start_line = max(0, best_match_index - CONTEXT_LINES_BEFORE)
            end_line = min(len(text_lines), best_match_index + CONTEXT_LINES_AFTER)

            # --- Extract context using ORIGINAL lines ---
            context_lines_original = text_lines[start_line:end_line]
            context_text = "\n".join(context_lines_original).strip()
            # -----------------------------------------

            if len(context_text) > MAX_CONTEXT_CHARS:
                original_len = len(context_text)
                context_text = context_text[:MAX_CONTEXT_CHARS]
                logger.warning(f"Truncated context from {original_len} to {MAX_CONTEXT_CHARS} characters for LLM.")

            return context_text, best_match_index
        else:
            logger.warning(f"No section found for keywords starting with '{keywords[0]}...' with threshold {threshold}. Best score found: {best_match_score:.2f}")
            return None

    def _all_values_null(self, data: Optional[Dict[str, Any]]) -> bool:
        # ... (Unchanged) ...
        if not data or not isinstance(data, dict):
            return True
        return all(value is None for value in data.values())


    def _call_llm_for_extraction(self, context: str, prompt_template: str, section_name: str) -> Optional[Dict[str, Any]]:
        # ... (LLM Call logic - unchanged, still includes null check) ...
        if not self.client:
            logger.error(f"Cannot extract {section_name}: OpenAI client not initialized.")
            return None
        if not context:
            logger.warning(f"Cannot extract {section_name}: No context provided.")
            return None

        prompt = prompt_template.format(context=context)

        try:
            logger.info(f"Sending request to LLM for {section_name} extraction...")

            response = self.client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a highly accurate financial data extraction assistant expert in Czech financial documents, including those with OCR errors. Return ONLY the requested valid JSON object. Use `null` only if a value is genuinely missing or unreadable."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.05,
                response_format={"type": "json_object"}
            )

            response_content = response.choices[0].message.content
            logger.info(f"LLM response received for {section_name}.")
            # logger.debug(f"LLM Raw Response for {section_name}: {response_content}")

            try:
                extracted_data = json.loads(response_content)
                if not isinstance(extracted_data, dict):
                     logger.error(f"LLM did not return a JSON object for {section_name}. Response: {response_content}")
                     return None
                logger.info(f"Successfully parsed LLM JSON response for {section_name}.")
                logger.debug(f"Parsed LLM data for {section_name}: {json.dumps(extracted_data)}") # Log the parsed data

                if self._all_values_null(extracted_data):
                    logger.warning(f"LLM returned only null values for {section_name}. Treating as extraction failure.")
                    return None

                return extracted_data
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse LLM JSON response for {section_name}: {json_err}")
                logger.error(f"LLM Raw Response causing error: {response_content}")
                # Regex fallback omitted for brevity, but can be added back if needed
                return None

        except openai.APIError as api_err:
            logger.error(f"OpenAI API error during {section_name} extraction: {api_err}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected error during {section_name} LLM extraction: {e}", exc_info=True)
            return None


    # --- LLM Prompt Templates (Modified Company Info Prompt) ---
    COMPANY_INFO_PROMPT = """
    Extract the following company information from the provided text, which comes from the beginning of a Czech financial report (likely OCR'd). The information is typically found in the section titled "IDENTIFIKAČNÍ ÚDAJE SPOLEČNOSTI" or similar. Return ONLY a valid JSON object containing the specified keys. Use `null` if a value cannot be reliably found or is explicitly missing. Ensure numbers are integers.

    Text Context:
    ---
    {context}
    ---

    Required JSON Object Structure:
    {{
        "IC": "string or null (Find IČ or IČO, typically 8 digits - for example '25849026')",
        "registered_capital": "string or null (Find 'Základní kapitál', might include currency like Kč, for example '530000 000,- Kč')",
        "employee_count": "integer or null (Find 'Počet zaměstnanců' or words like 'Průměrný přepočtený stav', might be mentioned at the end of the document)",
        "accounting_period": "string or null (Find the reporting year, e.g., '2023' from document title or 'ke dni: 31.12.2023')",
        "company_name": "string or null (Find full company name like 'VÍTKOVICE CYLINDERS a.s.')",
        "legal_form": "string or null (Find 'Právní forma', e.g., 'akciová společnost')",
        "main_activities": ["string", ...] or [] (Find 'Hlavní činnost' or 'Předmět podnikání' - list activities mentioned),
        "established": "string or null (Find 'Datum vzniku / založení' or 'Den zápisu do obchodního rejstříku', if available)",
        "headquarters": "string or null (Find 'Sídlo', the address like 'Vítkovice 3041, 70300 Ostrava')",
        "news": "string or null (Find summary of what happened and save most important information in few sentences, usually at start of document in section 'ÚVODNÍ SLOVO GENERÁLNÍHO ŘEDITELE')",
        "industry": "string or null (Infer based on activities/description. Choose from: [Advertising, Aerospace/Defense, Apparel, Auto & Truck, Auto Parts, Beverage (Alcoholic), Beverage (Soft), Broadcasting, Building Materials, Business & Consumer Services, Cable TV, Chemical (Basic), Chemical (Diversified), Chemical (Specialty), Coal & Related Energy, Computer Services, Computers/Peripherals, Construction Supplies, Diversified, Drugs (Pharmaceutical), Education, Electrical Equipment, Electronics (General), Engineering/Construction, Farming/Agriculture, Food Processing, Food Wholesalers, Furn/Home Furnishings, Homebuilding, Hotel/Gaming, Household Products, Information Services, Machinery, Manufacturing, Metals & Mining, Office Equipment & Services, Paper/Forest Products, Power, Real Estate, Recreation, Restaurant/Dining, Retail, Rubber& Tires, Semiconductor, Software, Steel, Telecommunications, Transportation, Trucking, Utility]. Default to 'Manufacturing' or 'Services' if unclear.)"
    }}

    Provide ONLY the JSON object. Do not add explanations. Pay special attention to find all values listed in the identification section, even if they appear on different lines or in other parts of the document.
    """

    INCOME_STATEMENT_PROMPT = """
    Extract key financial metrics from the provided text, expected to be a Czech Income Statement (Výkaz zisku a ztráty), possibly with OCR errors. Focus on the CURRENT accounting period ('běžném období'). Identify the correct column for the current period - it's usually the first numeric column after the text description, often labeled 'běžném' or '1'. Return ONLY a valid JSON object with the specified keys. Use `null` if a value is genuinely missing or unreadable. Convert values to integers (remove spaces, currency symbols like '$'). Handle negative numbers (parentheses or minus signs).

    Text Context:
    ---
    {context}
    ---

    Required JSON Object Structure (CURRENT PERIOD ONLY):
    {{
        "revenue_from_products_and_services_current": "integer or null (Find 'Tržby z prodeje výrobků a služeb' or 'I.', look in the 'běžném' column)",
        "revenue_from_goods_current": "integer or null (Find 'Tržby za prodej zboží' or 'II.', look in the 'běžném' column)",
        "production_consumption_current": "integer or null (Find 'Výkonová spotřeba' usually marked 'A.' or similar, look in the 'běžném' column)",
        "personnel_costs_current": "integer or null (Find 'Osobní náklady' usually marked 'D.' or similar, look in the 'běžném' column)",
        "wage_costs_current": "integer or null (Find 'Mzdové náklady' often 'D.1.', look in the 'běžném' column)",
        "depreciation_current": "integer or null (Find 'Odpisy...' or 'Úpravy hodnot v provozní oblasti' usually 'E.' or similar, look in the 'běžném' column)",
        "operating_profit_current": "integer or null (Find 'Provozní výsledek hospodaření' or 'Výsledek hospodaření z provozní činnosti', often marked '*' or similar, look in 'běžném' column)",
        "ebit_current": "integer or null (Report the same value as operating_profit_current)"
    }}

    Provide ONLY the JSON object. Be very careful to extract from the CURRENT ('běžném' or column '1') period column.
    """

    BALANCE_SHEET_PROMPT = """
    Extract key financial metrics from the provided text, expected to be a Czech Balance Sheet (Rozvaha), possibly with OCR errors. Extract values for BOTH the CURRENT ('Běžné účetní období', often 'Netto' column 3) and PREVIOUS ('Minulé úč. období', often 'Netto' column 4) periods. Identify the 'Netto' columns for assets and the corresponding columns for liabilities/equity. Return ONLY a valid JSON object. Use `null` if a value is genuinely missing or unreadable. Convert values to integers (remove spaces, currency). Handle negative numbers.

    Text Context:
    ---
    {context}
    ---

    Required JSON Object Structure:
    {{
        "total_assets_current": "integer or null (Find 'AKTIVA CELKEM', use the CURRENT 'Netto' column)",
        "total_assets_previous": "integer or null (Find 'AKTIVA CELKEM', use the PREVIOUS 'Netto' column)",
        "intangible_assets_current": "integer or null (Find 'Dlouhodobý nehmotný majetek', often 'B.I.', use CURRENT 'Netto')",
        "intangible_assets_previous": "integer or null (Find 'Dlouhodobý nehmotný majetek', often 'B.I.', use PREVIOUS 'Netto')",
        "tangible_assets_current": "integer or null (Find 'Dlouhodobý hmotný majetek', often 'B.II.', use CURRENT 'Netto')",
        "tangible_assets_previous": "integer or null (Find 'Dlouhodobý hmotný majetek', often 'B.II.', use PREVIOUS 'Netto')",
        "current_assets_current": "integer or null (Find 'Oběžná aktiva', often 'C.', use CURRENT 'Netto')",
        "current_assets_previous": "integer or null (Find 'Oběžná aktiva', often 'C.', use PREVIOUS 'Netto')",
        "total_liabilities_equity_current": "integer or null (Find 'PASIVA CELKEM', use the CURRENT period column)",
        "total_liabilities_equity_previous": "integer or null (Find 'PASIVA CELKEM', use the PREVIOUS period column)",
        "equity_current": "integer or null (Find 'Vlastní kapitál', often 'A.', use CURRENT period column)",
        "equity_previous": "integer or null (Find 'Vlastní kapitál', often 'A.', use PREVIOUS period column)",
        "liabilities_current": "integer or null (Find 'Cizí zdroje', often 'B.', use CURRENT period column)",
        "liabilities_previous": "integer or null (Find 'Cizí zdroje', often 'B.', use PREVIOUS period column)"
    }}

    Provide ONLY the JSON object. Focus on the 'Netto' columns (usually 3 and 4) for Assets and the corresponding Current/Previous columns for Liabilities/Equity.
    """

    CASH_FLOW_PROMPT = """
    Extract key financial metrics from the provided text, expected to be a Czech Cash Flow Statement (Přehled o peněžních tocích), possibly with OCR errors. Extract values for BOTH the CURRENT ('běžné období') and PREVIOUS ('minulé období') accounting periods. Identify the correct columns. Return ONLY a valid JSON object. Use `null` if a value is genuinely missing or unreadable. Convert values to integers (remove spaces, currency). Handle negative numbers (parentheses or minus signs). CAPEX is usually negative.

    Text Context:
    ---
    {context}
    ---

    Required JSON Object Structure:
    {{
        "initial_cash_balance_current": "integer or null (Find 'Počáteční stav peněžních prostředků', CURRENT column)",
        "initial_cash_balance_previous": "integer or null (Find 'Počáteční stav peněžních prostředků', PREVIOUS column)",
        "profit_before_tax_current": "integer or null (Find 'Výsledek hospodaření před zdaněním' or 'Zisk před zdaněním', CURRENT column)",
        "profit_before_tax_previous": "integer or null (Find 'Výsledek hospodaření před zdaněním' or 'Zisk před zdaněním', PREVIOUS column)",
        "net_operating_cash_flow_current": "integer or null (Find 'Čistý peněžní tok z provozní činnosti', often 'A.*', CURRENT column)",
        "net_operating_cash_flow_previous": "integer or null (Find 'Čistý peněžní tok z provozní činnosti', often 'A.*', PREVIOUS column)",
        "capex_current": "integer or null (Find 'Výdaje spojené s nabytím dl. majetku' / 'Výdaje ... stálých aktiv', often 'B.1.', CURRENT column, usually negative)",
        "capex_previous": "integer or null (Find 'Výdaje spojené s nabytím dl. majetku' / 'Výdaje ... stálých aktiv', often 'B.1.', PREVIOUS column, usually negative)",
        "proceeds_from_sale_of_fixed_assets_current": "integer or null (Find 'Příjmy z prodeje dl. majetku' / 'Příjmy ... stálých aktiv', often 'B.2.', CURRENT column)",
        "proceeds_from_sale_of_fixed_assets_previous": "integer or null (Find 'Příjmy z prodeje dl. majetku' / 'Příjmy ... stálých aktiv', often 'B.2.', PREVIOUS column)"
        // "ending_cash_balance_current": "integer or null (Find 'Konečný stav peněžních prostředků', CURRENT column)",
        // "ending_cash_balance_previous": "integer or null (Find 'Konečný stav peněžních prostředků', PREVIOUS column)",
    }}

    Provide ONLY the JSON object. Carefully map line items to the correct CURRENT and PREVIOUS columns.
    """


    # --- Main Extraction Method ---
    def extract_from_html(self, html_content: str) -> Dict[str, Any]:
        """
        Extracts financial data from HTML content using fuzzy search and LLM.
        """
        # ... (Guard clauses for client and html_content - unchanged) ...
        if not self.client:
             logger.error("Extraction cancelled: OpenAI client is not initialized.")
             return self._initialize_financial_data()

        if not html_content or not isinstance(html_content, str) or len(html_content.strip()) == 0:
            logger.error("Invalid or empty HTML content provided.")
            return self._initialize_financial_data()

        logger.info("Starting financial data extraction from HTML...")
        self.financial_data = self._initialize_financial_data()

        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            # Try extracting text preserving paragraphs better
            text_parts = []
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'table']):
                 # Get text from tables differently maybe? For now, standard get_text
                 text = element.get_text(separator=' ', strip=True)
                 if text:
                      text_parts.append(text)
                      # Add extra newline after headers and tables might help structure
                      if element.name.startswith('h') or element.name == 'table':
                           text_parts.append("\n") # Explicit newline

            full_text = "\n".join(text_parts)
            text_lines = [line.strip() for line in full_text.split('\n') if line.strip()]

            # Fallback if above method fails
            if not text_lines or len(text_lines) < 10:
                 logger.warning("Paragraph/header text extraction yielded few lines, falling back to simple get_text.")
                 full_text_simple = soup.get_text()
                 text_lines = [line.strip() for line in full_text_simple.split('\n') if line.strip()]

            logger.info(f"Processed HTML into {len(text_lines)} non-empty text lines.")
            # Optional: Log first few lines to check extraction quality
            # logger.debug("First 20 extracted lines:")
            # for i, line in enumerate(text_lines[:20]):
            #      logger.debug(f"  Line {i}: {line}")

            # --- Extraction Sequence (Modified for better company information extraction) ---
            sections_to_extract = [
                ("Company Information", INFO_CONTEXT_LINES, self.COMPANY_INFO_PROMPT, 'information', None),
                ("Income Statement", CONTEXT_LINES_AFTER, self.INCOME_STATEMENT_PROMPT, 'income_statement', INCOME_KEYWORDS),
                ("Balance Sheet", CONTEXT_LINES_AFTER, self.BALANCE_SHEET_PROMPT, 'balance_sheet', BALANCE_KEYWORDS),
                ("Cash Flow Statement", CONTEXT_LINES_AFTER, self.CASH_FLOW_PROMPT, 'cash_flow', CASHFLOW_KEYWORDS, 0.75),  # Lower threshold
            ]

            for section_info in sections_to_extract:
                if len(section_info) == 5:
                    name, context_size, prompt, data_key, keywords = section_info
                    threshold = 0.85  # Default threshold
                else:
                    name, context_size, prompt, data_key, keywords, threshold = section_info
                
                logger.info(f"Attempting to extract {name}...")
                context = None
                context_tuple = None

                if keywords:
                    context_tuple = self._find_section_context(text_lines, keywords, threshold)
                    if context_tuple:
                        context, _ = context_tuple
                elif name == "Company Information":
                    context = "\n".join(text_lines[:context_size]).strip()
                    if len(context) > MAX_CONTEXT_CHARS:
                         context = context[:MAX_CONTEXT_CHARS]
                         logger.warning(f"Truncated context for {name} to {MAX_CONTEXT_CHARS} chars.")

                if context:
                    extracted_data = self._call_llm_for_extraction(context, prompt, name)
                    if extracted_data:
                        self.financial_data[data_key] = extracted_data
                        logger.info(f"Successfully extracted data for {name}.")
                    else:
                        logger.warning(f"LLM extraction failed or returned only nulls for {name}. Storing empty dictionary.")
                        self.financial_data[data_key] = {}
                elif keywords:
                    logger.warning(f"Could not locate {name} section in the document. Storing empty dictionary.")
                    self.financial_data[data_key] = {}
                else:
                     logger.warning(f"LLM extraction failed for {name} (Info). Storing empty dictionary.")
                     self.financial_data[data_key] = {}


            logger.info("Financial data extraction process completed.")

        except Exception as e:
            logger.error(f"Critical error during HTML processing or extraction orchestration: {e}", exc_info=True)
            return self.financial_data

        return self.financial_data


# --- Example Usage (Unchanged) ---
if __name__ == "__main__":
    # --- Make sure to set logging level to DEBUG for the extractor if needed ---
    # logging.getLogger('__main__').setLevel(logging.DEBUG) # Or use the module name if different
    # -------------------------------------------------------------------------

    extractor = FinancialExtractor()

    input_html_file = 'output_vz_2023_isotra.html' # Or 'isotra_2020.html' if that's the target
    output_json_file = 'financial_data_llm_fixed_v3.json'

    if not extractor.client:
         print(f"\nError: OpenAI client not initialized. Please ensure API key is valid and accessible via {OPENAI_API_KEY_SOURCE}.")
    else:
        try:
            logger.info(f"Reading HTML content from: {input_html_file}")
            with open(input_html_file, 'r', encoding='utf-8') as f:
                html_content_example = f.read()
            logger.info("HTML content read successfully.")

            logger.info("Starting extraction...")
            financial_data_output = extractor.extract_from_html(html_content_example)
            logger.info("Extraction finished.")

            logger.info(f"Saving extracted data to: {output_json_file}")
            with open(output_json_file, 'w', encoding='utf-8') as f:
                json.dump(financial_data_output, f, indent=4, ensure_ascii=False)
            logger.info("Data saved successfully.")

            print("\n--- Extraction Summary ---")
            print(f"Data saved to: {output_json_file}")
            print(f"Company Info extracted: {'Yes' if financial_data_output.get('information') else 'No'}")
            print(f"Income Statement extracted: {'Yes' if financial_data_output.get('income_statement') else 'No'}")
            print(f"Balance Sheet extracted: {'Yes' if financial_data_output.get('balance_sheet') else 'No'}")
            print(f"Cash Flow extracted: {'Yes' if financial_data_output.get('cash_flow') else 'No'}")
            print("------------------------\n")

        except FileNotFoundError:
            logger.error(f"Error: Input HTML file not found at '{input_html_file}'")
            print(f"Error: Input HTML file not found at '{input_html_file}'")
        except openai.AuthenticationError:
             logger.error("OpenAI Authentication Error: Check your API key.")
             print("OpenAI Authentication Error: Check your API key.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in main execution: {e}", exc_info=True)
            print(f"An unexpected error occurred: {str(e)}")