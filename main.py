import os
import json
import re
import datetime
import tempfile
import shutil
import logging
import traceback
from typing import Optional, Dict, Any, List, Tuple, Mapping


try:
    from google import generativeai as genai
    from google.generativeai import types
except ImportError:
    print("Error: google-generativeai package not found. Install: pip install google-generativeai")
    exit(1)
try:
    from asteval import Interpreter
except ImportError:
    print("Error: asteval package not found. Install: pip install asteval")
    exit(1)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

API_KEY = os.environ.get("GOOGLE_API_KEY")
DEFAULT_MODEL_NAME = os.environ.get("LLM_MODEL", "gemini-2.0-flash")
JSON_FILE = os.environ.get("AGENT_MEMORY_FILE", "agent_memory.json")


CALC_REF_PREFIX_CPT = "___" 
DELIMITER_CPT = "__"

MAX_CONTEXT_ITEMS = 7
MAX_CALCULATION_ITERATIONS = 20
SAFE_ASTEVAL_FUNCS = {'round': round, 'abs': abs, 'max': max, 'min': min}

DEFAULT_SAFETY_SETTINGS = {
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}


try:
    
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_INSTRUCTION_COMPACT_TEXT = f.read()
except FileNotFoundError:
    logging.error("system_prompt.txt not found! Agent will likely fail.")
    SYSTEM_INSTRUCTION_COMPACT_TEXT = "ERROR: Compact V2 system prompt file missing." 


def _load_data(filepath: str = JSON_FILE) -> Dict[str, List[Dict[str, Any]]]:
    """Loads data (only 'facts') from the JSON file."""
    try:
        if not os.path.exists(filepath):
            logging.info(f"Data file '{filepath}' not found. Returning empty structure.")
            return {"facts": []} 
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                 logging.error(f"Invalid data format in {filepath}: expected dict. Returning empty.")
                 return {"facts": []}
            if "facts" not in data or not isinstance(data.get("facts"), list):
                logging.warning(f"'facts' key missing/invalid in {filepath}. Initializing.")
                data["facts"] = []
            
            return {"facts": data.get("facts", [])} 
    except json.JSONDecodeError:
        logging.exception(f"Error decoding JSON from {filepath}. Returning empty structure.")
        return {"facts": []}
    except Exception as e:
        logging.exception(f"Unexpected error loading data from {filepath}: {e}")
        raise IOError(f"Failed to load data from {filepath}") from e

def _save_data(data: Dict[str, List[Dict[str, Any]]], filepath: str = JSON_FILE) -> None:
    """Saves data (only 'facts') to the JSON file atomically."""
    
    data_to_save = {"facts": data.get("facts", [])}
    temp_path = ""
    try:
        target_dir = os.path.dirname(filepath) or '.'
        os.makedirs(target_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', delete=False, dir=target_dir, suffix='.tmp') as tmp_file:
            
            json.dump(data_to_save, tmp_file, indent=4, ensure_ascii=False)
            temp_path = tmp_file.name
        try:
            os.replace(temp_path, filepath)
            logging.debug(f"Atomically replaced '{filepath}' with '{temp_path}'.")
        except OSError:
            shutil.move(temp_path, filepath)
            logging.debug(f"Moved temp file '{temp_path}' to '{filepath}'.")
    except Exception as e:
        logging.exception(f"Error saving data to {filepath}: {e}")
        if temp_path and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except OSError as rm_err: logging.error(f"Could not remove temp file {temp_path}: {rm_err}")
        raise IOError(f"Failed to save data to {filepath}") from e

def _init_json_store(filepath: str = JSON_FILE) -> None:
    """Initializes the JSON store (only 'facts') if it doesn't exist."""
    if not os.path.exists(filepath):
        logging.info(f"JSON store file '{filepath}' not found. Creating...")
        try:
            _save_data({"facts": []}, filepath) 
            logging.info(f"JSON store '{filepath}' created successfully.")
        except IOError as e:
            logging.error(f"Failed to create initial JSON store '{filepath}': {e}")
            raise
    else:
        
        logging.info(f"JSON store '{filepath}' already exists.")
        

def _get_next_id(data: Dict[str, List[Dict[str, Any]]], list_key: str = "facts") -> int:
    """Gets the next available integer ID for the 'facts' list."""
    
    items = data.get(list_key, [])
    if not items: return 1
    max_id = 0
    for item in items:
        item_id = item.get("id")
        if isinstance(item_id, int): max_id = max(max_id, item_id)
        elif isinstance(item_id, str) and item_id.isdigit():
             try: max_id = max(max_id, int(item_id))
             except ValueError: pass 
    return max_id + 1


class CompactAgent: 
    """ AI agent exclusively using the ultra-compact plain text format V2 (no /R). """
    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = DEFAULT_MODEL_NAME,
                 json_filepath: str = JSON_FILE,
                 safety_settings: Optional[Dict[Any, Any]] = None):
        """ Initializes the agent for the compact format. """
        self.api_key = api_key or API_KEY
        if not self.api_key: raise ValueError("API key required via argument or GEMINI_API_KEY env var.")

        self.model_name = model_name
        self.json_filepath = json_filepath
        self.safety_settings = safety_settings if safety_settings is not None else DEFAULT_SAFETY_SETTINGS

        
        self.calc_ref_prefix = CALC_REF_PREFIX_CPT
        self.delimiter = DELIMITER_CPT
        logging.info("Agent configured for Compact Plain Text V2 format.")

        
        self.aeval = Interpreter(builtins_allowed=False, minimal=True, use_numpy=False)
        for name, func in SAFE_ASTEVAL_FUNCS.items(): self.aeval.symtable[name] = func
        logging.info(f"asteval initialized with functions: {list(self.aeval.symtable.keys())}")

        
        _init_json_store(self.json_filepath)

        
        system_instruction_text = SYSTEM_INSTRUCTION_COMPACT_TEXT

        
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                self.model_name,
                system_instruction=system_instruction_text,
                safety_settings=self.safety_settings
            )
            logging.info(f"Gemini client initialized for model: {self.model_name} using Compact V2 format.")
        except Exception as e:
            logging.exception(f"Error initializing Gemini client: {e}")
            raise ConnectionError(f"Failed to initialize Gemini client: {e}") from e

    def _get_relevant_context(self, user_prompt: str, max_items: int = MAX_CONTEXT_ITEMS) -> str:
        """ Retrieves context formatted simply. """
        context_lines = []
        try:
            
            data = _load_data(self.json_filepath)
            facts = data.get("facts", []) 

            def sort_key(item):
                ts = item.get('last_updated', '')
                try: return datetime.datetime.fromisoformat(ts.replace('Z', '+00:00'))
                except: return datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

            sorted_facts = sorted(facts, key=sort_key, reverse=True)

            if sorted_facts:
                context_lines.append("PREVIOUS_CONTEXT:")
                for fact in sorted_facts[:max_items]:
                    value_str = str(fact.get('value', 'NA'))[:60]
                    context_lines.append(f"{fact.get('entity_type','T')} {fact.get('entity_key','K')}: {value_str}")
        except Exception as e:
            logging.error(f"Error fetching context: {e}")
            context_lines.append("CONTEXT_ERROR")
        return "\n".join(context_lines)

    def _evaluate_expression(self, expression: str, context_str: str = "expression") -> Tuple[Optional[Any], Optional[str]]:
        """ Safely evaluates a mathematical expression string using asteval. (No change) """
        if not isinstance(expression, str) or not expression.strip(): return None, f"Invalid expr {context_str}"
        logging.info(f"Evaluating {context_str}: '{expression}'")
        try:
            result = self.aeval(expression);
            if self.aeval.error:
                err=f"Eval Error {context_str} '{expression}': {self.aeval.error_msg}"; self.aeval.error=[]; self.aeval.error_msg=None; logging.error(err); return None, err
            logging.info(f"Eval result for '{expression}': {result}"); return result, None
        except Exception as e: err=f"Eval Excp {context_str} '{expression}': {e}"; logging.error(err, exc_info=False); self.aeval.error=[]; self.aeval.error_msg=None; return None, err

    def _perform_inline_calculations(self, text: str, expression_map: Mapping[str, str]) -> str:
        """ Finds ___<id> markers, looks up expression, evaluates, and replaces. (No change) """
        if not isinstance(text, str) or not expression_map: return str(text)
        processed_text = text; iterations = 0
        id_pattern = re.escape(self.calc_ref_prefix) + r'(\d+)'; calculation_pattern = re.compile(id_pattern)
        while iterations < MAX_CALCULATION_ITERATIONS:
            match = calculation_pattern.search(processed_text);
            if not match: break
            iterations += 1; marker_full = match.group(0); ref_id_num = match.group(1); ref_id_full = marker_full
            replacement = f"[CalcErr:NoExpr {ref_id_full}]"
            try:
                expression = expression_map.get(ref_id_full)
                if expression:
                    result, error_msg = self._evaluate_expression(expression, f"inline calc ref {ref_id_full}")
                    if error_msg: replacement = f"[CalcErr:{error_msg} ref {ref_id_full}]"; logging.warning(f"Failed calc ref {ref_id_full}: {error_msg}. Expr: '{expression}'")
                    else: replacement = str(result)
                else: logging.warning(f"Ref ID '{ref_id_full}' not found in map. Keys: {list(expression_map.keys())}")
            except Exception as e: logging.error(f"Error processing inline calc ref '{ref_id_full}': {e}", exc_info=True); replacement = f"[CalcErr:Proc {ref_id_full}]"
            processed_text = processed_text.replace(marker_full, replacement, 1)
        if iterations == MAX_CALCULATION_ITERATIONS: logging.warning(f"Max inline calc iterations ({MAX_CALCULATION_ITERATIONS}) reached.")
        return processed_text

    def _execute_crud(self, crud_data: Dict[str, Any]) -> str:
        """ Executes a single CRUD operation. Expects pre-calculated value. (No change) """
        action=crud_data.get("action"); etype=crud_data.get("entity_type"); ekey=crud_data.get("entity_key"); value=crud_data.get("value")
        required={"action","entity_type","entity_key"}; missing = required - crud_data.keys()
        if missing: return f"StoreErr:Miss Fields ({', '.join(missing)}) in {crud_data}"
        if action not in ["create", "update", "delete"]: return f"StoreErr:Invalid Action '{action}'"
        if action != 'delete' and value is None: return f"StoreErr:'value' is None for action '{action}'"
        msg="";
        try:
            data=_load_data(self.json_filepath); facts=data.get("facts",[]) 
            if not isinstance(facts, list): logging.warning("Facts data invalid, resetting."); facts=[]
            now=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')+"Z"; idx=-1
            for i,f in enumerate(facts):
                if isinstance(f,dict) and f.get("entity_type")==etype and f.get("entity_key")==ekey: idx=i; break
            store_value = str(value) if value is not None else None
            if action=="create":
                if idx!=-1: facts[idx]['value']=store_value; facts[idx]['last_updated']=now; msg=f"StoreUpd(on create):{ekey}({etype})"
                else: nid=_get_next_id({"facts": facts}); nf={"id":nid,"entity_type":etype,"entity_key":ekey,"value":store_value,"last_updated":now}; facts.append(nf); msg=f"StoreCreate:{ekey}({etype}) ID:{nid}"
            elif action=="update":
                if idx!=-1: facts[idx]['value']=store_value; facts[idx]['last_updated']=now; msg=f"StoreUpdate:{ekey}({etype}) ID:{facts[idx].get('id')}"
                else: nid=_get_next_id({"facts": facts}); nf={"id":nid,"entity_type":etype,"entity_key":ekey,"value":store_value,"last_updated":now}; facts.append(nf); msg=f"StoreCreate(on update miss):{ekey}({etype}) ID:{nid}"
            elif action=="delete":
                if idx!=-1: d=facts.pop(idx); v=str(d.get('value','NA'))[:20]; msg=f"StoreDelete:{ekey}({etype}) ID:{d.get('id','NA')} Val:'{v}...'"
                else: msg=f"StoreInfo:NotFound for delete {ekey}({etype})"
            _save_data({"facts": facts}, self.json_filepath) 
        except Exception as e: msg=f"StoreFail Action:{action} Key:{ekey} Err:{e}"; logging.exception(f"CRUD execution failed: {msg}")
        return msg

    def _parse_compact_format(self, structured_line: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], List[str]]:
        """ Parses the ultra-compact structured line V2. Handles pre-calculation. (No change) """
        crud_ops_to_prepare = []; expression_map: Dict[str, str] = {}; parsing_messages = []; expression_counter = -1
        if not structured_line or not isinstance(structured_line, str):
            parsing_messages.append("ParseError CPT: Input line empty/not string."); return [], {}, parsing_messages
        blocks = structured_line.strip().split(self.delimiter)
        if not blocks or not blocks[0]:
            parsing_messages.append("ParseError CPT: No valid blocks after splitting."); return [], {}, parsing_messages
        try:
            i = 0
            while i < len(blocks):
                action = blocks[i].strip();
                if action not in ['create', 'update', 'delete']:
                    msg=f"ParseError CPT: Block {i}: Expected action, found '{action}'."; parsing_messages.append(msg); logging.error(msg + f" Context: '{self.delimiter.join(blocks[max(0,i-2):i+3])}'"); break
                i += 1
                if i + 3 >= len(blocks):
                    msg=f"ParseError CPT: Block {i-1} ('{action}'): Incomplete block."; parsing_messages.append(msg); logging.error(msg + f" Rem blocks: {len(blocks)-i}"); break
                entity_types_str=blocks[i].strip(); i+=1; entity_keys_str=blocks[i].strip(); i+=1; flags_str=blocks[i].strip(); i+=1; values_str=blocks[i].strip(); i+=1
                entity_types=list(filter(None, entity_types_str.split(' '))); entity_keys=list(filter(None, entity_keys_str.split(' '))); flags=list(filter(None, flags_str.split(' '))); values=list(filter(None, values_str.split(' ')))
                list_len = len(entity_types)
                if not (list_len == len(entity_keys) == len(flags) == len(values)):
                    msg=f"ParseError CPT: Action '{action}': Mismatched count T({len(entity_types)}) K({len(entity_keys)}) F({len(flags)}) V({len(values)})."; parsing_messages.append(msg); logging.error(msg + f" Data: ET='{entity_types_str}' EK='{entity_keys_str}' F='{flags_str}' V='{values_str}'"); continue
                if list_len == 0: msg=f"ParseInfo CPT: Action '{action}' block present but empty."; parsing_messages.append(msg); logging.info(msg); continue
                for idx in range(list_len):
                    etype=entity_types[idx]; ekey=entity_keys[idx]; flag_str=flags[idx]; value_expr=values[idx]
                    if flag_str not in ['true', 'false']: msg=f"ParseWarn CPT: Action '{action}' Item {idx+1} ('{etype}/{ekey}'): Invalid flag '{flag_str}'. Assuming 'false'."; parsing_messages.append(msg); logging.warning(msg); needs_calculation=False
                    else: needs_calculation = flag_str == 'true'
                    actual_value = None if action == 'delete' and value_expr == 'none' else value_expr
                    if action == 'delete' and needs_calculation: msg=f"ParseWarn CPT: Action 'delete' Item {idx+1} ('{etype}/{ekey}'): 'true' flag ignored."; parsing_messages.append(msg); logging.warning(msg); needs_calculation=False
                    ref_id = None
                    if needs_calculation:
                        expression_counter += 1; ref_id = f"{self.calc_ref_prefix}{expression_counter}"; expression_map[ref_id] = actual_value; logging.info(f"Assigned ref {ref_id} -> '{actual_value}' for {action} {etype}/{ekey}")
                    crud_ops_to_prepare.append({"action":action, "entity_type":etype, "entity_key":ekey, "value":actual_value, "_needs_calculation":needs_calculation, "_ref_id":ref_id})
        except Exception as e: msg=f"ParseException CPT: Error during parsing: {e}"; parsing_messages.append(msg); logging.exception(msg + f" Line: '{structured_line}'")
        final_crud_ops = []
        for crud_data in crud_ops_to_prepare:
             needs_calc=crud_data.pop("_needs_calculation",False); ref_id=crud_data.pop("_ref_id",None)
             if needs_calc:
                 expression=crud_data["value"];
                 if expression is None: err_msg=f"[StoreErr:CalcFail {crud_data.get('action','?')} {crud_data.get('entity_key','?')} (ref {ref_id or 'N/A'}) expr was None]"; parsing_messages.append(err_msg); logging.error(err_msg); continue
                 calculated_value, calc_error = self._evaluate_expression(expression, f"CRUD val (ref {ref_id or 'N/A'})")
                 if calc_error: err_msg=f"[StoreErr:CalcFail {crud_data.get('action','?')} {crud_data.get('entity_key','?')} (ref {ref_id or 'N/A'}): {calc_error}. Expr: '{expression}']"; parsing_messages.append(err_msg); logging.error(err_msg); continue
                 else: crud_data['value'] = str(calculated_value)
             final_crud_ops.append(crud_data)
        return final_crud_ops, expression_map, parsing_messages

    def process_request(self, user_prompt: str) -> str:
        """ Processes user request using compact plain text V2 format. """
        if not self.model: return "[System Error: AI Model not initialized]"

        
        try:
            store_context = self._get_relevant_context(user_prompt)
            prompt_content = f"{store_context}\n\nUSER:\n{user_prompt}".strip()
        except Exception as e:
             logging.exception("Context retrieval failed.")
             return f"[System Error: Context retrieval failed: {e}]"

        crud_ops_to_run: List[Dict[str, Any]] = []
        expression_map: Dict[str, str] = {}
        natural_response_processed: str = ""
        system_notes: List[str] = []

        try:
            
            logging.info(f"--- Sending prompt to AI (Compact V2 Format) ---")
            logging.debug(f"Prompt Content (start):\n{prompt_content[:600]}{'...' if len(prompt_content)>600 else ''}")
            response = self.model.generate_content(prompt_content)

            
            if not response.candidates:
                 feedback=response.prompt_feedback; reason=getattr(feedback,'block_reason','Unk'); ratings=getattr(feedback,'safety_ratings',[])
                 ratings_str=', '.join([f"{r.category.name}:{r.probability.name}" for r in ratings])
                 error_msg=f"AI resp blocked Reason:{reason} Ratings:[{ratings_str or 'N/A'}]"; logging.error(error_msg+f" FB:{feedback}"); return f"[System Error: {error_msg}]"

            ai_response_raw = response.text
            logging.info(f"--- AI Raw Response (Compact V2) ---\n{ai_response_raw}\n" + "-"*23)

            
            lines = ai_response_raw.strip().split('\n', 1) 
            structured_line = lines[0].strip() if lines else ""
            natural_response_raw = lines[1].strip() if len(lines) > 1 else "" 

            if structured_line:
                crud_ops_to_run, expression_map, parse_msgs = self._parse_compact_format(structured_line)
                system_notes.extend(parse_msgs)
                logging.info(f"Parsed {len(crud_ops_to_run)} CRUD ops, {len(expression_map)} expressions for ref.")
            else:
                system_notes.append("ParseInfo CPT: No structured data line found.")
                logging.info("No structured data (first line) found in AI response.")

            
            if crud_ops_to_run:
                logging.info(f"Executing {len(crud_ops_to_run)} prepared CRUD operations.")
                for i, crud_data in enumerate(crud_ops_to_run):
                    crud_result_msg = self._execute_crud(crud_data)
                    system_notes.append(f"CRUD {i+1}: {crud_result_msg}")
            else:
                logging.info("No valid CRUD operations prepared.")

            
            if natural_response_raw:
                logging.info("Performing inline calculations on natural response.")
                natural_response_processed = self._perform_inline_calculations(natural_response_raw, expression_map)
            else:
                 logging.info("No natural response (second line) provided.")
                 natural_response_processed = ""

        except types.StopCandidateException as e:
             logging.warning(f"Generation stopped unexpectedly: {e}")
             natural_response_processed = "[Warn: AI response truncated]"
             system_notes.append("Warn: AI response truncated")
        except Exception as e:
            logging.exception(f"Critical error during processing: {e}")
            natural_response_processed = f"[System Error: Processing failed: {e}]"
            system_notes.append(f"System Error: {e}")

        
        final_output = natural_response_processed.strip()
        if system_notes:
            notes_str = "\n".join(filter(None, [f"- {note}" for note in system_notes]))
            if notes_str: final_output += f"\n\n[-- System Notes --\n{notes_str}\n-- End Notes --]"
        return final_output



if __name__ == "__main__":
    print("--- AI Agent ---")
    print(f"Data Store: {JSON_FILE}")
    print(f"Default Model: {DEFAULT_MODEL_NAME}")
    print("-" * 30)

    agent = None
    try:
        
        agent = CompactAgent(
            api_key=API_KEY,
            model_name=DEFAULT_MODEL_NAME,
            json_filepath=JSON_FILE
        )
        print(f"\nAgent initialized successfully.")
        print("Ready. Enter request ('quit' to exit):")

    except (ValueError, ConnectionError, IOError, Exception) as init_err:
        logging.exception("Agent initialization failed.")
        print(f"\nFATAL ERROR: Init Failed: {init_err}"); exit(1)

    
    while True:
        try:
            prompt = input("> ")
            if prompt.lower().strip() == 'quit': break
            if not prompt.strip(): continue
            print("   ... Processing ...")
            result = agent.process_request(prompt)
            print("\n--- Agent Response ---"); print(result); print("-" * 30 + "\n")
        except KeyboardInterrupt: print("\nExiting..."); break
        except EOFError: print("\nExiting..."); break
        except Exception as loop_err:
            logging.exception("Unexpected loop error.")
            print(f"\n[LOOP ERROR]: {loop_err}. Check logs. Try again or 'quit'.")

    print("Agent shutdown.")