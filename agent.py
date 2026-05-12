# AgentAI
import re
import builtins
import traceback
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.figure import Figure
from time import sleep
from typing import Union
from colorama import Fore
from abc import abstractmethod

import streamlit as st


WHITELIST_DEFAULT = [
    "matplotlib",
    "mpl_toolkits",
    "seaborn",
    "numpy",
    "pandas",
    "sklearn",
]

# ── Streaming config ──────────────────────────────────────────────────────────
# Khi True: dùng llm.stream() để hiển thị token real-time trong chat bubble.
# Tắt nếu model không hỗ trợ streaming.
ENABLE_LLM_STREAMING = True


class AgentAI:
    """
    The class creates an LLM agent for execute python code in a safety environment.

    Attributes:

        data: list[pd.DataFrame]
            A list of pandas DataFrames containing the data to be processed.
        llm: object
            The instantiated large language model (LLM) used to process the data.
        max_attempts: int
            The maximum number of attempts allowed for execution.
        whitelist: list[str]
            A list of modules allowed to run in the environment.
        verbose: bool, optional
            If set to True, enables detailed. Default is False.
        break_run: bool
            Stop atual chat runtime.
        last_prompt: str
            The last prompt send to llm.

    Methods:

        chat(prompt: str)
            Process chat and return code execution response.
        chat_stop()
            Stop atual chat runtime.
        get_last_code()
            Return the last llm response code.
        get_last_prompt()
            Return the last prompt.
    """

    def __init__(
        self,
        data: list[pd.DataFrame],
        llm: object,
        max_attempts: int,
        whitelist: list[str],
        verbose: bool = False,
    ):
        self.data = data
        self.llm = llm
        self.max_attempts = max_attempts
        self.whitelist = whitelist
        self.break_run = False
        self.verbose = verbose
        self.last_prompt = None
        self.last_code = ""

    def __del__(self):
        pass

    # ── Streaming LLM call ────────────────────────────────────────────────────

    def _invoke_llm_with_stream(self, prompt: str) -> str:
        """
        Gọi LLM với streaming: hiển thị token real-time trong Streamlit chat bubble.
        Trả về full content string khi hoàn tất.

        Áp dụng kỹ thuật từ ollama.py: placeholder.markdown(text + "▌") mỗi token.
        Early-exit khi đã đủ closing ``` để không chờ phần text thừa sau code block.
        """
        full_text = ""
        _open_tag = "```python"
        _close_tag = "```"

        try:
            placeholder = st.empty()
            for chunk in self.llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                full_text += token
                placeholder.markdown(full_text + "▌")

                # Early-exit: đã có đủ 1 python block hoàn chỉnh
                open_pos = full_text.find(_open_tag)
                if open_pos != -1:
                    close_pos = full_text.find(_close_tag, open_pos + len(_open_tag))
                    if close_pos != -1:
                        # Block đã đóng — không cần stream thêm
                        break

            placeholder.markdown(full_text)
            return full_text

        except (AttributeError, NotImplementedError):
            # Fallback: model không hỗ trợ streaming
            print(f"{Fore.YELLOW}[AGENT] Streaming không khả dụng — fallback invoke{Fore.RESET}")
            resp = self.llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)

    def _invoke_llm_blocking(self, prompt: str) -> str:
        """Blocking invoke (dùng cho retry attempts sau attempt đầu)."""
        response = self.llm.invoke(prompt)
        return response.content if response and hasattr(response, "content") else ""

    # ── Main chat loop ────────────────────────────────────────────────────────

    def chat(self, prompt: str) -> Union[list, str, pd.DataFrame, Figure, dict]:
        """
        Invoke the chat language model with the provided prompt.
        Execute the returned code from the chat model.
        Return the results of the execution code in one of the defined types.

        Thay đổi so với bản gốc:
        - Attempt 1: dùng streaming (hiển thị token real-time → perceived latency thấp hơn)
        - Attempt 2+: dùng blocking invoke (không cần stream lại khi đang fix lỗi)
        - sleep(3) → exponential backoff bắt đầu từ 0.5s (tiết kiệm ~25s worst case)

        Args:
            prompt (str): the prompt to send to the LLM.

        Returns:
            Union[list, str, pd.DataFrame, Figure, dict]: code execution result.
        """

        attempts_var = 0
        prompt_ = prompt

        while attempts_var <= self.max_attempts:
            self.last_prompt = prompt_

            if self.break_run:
                self.break_run = False
                print(f"\n{Fore.LIGHTRED_EX}STOPPED INSTANCE!!!{Fore.RESET}\n")
                break

            if self.verbose:
                print(f"\n{Fore.LIGHTYELLOW_EX}FINAL PROMPT:{Fore.RESET}{prompt_}\n")

            attempts_var += 1
            is_first_attempt = attempts_var == 1
            print(
                f"\n{Fore.LIGHTBLUE_EX}[AGENT] Attempt {attempts_var}/{self.max_attempts} — "
                f"{'streaming' if (ENABLE_LLM_STREAMING and is_first_attempt) else 'blocking'} LLM...{Fore.RESET}"
            )

            try:
                # ── Attempt 1: streaming để user thấy output ngay ────────────
                # ── Attempt 2+: blocking (đang fix lỗi, không cần stream) ────
                if ENABLE_LLM_STREAMING and is_first_attempt:
                    content = self._invoke_llm_with_stream(prompt_)
                else:
                    content = self._invoke_llm_blocking(prompt_)

                print(
                    f"{Fore.WHITE}[AGENT] LLM response (content len={len(content)}){Fore.RESET}"
                )

                if content:
                    match = re.search(r"```python(.*?)```", content, re.DOTALL)

                    if match:
                        code = match.group(1)
                        lines = len(code.strip().splitlines())
                        print(f"{Fore.WHITE}[AGENT] Code extracted ({lines} lines){Fore.RESET}")
                    else:
                        print(f"{Fore.YELLOW}[AGENT] No code block found in response{Fore.RESET}")
                        code = 'raise Exception("No code returned, try again.")'

                    self.last_code = code

                    print(f"{Fore.WHITE}[AGENT] Executing in sandbox...{Fore.RESET}")
                    code_result = self.exec_code(code)
                    print(
                        f"{Fore.LIGHTGREEN_EX}[AGENT] ✓ Execution OK — result type: {type(code_result).__name__}{Fore.RESET}"
                    )
                    return code_result

                else:
                    print(f"{Fore.YELLOW}[AGENT] Empty response — retry (attempt {attempts_var}){Fore.RESET}")

            except Exception as e:
                # More randomness for error correction in later attempts
                if attempts_var > self.max_attempts / 2:
                    self.llm.temperature = 0.5

                error_message = str(e)
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_track = f"EXCEPTION_TRACK: {traceback.format_list(traceback.extract_tb(e.__traceback__)[-1:])[0].strip()}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                exception_msg = f"{exception_type}{exception_track}{exception_message}"

                print(
                    f"{Fore.LIGHTRED_EX}[AGENT] ✗ Error attempt {attempts_var}: {type(e).__name__}: {str(e)[:120]}{Fore.RESET}"
                )
                if self.verbose:
                    print(f"{Fore.LIGHTRED_EX}\nExecution error:\n{error_message}{Fore.RESET}\n")

                # Format error code with line numbers
                lines = self.last_code.split("\n")
                formatted_lines = [f"|Line-{i+1:03}| {line}" for i, line in enumerate(lines)]
                code_withlines = "\n".join(formatted_lines)

                tag_last_code = f"\n<code_error>\n```python\n{code_withlines}```\n</code_error>\n"
                tag_error = f"\n<message_error>\n{exception_msg}\n</message_error>\n"

                if len(error_message.split()) > 1:
                    if error_message.split()[1] == "SAFETY:":
                        return error_message

                prompt_ = prompt + tag_last_code + tag_error

                if attempts_var == self.max_attempts:
                    self.llm.temperature = 0.0
                    print(f"{Fore.LIGHTRED_EX}[AGENT] Max attempts ({self.max_attempts}) reached{Fore.RESET}")
                    return f"EXCEPTION ERROR: {error_message}"

                # ── Exponential backoff thay vì sleep(3) cứng ────────────────
                # attempt 1→0.5s, 2→0.75s, 3→1.1s, ... cap tại 5s
                # So với bản gốc (sleep(3) mỗi lần): tiết kiệm ~1-2s/retry
                import random
                backoff = min(0.5 * (1.5 ** (attempts_var - 1)) + random.uniform(0, 0.2), 5.0)
                print(f"{Fore.YELLOW}[AGENT] Retry in {backoff:.1f}s...{Fore.RESET}")
                sleep(backoff)

    # ── Approval-workflow methods ─────────────────────────────────────────────

    def chat_generate(self, prompt: str) -> str:
        """
        Phase 1 of the approval workflow: call the LLM and extract the code block.
        Does NOT execute anything.  Returns the extracted code string (or a
        placeholder that will raise on exec so the caller knows generation failed).

        Args:
            prompt (str): the full prompt to send to the LLM.

        Returns:
            str: the extracted Python code (not yet executed).
        """
        self.last_prompt = prompt

        if self.verbose:
            print(f"\n{Fore.LIGHTYELLOW_EX}[AGENT] GENERATE — FINAL PROMPT:{Fore.RESET}{prompt}\n")

        print(f"\n{Fore.LIGHTBLUE_EX}[AGENT] chat_generate — streaming LLM...{Fore.RESET}")

        if ENABLE_LLM_STREAMING:
            content = self._invoke_llm_with_stream(prompt)
        else:
            content = self._invoke_llm_blocking(prompt)

        print(f"{Fore.WHITE}[AGENT] LLM response (content len={len(content)}){Fore.RESET}")

        if content:
            match = re.search(r"```python(.*?)```", content, re.DOTALL)
            if match:
                code = match.group(1)
                lines = len(code.strip().splitlines())
                print(f"{Fore.WHITE}[AGENT] Code extracted ({lines} lines){Fore.RESET}")
            else:
                print(f"{Fore.YELLOW}[AGENT] No code block found in response{Fore.RESET}")
                code = '# ⚠ Không tìm thấy code block trong phản hồi của LLM\nresult = None'
        else:
            print(f"{Fore.YELLOW}[AGENT] Empty LLM response{Fore.RESET}")
            code = '# ⚠ LLM trả về phản hồi rỗng\nresult = None'

        self.last_code = code
        return code

    def chat_execute(self, code: str) -> Union[list, str, pd.DataFrame, Figure, dict]:
        """
        Phase 2 of the approval workflow: execute *exactly* the code supplied by
        the human (after optional edits).  Retries with the LLM on exception,
        just like the original chat() loop.

        Args:
            code (str): the (possibly human-edited) Python code to execute.

        Returns:
            Union[list, str, pd.DataFrame, Figure, dict]: code execution result.
        """
        prompt_ = self.last_prompt or ""
        attempts_var = 0

        # First attempt uses the human-approved code directly
        approved_code = code

        while attempts_var <= self.max_attempts:
            if self.break_run:
                self.break_run = False
                print(f"\n{Fore.LIGHTRED_EX}STOPPED INSTANCE!!!{Fore.RESET}\n")
                break

            attempts_var += 1
            current_code = approved_code if attempts_var == 1 else self.last_code

            print(
                f"\n{Fore.LIGHTBLUE_EX}[AGENT] chat_execute attempt {attempts_var}/{self.max_attempts}{Fore.RESET}"
            )

            try:
                self.last_code = current_code
                print(f"{Fore.WHITE}[AGENT] Executing in sandbox...{Fore.RESET}")
                code_result = self.exec_code(current_code)
                print(
                    f"{Fore.LIGHTGREEN_EX}[AGENT] ✓ Execution OK — result type: {type(code_result).__name__}{Fore.RESET}"
                )
                return code_result

            except Exception as e:
                if attempts_var > self.max_attempts / 2:
                    self.llm.temperature = 0.5

                error_message = str(e)
                exception_type = f"EXCEPTION_TYPE: {type(e).__name__}\n"
                exception_track = f"EXCEPTION_TRACK: {traceback.format_list(traceback.extract_tb(e.__traceback__)[-1:])[0].strip()}\n"
                exception_message = f"EXCEPTION_MESSAGE: {str(e)}"
                exception_msg = f"{exception_type}{exception_track}{exception_message}"

                print(
                    f"{Fore.LIGHTRED_EX}[AGENT] ✗ Execute error attempt {attempts_var}: "
                    f"{type(e).__name__}: {str(e)[:120]}{Fore.RESET}"
                )

                if len(error_message.split()) > 1 and error_message.split()[1] == "SAFETY:":
                    return error_message

                if attempts_var == self.max_attempts:
                    self.llm.temperature = 0.0
                    print(f"{Fore.LIGHTRED_EX}[AGENT] Max attempts reached{Fore.RESET}")
                    return f"EXCEPTION ERROR: {error_message}"

                # Ask the LLM to fix the broken code
                lines = current_code.split("\n")
                formatted_lines = [f"|Line-{i+1:03}| {line}" for i, line in enumerate(lines)]
                code_withlines = "\n".join(formatted_lines)
                tag_last_code = f"\n<code_error>\n```python\n{code_withlines}```\n</code_error>\n"
                tag_error = f"\n<message_error>\n{exception_msg}\n</message_error>\n"

                fix_prompt = prompt_ + tag_last_code + tag_error
                fix_content = self._invoke_llm_blocking(fix_prompt)
                match = re.search(r"```python(.*?)```", fix_content, re.DOTALL)
                if match:
                    self.last_code = match.group(1)
                else:
                    self.last_code = current_code  # no new code — will fail again → exhaust attempts

                import random
                backoff = min(0.5 * (1.5 ** (attempts_var - 1)) + random.uniform(0, 0.2), 5.0)
                print(f"{Fore.YELLOW}[AGENT] Retry in {backoff:.1f}s...{Fore.RESET}")
                sleep(backoff)

    def chat_stop(self):
        """Stop atual chat runtime by set the attribute break_run to True."""
        self.break_run = True

    def get_last_code(self) -> str:
        """Return the last executed code."""
        return self.last_code

    def get_last_prompt(self) -> str:
        """Return the last prompt."""
        return self.last_prompt

    @abstractmethod
    def restricted_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """
        Imports a module with restrictions based on a whitelist.
        """
        allowed_modules = (
            WHITELIST_DEFAULT
            if not self.whitelist
            else WHITELIST_DEFAULT + self.whitelist
        )
        if not any(name == mod or name.startswith(f"{mod}.") for mod in allowed_modules):
            raise ImportError(
                f"EXCEPTION SAFETY: importing the module '{name.split('.')[0]}' is restricted, is not in whitelist."
            )
        return builtins.__import__(name, globals, locals, fromlist, level)

    @abstractmethod
    def create_isolated_env(self) -> dict:
        """
        Creates an isolated execution environment with restricted built-ins and pre-defined variables.
        """
        allowed_builtins = {
            "abs": abs, "all": all, "any": any, "ascii": ascii, "bin": bin,
            "bool": bool, "bytearray": bytearray, "bytes": bytes, "callable": callable,
            "chr": chr, "classmethod": classmethod, "complex": complex, "delattr": delattr,
            "dict": dict, "dir": dir, "divmod": divmod, "enumerate": enumerate,
            "filter": filter, "float": float, "format": format, "frozenset": frozenset,
            "getattr": getattr, "hasattr": hasattr, "hash": hash, "help": help,
            "hex": hex, "id": id, "int": int, "isinstance": isinstance,
            "issubclass": issubclass, "iter": iter, "len": len, "list": list,
            "locals": locals, "map": map, "max": max, "memoryview": memoryview,
            "min": min, "next": next, "object": object, "oct": oct, "ord": ord,
            "pow": pow, "property": property, "range": range, "repr": repr,
            "reversed": reversed, "round": round, "set": set, "setattr": setattr,
            "slice": slice, "sorted": sorted, "staticmethod": staticmethod, "str": str,
            "sum": sum, "super": super, "tuple": tuple, "type": type, "vars": vars,
            "zip": zip, "print": print, "Exception": Exception,
            "__import__": self.restricted_import,
        }

        global_env = {"__builtins__": allowed_builtins}
        for i, var in enumerate(self.data):
            global_env[f"DF_{i+1}"] = var
        return global_env

    @abstractmethod
    def exec_code(self, code: str) -> str:
        """
        Executes provided code in a restricted environment.
        """
        plt.clf()
        plt.close("all")
        _saved_show = plt.show
        plt.show = self._intercept_plt_show

        env = self.create_isolated_env()
        context = {}
        try:
            exec(code, env, context)
        finally:
            plt.show = _saved_show

        code_result = context["result"]
        return code_result

    @staticmethod
    def _intercept_plt_show(*_args, **_kwargs):
        raise RuntimeError(
            "plt.show() is not allowed; assign the matplotlib Figure to result per the prompt."
        )