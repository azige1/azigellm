"""规格合成的提示词模板。"""

SYSTEM_MESSAGE = """You are a python programming expert who writes concise behavioral specifications for python functions.
You will be given a python function signature and its existing docstring, together with unit tests that have passed against a reference implementation.
Your goal is to rewrite the specification as a Google-style docstring of 3 to 8 sentences.
Do NOT translate the test code line by line, and do NOT describe implementation details.
"""

SYSTEM_MESSAGE_TESTS = """You are a python programming expert who writes concise behavioral specifications for python functions.
You will be given a python function signature and its existing docstring, together with unit tests that have passed against a reference implementation and optionally some observed input/output examples.
Your goal is to rewrite the specification as a Google-style docstring of 3 to 8 sentences.
The first sentence describes the function's purpose; the following sentences describe parameter conventions, return value and behavior; the last one or two sentences give concrete input/output examples and exception conventions taken from the passing tests, quoting exception types and messages verbatim.
Do NOT translate the test code line by line, and do NOT describe implementation details.
Only state content you are sure about, and keep the specification precise without being verbose.
You will return the function definition with the rewritten docstring enclosed in markdown code delimiters.
"""

TASK_MESSAGE = """Code Snippet:

{code_snipppet}

Unit TestBatch:

```python
{test_code}```

Argument Types:
{argument_types}

Output Types: {output_type}

{example_substring}
Rewrite the specification docstring for the function {function_name} as 3 to 8 sentences.
Return only the updated function with docstring enclosed in markdown and ignore the remaining code.
The first sentence states what the function does; subsequent sentences document parameter and return conventions; the last one or two sentences give examples and exception conventions observed in the passing tests (quote exception types and messages verbatim).
"""

SOLVABILITY_SYSTEM_MESSAGE = """You are a python programming expert who implements functions from their signatures and specifications.
You will be given a function signature and a specification docstring.
Implement the function exactly as specified. Return only the function definition (and any helper imports it needs) in markdown code delimiters.
"""

SOLVABILITY_TASK_MESSAGE = """Signature and specification:

```python
{code_snipppet}
```

Implement the function `{function_name}` according to the specification above.
Return only the implementation in markdown code delimiters, for example:
```python
# YOUR CODE HERE
```
"""
