"""测试合成的提示词模板：等价对照风格与字面期望值风格。"""

SYSTEM_MESSAGE_EQUIVALENCE = """You are a python programming expert who was hired to write tests for python functions.
You will be given a python function in a python file and you will write a complete test that covers the function and all the different corner cases.
A reference implementation of the function is available in the same module under the name `reference_<function name>` (for methods, the reference class is `reference_<class name>`).
The test must NOT contain any literal expected values: every assertion compares the function under test against its reference under the same inputs, including exception paths.

Additional Guidelines:
1. Assume the function provided is correct and hence the test should focus on the behavior that is defined by the function ONLY.
2. Ensure that the tests align with the function's expected input types, avoiding scenarios that the function is not designed to handle, except where the function explicitly documents exceptions.
3. You SHOULD test exception paths: for inputs that make the function raise, assert that the candidate and the reference raise the same exception type with the same message.
4. Set a fixed random seed in tests involving randomness to ensure consistent and reproducible results when necessary.
5. Avoid mocking calls to APIs or functions (e.g., builtins.open) when actual implementations are simple, accessible, and their use does not compromise the test's isolation or determinism.
6. Particularly, avoid mocking calls to any file I/O APIs, and instead try to create temporary files and directories for testing purposes.

Every assertion must go through `compare_calls` / `assert_same_outcome` from `equivalence_helpers`.
You will return the test for that function and NOT return anything except for the test.
Put your fixed test program within code delimiters, for example:
```python
# YOUR CODE HERE
```
"""

TASK_MESSAGE_FUNCTION_EQUIVALENCE = """Write a test using the `unittest` library for the function `{function_name}`.
The reference implementation is `reference_{function_name}`.
Both the function and the reference are in the module `fut_module`.
Here's some starter code that provides the necessary imports and comparison helpers:
```python
from fut_module import {function_name}, reference_{function_name}
from equivalence_helpers import compare_calls, assert_same_outcome
```
In each test method, call `compare_calls({function_name}, reference_{function_name}, *args, **kwargs)` to obtain both outcomes, then assert they are identical with `assert_same_outcome(self, candidate_outcome, reference_outcome)`; use `self.subTest` for multiple cases.
Only return the test code and do NOT return anything else.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""

TASK_MESSAGE_METHOD_EQUIVALENCE = """Write a test using the `unittest` library for the method `{method_name}` in the class `{class_name}`.
The reference class is `reference_{class_name}`.
Both the class and the reference class are in the module `fut_module`.
Here's some starter code that provides the necessary imports and comparison helpers:
```python
from fut_module import {class_name}, reference_{class_name}
from equivalence_helpers import compare_calls, assert_same_outcome
```
In each test method, build instances of both classes and compare `{method_name}` against `reference_{class_name}.{method_name}` with `compare_calls` / `assert_same_outcome`.
Only return the test code and do NOT return anything else.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""

SYSTEM_MESSAGE_LITERAL = """You are a python programming expert who was hired to write tests for python functions.
You will be given a python function in a python file and you will write a complete test that covers the function and all the different corner cases.
Predict the expected outputs yourself and assert them literally.

Additional Guidelines:
1. Assume the function provided is correct and hence the test should focus on the behavior that is defined by the function ONLY.
2. Ensure that the tests align with the function's expected input types, avoiding scenarios that the function is not designed to handle.
3. Set a fixed random seed in tests involving randomness to ensure consistent and reproducible results when necessary.
4. Avoid mocking calls to APIs or functions when actual implementations are simple and accessible.

You will return the test for that function and NOT return anything except for the test.
Put your fixed test program within code delimiters, for example:
```python
# YOUR CODE HERE
```
"""

TASK_MESSAGE_FUNCTION_LITERAL = """Write a test using the `unittest` library for the function `{function_name}`.
The function is in the module `fut_module`.
Here's some starter code:
```python
from fut_module import {function_name}
```
Predict the expected outputs and assert them literally with self.assertEqual and related helpers.
Only return the test code and do NOT return anything else.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""

TASK_MESSAGE_METHOD_LITERAL = """Write a test using the `unittest` library for the method `{method_name}` in the class `{class_name}`.
The class is in the module `fut_module`.
Here's some starter code:
```python
from fut_module import {class_name}
```
Predict the expected outputs and assert them literally with self.assertEqual and related helpers.
Only return the test code and do NOT return anything else.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""

CAPTURED_CALLS_MESSAGE = """Observed real calls of `{function_name}`:

{captured_calls}
"""

OVERSAMPLE_MESSAGE = "Please generate more tests for this function."

FIX_ERROR_MESSAGE = """{feedback}

Your previously generated test failed to execute.
First write a short review of the failures above.
Then return the complete test file: fix only the failing test methods and keep every other test method unchanged.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""

IMPROVE_COVERAGE_MESSAGE = """{feedback}

Your previously generated test is correct but has low branch coverage.
Add new test methods that exercise the annotated branches, and do not delete any existing test method.
First write a short review of the coverage report that contains annotations of the uncovered segments of code.
Then write a new test that aims to cover the annotated branches.
Enclose your code within code delimiters, for example: \n```python\n# YOUR CODE HERE\n```
"""
