---
description: "Python coding conventions and guidelines"
applyTo: "**/*.py"
---

# Python Coding Conventions

## Python Instructions

- Write clear and concise comments for each function.
- Ensure functions have descriptive names and include type hints.
- Provide docstrings following PEP 257 conventions.
- Use the `typing` module for type annotations (e.g., `List[str]`, `Dict[str, int]`).
- Break down complex functions into smaller, more manageable functions.

## General Instructions

- Always prioritize readability and clarity.
- For algorithm-related code, include explanations of the approach used.
- Write code with good maintainability practices, including comments on why certain design decisions were made.
- Handle edge cases and write clear exception handling.
- For libraries or external dependencies, mention their usage and purpose in comments.
- Use consistent naming conventions and follow language-specific best practices.
- Write concise, efficient, and idiomatic code that is also easily understandable.

## Code Style and Formatting

- Follow the **PEP 8** style guide for Python.
- Maintain proper indentation (use 4 spaces for each level of indentation).
- Ensure lines do not exceed 79 characters.
- Place function and class docstrings immediately after the `def` or `class` keyword.
- Use blank lines to separate functions, classes, and code blocks where appropriate.

## Edge Cases and Testing

- Always include test cases for critical paths of the application.
- Account for common edge cases like empty inputs, invalid data types, and large datasets.
- Include comments for edge cases and the expected behavior in those cases.
- Write unit tests for functions and document them with docstrings explaining the test cases.

## Example of Proper Documentation

```python
def calculate_area(radius: float) -> float:
    """
    Calculate the area of a circle given the radius.

    Parameters:
    radius (float): The radius of the circle.

    Returns:
    float: The area of the circle, calculated as π * radius^2.
    """
    import math
    return math.pi * radius ** 2
```

# Copilot Instructions for Python Development

This document outlines the standards and best practices for writing professional, production-ready Python code. All code must adhere to the Google Python Style Guide: A comprehensive set of rules and recommendations for writing Python code that is readable, maintainable, and consistent..

## General Principles

- Write clean, readable, and maintainable code.
- Keep methods small and focused.
- Avoid duplication and unnecessary complexity.
- Use meaningful names for variables, functions, and classes.

## Naming Conventions

- Use `snake_case` for functions and variables.
- Use `CapWords` for class names.
- Constants should be `UPPER_CASE`.

## Imports

- Use absolute imports.
- Group imports in the following order: standard libraries, third-party libraries, local application imports.
- Separate each group with a blank line.

## Functions and Methods

- Keep functions small and focused.
- Each function should do one thing and do it well.
- Use type hints for function arguments and return values.

```python
def calculate_area(length: float, width: float) -> float:
    return length * width
```

## Documentation

- Use docstrings for all public modules, classes, and functions.
- Follow the PEP 257: Python Enhancement Proposal that describes conventions for Python docstrings, including formatting and usage guidelines. conventions.

```python
def add(a: int, b: int) -> int:
    """
    Add two integers and return the result.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The sum of a and b.
    """
    return a + b
```

## Error Handling

- Use exceptions for error handling.
- Avoid using bare `except` clauses.
- Always clean up resources using `with` statements or `try/finally` blocks.

```python
try:
    with open('file.txt', 'r') as file:
        data = file.read()
except FileNotFoundError as e:
    print(f"Error: {e}")
```

## Testing

- Write unit tests for all functions and classes.
- Use `unittest` or `pytest` frameworks.
- Ensure tests are isolated and repeatable.

## Code Style

- Limit lines to 80 characters.
- Use 4 spaces per indentation level.
- Avoid trailing whitespace.
- Use blank lines to separate functions and classes.

## Version Control

- Use meaningful commit messages.
- Commit small, logically grouped changes.

## Final Notes

- Always review code before committing.
- Use linters and formatters (e.g., `flake8`, `black`) to enforce style.
- Document any deviations from the style guide.
