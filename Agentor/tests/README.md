# Agentor Test Suite

This directory contains a comprehensive test suite for the Agentor framework.

## Test Structure

The test suite is organized into the following directories:

- `tests/`: Root directory for all tests
  - `core/`: Tests for core components (logging, config, plugin system, etc.)
  - `agents/`: Tests for agent components (base agents, enhanced agents, tools, etc.)
  - `components/`: Tests for agent components (memory, environments, etc.)
  - `integration/`: Integration tests for multiple components working together
  - `benchmarks/`: Performance benchmarks
  - `conftest.py`: Common fixtures and utilities

## Running Tests

### Running All Tests

To run all tests:

```bash
pytest
```

### Running Specific Test Files

To run a specific test file:

```bash
pytest tests/core/test_caching.py
```

### Running Specific Test Functions

To run a specific test function:

```bash
pytest tests/core/test_caching.py::test_in_memory_cache_get_set
```

### Running Tests with Markers

To run tests with a specific marker:

```bash
pytest -m asyncio
```

### Running Benchmarks

Benchmarks are skipped by default. To run them:

```bash
pytest tests/benchmarks/test_caching_benchmarks.py -v --no-skip
```

## Test Coverage

To generate a test coverage report:

```bash
pytest --cov=agentor
```

For a more detailed HTML report:

```bash
pytest --cov=agentor --cov-report=html
```

## Writing Tests

### Test Naming Conventions

- Test files should be named `test_*.py`
- Test functions should be named `test_*`
- Test classes should be named `Test*`

### Using Fixtures

Fixtures are defined in `conftest.py` and can be used in any test:

```python
def test_something(in_memory_cache):
    # in_memory_cache is a fixture
    assert in_memory_cache is not None
```

### Testing Asynchronous Code

Use the `pytest.mark.asyncio` decorator for testing asynchronous functions:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result is not None
```

### Mocking Dependencies

Use the `unittest.mock` module for mocking dependencies:

```python
from unittest.mock import AsyncMock, patch

@patch('agentor.some_module.some_function')
def test_with_mock(mock_function):
    mock_function.return_value = 'mocked result'
    # Test code that uses some_function
```

## Test Categories

### Unit Tests

Unit tests focus on testing individual components in isolation. They should be fast and not depend on external services.

### Integration Tests

Integration tests verify that different components work together correctly. They may involve multiple components but should still avoid external dependencies when possible.

### Benchmarks

Benchmarks measure the performance of components under different conditions. They are useful for identifying performance regressions and optimizing code.

## Continuous Integration

The test suite is integrated with the CI/CD pipeline. All tests must pass before code can be merged into the main branch.
