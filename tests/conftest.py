"""
Pytest configuration and fixtures.

Provides test configuration, fixtures, and warning filters for handling
third-party deprecation warnings that we cannot control.

ChromaDB has a known issue where it accesses model_fields on Pydantic model instances,
which is deprecated in Pydantic v2.11+. Since we cannot control ChromaDB's code,
we suppress this specific warning at the pytest level.
"""

import warnings


def pytest_configure(config):
    """Configure pytest and register warning filters."""
    # Register ChromaDB warning filter in warnings module
    warnings.filterwarnings(
        "ignore",
        message="Accessing the 'model_fields' attribute on the instance is deprecated.*",
        category=DeprecationWarning,
    )


def pytest_collection_modifyitems(config, items):
    """
    Hook that runs after test collection to suppress ChromaDB warnings.
    
    This runs even when -W error::DeprecationWarning is used, because it modifies
    how warnings are handled for each test item.
    """
    # This ensures the warning filter is in place before tests run
    warnings.filterwarnings(
        "ignore",
        message="Accessing the 'model_fields' attribute on the instance is deprecated.*",
        category=DeprecationWarning,
    )





