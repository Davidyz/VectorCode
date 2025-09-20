# Database Connectors

A database connector is a compatibility layer that converts data structures that a 
database natively works with to the ones that VectorCode works with. The connector 
classes provide abstractions for VectorCode operations (`vectorise`, `query`, etc.), 
which enables the use of different database backends.

<!-- mtoc-start -->

* [Adding a New Database Connector](#adding-a-new-database-connector)
* [Key Implementation Details](#key-implementation-details)
  * [The `Config` Object](#the-config-object)
  * [Implementing Abstract Methods](#implementing-abstract-methods)
  * [Error Handling](#error-handling)
* [Testing](#testing)

<!-- mtoc-end -->

# Adding a New Database Connector

To add support for a new database backend, you will need to:

1.  **Implement a connector class**: Create a new file in this directory and implement a child class of `vectorcode.database.base.DatabaseConnectorBase`. You must implement all of its abstract methods.
2.  **Write tests**: Add tests for your new connector in the `tests/database/` directory. The tests should mock the database's API and verify that your connector correctly converts data between the database's native format and VectorCode's data structures.
3.  **Register your connector**: Add a new entry in the `get_database_connector` function in `src/vectorcode/database/__init__.py` to initialize your new connector.

For a concrete example, refer to the implementation of `DatabaseConnectorBase` and the `ChromaDB0Connector`.

# Key Implementation Details

## The `Config` Object

All settings for a connector are passed through a single `vectorcode.cli_utils.Config` object, which is available as `self._configs`. This includes:

-   **Database Settings**: The `db_type` string and `db_params` dictionary are used to configure the connection to the database backend. As a contributor, you should document the specific `db_params` your connector requires in the class's docstring.
-   **Operation Parameters**: Parameters for operations like `query` or `vectorise` are also present in this object.

The `self._configs` attribute is mutable and can be updated for subsequent operations, but the database connection settings (`db_type`, `db_params`) should not be changed after initialization.

## Implementing Abstract Methods

When implementing the abstract methods from `DatabaseConnectorBase`, you should:

-   Read the necessary parameters from the `self._configs` object.
-   Perform the corresponding operation against the database.
-   Return data in the format specified by the method's type hints (e.g., `QueryResult`, `CollectionInfo`).

**Please refer to the docstrings in `DatabaseConnectorBase` for the specific API contract of each method.** They contain detailed information about what each method is expected to do and what parameters it uses from the `Config` object.

## Error Handling

If the underlying database library raises a specific exception (e.g., for a collection not being found), you should consider catching it and re-raise it as one of VectorCode's custom database exceptions from `vectorcode.database.errors`. This ensures consistent error handling in the CLI and other clients.

For example:
```python
from vectorcode.database.errors import CollectionNotFoundError

try:
    some_action_here()
except SomeCustomException as e:
    raise CollectionNotFoundError("The collection was not found.") from e
```

# Testing

The unit tests for database backends should go under [`tests/database/`](../../../tests/database/). 
The tests should mock the request body and return values of the database. Integration 
tests that interact with an actual database are out of scope for now.

> The tests for the subcommands currently use mocked database connectors. They're not 
> supposed to interact with live databases.
