# Database Connectors

A database connector is a compatibility layer that converts data structures that a 
database natively works with to the ones that VectorCode works with. The connector 
classes provides abstractions for VectorCode operations (`vectorise`, `query`, etc.), 
which enables the use of different database backends.

<!-- mtoc-start -->

* [Creating Database Connectors](#creating-database-connectors)
* [Implementation Details](#implementation-details)
  * [Connector Configuration](#connector-configuration)
  * [Database Settings](#database-settings)
    * [Documenting the Database Settings](#documenting-the-database-settings)
  * [CRUD Operations](#crud-operations)

<!-- mtoc-end -->

# Creating Database Connectors

To add support for a new database backend, you'd need to: 

1. Implement a child class of `vectorcode.database.base.DatabaseConnectorBase` and all 
   of its abstract methods, and put it under this directory.
2. Add a new entry in the [`get_database_connector`](./__init__.py) function that 
   initialises your new database connector when the `configs.db_type` points to the new 
   database.
3. Add tests for your new database connector. The new tests should verify that your 
   connector correctly converts between the native data structures from the database and
   the VectorCode data structures that the rest of the codebase (embedding function, 
   reranker, etc.)can work with.

# Implementation Details

> Apart from this document, you may refer to [the `DatabaseConnectorBase`](./base.py) 
> and [the `ChromaDB0Connector`](./chroma0.py) implementations as reference designs of 
> a new database connector.

In the following sections, I'll use the term _database_ to refer to the actual database 
backends (chromadb, pgvector, etc.) that holds the data and performs the CRUD operations, 
and the term _connector_ to refer to our compatibility layer (child classes of 
`vectorcode.database.base.DatabaseConnectorBase`).

## Connector Configuration

The connector has a private attribute (that is, the attribute name is prefixed by a `_`) 
`self._configs`. This is a `vectorcode.cli_utils.Config` object that holds various 
configuration options, including the database settings used to initialise the 
connections to the database and the parameters used for the CRUD operations with the 
database. This attribute is **mutable** and _should_ be updated before calling a CRUD 
method using the `self.update_config(new_config)` or the `self.replace_config(new_config)` 
methods. However, the database-related settings shouldn't be changed. A new connector
instance should be created for that purpose.

## Database Settings

The database settings are configured in the JSON configuration file, and will be parsed 
and stored in the `config.db_type` and `config.db_params` attributes of the 
`self._configs` object.

The `db_type` attribute is a string that indicates the type of the database backend 
(for example, `ChromaDB0` for Chromadb 0.6.3). 

The `db_params` attribute is a dictionary that holds some database-specific settings 
(for example, the database API endpoint URL and/or database directory).

### Documenting the Database Settings

Please document about the database-specific settings (`db_params`) in the doc-string 
of your database connector. This doc-string will be presented in the error message when 
the database fails to initialise, and should provide instructions to help the user 
debug their configuration.

## CRUD Operations

Historically, the parameters of VectorCode operations have been stored and propagated 
in a `vectorcode.cli_utils.Config` object. The database connectors continue to follow 
this pattern. That is, each of the abstract methods that represent an abstracted 
database operation (`query()`, `vectorise()`, `list()`, etc.) should read the necessary 
parameters (`project_root`, file paths, query keywords, etc.) from the `self._configs` 
attribute. Note that the `self._configs` attribute is mutable, so you should always read 
the parameters from it directly for each of the operations.

> Some methods support keyword arguments that allows temporarily overriding some 
> parameters. For example, the `list_collection_content` method supports overriding 
> `self._configs` by passing `_collection_id` and `collection_path`. The idea is that 
> these methods can usually be used by the implementation of other methods or subcommands 
> (for example, `list_collection_content` is used in `count` and `check_orphanes`),
> and being able to pass such parameters are convenient when writing those implementations.
