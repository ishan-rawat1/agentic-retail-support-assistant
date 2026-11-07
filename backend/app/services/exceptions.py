"""Domain exceptions mapped to HTTP errors by the API layer."""


class NotFoundError(Exception):
    """The requested record does not exist *for this customer*.

    Deliberately raised both when a record is missing and when it belongs
    to another customer, so responses never reveal whether a guessed ID
    exists (prevents enumeration).
    """


class ValidationError(Exception):
    """Invalid input (bad ID format, unknown tool, bad arguments)."""


class AuthenticationError(Exception):
    """Missing or invalid credentials."""
