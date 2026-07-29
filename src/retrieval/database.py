from pathlib import Path
import duckdb
from pydantic import BaseModel


class Target(BaseModel):
    """
    Represents a unique entity dimension within the vector index.

    Attributes:
        company (str): The full corporate name.
        ticker (str): The stock ticker symbol.
        fiscal_year (int): The financial reporting year.
    """
    company: str
    ticker: str
    fiscal_year: int


class Database:
    """
    Manages the DuckDB metadata catalog for the vector store.

    Facilitates structured filtering and ambiguity resolution prior to 
    executing costly embedding-based similarity searches.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path

    def _connect(self):
        """
        Establishes a read-only connection to the DuckDB instance.

        Returns:
            duckdb.DuckDBPyConnection: The active database connection.
        """
        return duckdb.connect(str(self.db_path), read_only=True)

    def get_all(self) -> list[Target]:
        """
        Retrieves the complete catalog of indexed entities.

        Returns:
            list[Target]: A comprehensive list of all indexed filing targets.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT company, ticker, fiscal_year FROM chunks ORDER BY company, fiscal_year"
            ).fetchall()
            return [Target(company=r[0], ticker=r[1], fiscal_year=r[2]) for r in rows]

    def resolve(self, company: str = None, ticker: str = None, fiscal_year: int = None) -> list[Target]:
        """
        Resolves a partial metadata filter against the indexed catalog.

        Employed by the clarification gate to detect ambiguous or non-existent 
        entities before executing vector searches.

        Args:
            company (str, optional): The target company name or ticker.
            ticker (str, optional): The target ticker symbol.
            fiscal_year (int, optional): The target reporting year.

        Returns:
            list[Target]: A list of entities matching the partial filter.
        """
        with self._connect() as conn:
            sql = "SELECT DISTINCT company, ticker, fiscal_year FROM chunks WHERE 1=1"
            params = []
            if company is not None:
                sql += " AND (company = ? OR ticker = ?)"
                params.extend([company, company])
            if ticker is not None:
                sql += " AND ticker = ?"
                params.append(ticker)
            if fiscal_year is not None:
                sql += " AND fiscal_year = ?"
                params.append(fiscal_year)

            rows = conn.execute(sql, params).fetchall()
            return [Target(company=r[0], ticker=r[1], fiscal_year=r[2]) for r in rows]
