# db.py
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool

from config import DB_CONFIG

log = logging.getLogger(__name__)


class DBPool:
    """Wrapper cho ThreadedConnectionPool của psycopg2."""

    def __init__(self, minconn=2, maxconn=8):
        self._pool = pool.ThreadedConnectionPool(
            minconn, maxconn, **DB_CONFIG
        )
        log.info("DB pool created (min=%d, max=%d)", minconn, maxconn)

    @contextmanager
    def conn(self):
        """Context manager trả về connection, tự động commit/rollback và trả về pool."""
        conn = self._pool.getconn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def close(self):
        self._pool.closeall()
        log.info("DB pool closed")


# Tiện ích chuyển đổi vector từ string pgvector sang list[float]
def parse_vector_str(vec_str: str) -> list:
    """Chuyển '[1.2,3.4]' -> [1.2, 3.4]"""
    if vec_str is None:
        return []
    return [float(x) for x in vec_str.strip("[]").split(",")]