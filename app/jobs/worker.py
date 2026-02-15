"""RQ worker entry point: python -m app.jobs.worker."""

from __future__ import annotations

from rq import Worker

from app.core.config import get_settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.jobs.queue import get_queue, get_redis_connection


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    init_db()

    queue = get_queue()
    worker = Worker([queue], connection=get_redis_connection())
    worker.work()


if __name__ == "__main__":
    main()
