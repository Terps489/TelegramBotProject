import asyncio
import logging
import sys

from bot.main import run_bot
from config.settings import ConfigError


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )
    logging.getLogger("easyocr").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def main() -> None:
    configure_logging()
    log = logging.getLogger("run")
    try:
        asyncio.run(run_bot())
    except ConfigError as exc:
        log.error("Ошибка конфигурации: %s", exc)
        sys.exit(1)
    except (KeyboardInterrupt, SystemExit):
        log.info("Бот остановлен пользователем.")


if __name__ == "__main__":
    main()
