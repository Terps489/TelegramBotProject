"""Инициализация и запуск Telegram-бота на aiogram 3.x (long polling)."""
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers import router
from config.settings import get_settings
from ocr.recognizer import OCRRecognizer

log = logging.getLogger(__name__)


async def run_bot() -> None:
    settings = get_settings()
    log.info("Загружены настройки. Языки OCR: %s", settings.ocr_languages)

    # Инициализируем OCR заранее — первая загрузка моделей долгая,
    # лучше дождаться её до приёма сообщений, чем заставлять ждать пользователя.
    log.info("Инициализация EasyOCR (может занять до минуты при первом запуске)...")
    recognizer = OCRRecognizer(
        languages=settings.ocr_languages,
        gpu=settings.use_gpu,
    )
    recognizer.warmup()
    log.info("EasyOCR готов к работе.")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp["recognizer"] = recognizer
    dp["settings"] = settings
    dp.include_router(router)

    me = await bot.get_me()
    log.info("Бот запущен: @%s (id=%s). Режим: long polling.", me.username, me.id)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        log.info("Сессия бота закрыта.")
