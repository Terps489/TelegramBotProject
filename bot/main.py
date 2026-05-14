import logging

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.handlers import router
from config.settings import Settings, get_settings
from ocr.models import EasyOCRModel, FinetunedEasyOCRModel, OCRModel
from ocr.preprocessing import PreprocessingOptions
from ocr.recognizer import OCRRecognizer

log = logging.getLogger(__name__)


def build_model(settings: Settings) -> OCRModel:
    if settings.ocr_model == "finetuned":
        log.info(
            "Используется кастомная модель '%s' (dir=%s)",
            settings.custom_recog_network,
            settings.custom_models_dir,
        )
        return FinetunedEasyOCRModel(
            recog_network=settings.custom_recog_network,
            model_storage_directory=settings.custom_models_dir,
            user_network_directory=settings.custom_user_network_dir,
            languages=settings.ocr_languages,
            gpu=settings.use_gpu,
        )
    return EasyOCRModel(languages=settings.ocr_languages, gpu=settings.use_gpu)


async def run_bot() -> None:
    settings = get_settings()
    log.info("Языки OCR: %s, модель: %s", settings.ocr_languages, settings.ocr_model)
    log.info(
        "Предобработка: deskew=%s, orientation=%s",
        settings.use_deskew,
        settings.use_orientation_correction,
    )

    model = build_model(settings)
    options = PreprocessingOptions(
        deskew=settings.use_deskew,
        orientation_correction=settings.use_orientation_correction,
    )

    log.info("Инициализация модели...")
    recognizer = OCRRecognizer(model=model, preprocess_options=options)
    recognizer.warmup()
    log.info("Модель готова.")

    bot = Bot(
        token=settings.bot_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp["recognizer"] = recognizer
    dp["settings"] = settings
    dp.include_router(router)

    me = await bot.get_me()
    log.info("Бот запущен: @%s (id=%s)", me.username, me.id)
    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        log.info("Сессия бота закрыта.")
