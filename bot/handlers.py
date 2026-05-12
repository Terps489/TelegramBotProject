"""Хендлеры команд и входящих сообщений Telegram-бота."""
import logging
import uuid
from pathlib import Path

from aiogram import F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Document, Message, PhotoSize

from bot import messages
from config.settings import Settings
from ocr.recognizer import OCRRecognizer

log = logging.getLogger(__name__)
router = Router(name="ocr-router")

ALLOWED_MIME_PREFIXES = ("image/",)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(messages.START_MESSAGE)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(messages.HELP_MESSAGE)


@router.message(Command("about"))
async def cmd_about(message: Message) -> None:
    await message.answer(messages.ABOUT_MESSAGE)


@router.message(F.photo)
async def handle_photo(
    message: Message,
    recognizer: OCRRecognizer,
    settings: Settings,
) -> None:
    # Берём самый крупный размер — последний элемент списка.
    photo: PhotoSize = message.photo[-1]
    await _process_image(
        message=message,
        recognizer=recognizer,
        settings=settings,
        file_id=photo.file_id,
        file_size=photo.file_size or 0,
        suggested_ext=".jpg",
    )


@router.message(F.document)
async def handle_document(
    message: Message,
    recognizer: OCRRecognizer,
    settings: Settings,
) -> None:
    doc: Document = message.document
    mime = (doc.mime_type or "").lower()
    file_name = doc.file_name or ""
    ext = Path(file_name).suffix.lower()

    is_image_mime = any(mime.startswith(p) for p in ALLOWED_MIME_PREFIXES)
    is_image_ext = ext in ALLOWED_EXTENSIONS
    if not (is_image_mime or is_image_ext):
        await message.answer(messages.ERROR_NOT_AN_IMAGE)
        return

    await _process_image(
        message=message,
        recognizer=recognizer,
        settings=settings,
        file_id=doc.file_id,
        file_size=doc.file_size or 0,
        suggested_ext=ext or ".png",
    )


@router.message()
async def handle_other(message: Message) -> None:
    await message.answer(messages.ERROR_NOT_AN_IMAGE)


async def _process_image(
    *,
    message: Message,
    recognizer: OCRRecognizer,
    settings: Settings,
    file_id: str,
    file_size: int,
    suggested_ext: str,
) -> None:
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if file_size and file_size > max_bytes:
        await message.answer(
            messages.ERROR_FILE_TOO_LARGE.format(
                size_mb=file_size / (1024 * 1024),
                limit_mb=settings.max_file_size_mb,
            )
        )
        return

    status = await message.answer(messages.PROCESSING_MESSAGE)

    upload_path = settings.uploads_dir / f"{uuid.uuid4().hex}{suggested_ext}"
    try:
        await message.bot.download(file=file_id, destination=upload_path)
    except Exception:
        log.exception("Не удалось скачать файл file_id=%s", file_id)
        await status.edit_text(messages.ERROR_READ_IMAGE)
        return

    try:
        result = recognizer.recognize(upload_path)
    except FileNotFoundError:
        log.exception("Файл не найден после загрузки: %s", upload_path)
        await status.edit_text(messages.ERROR_READ_IMAGE)
        return
    except ValueError:
        log.exception("OpenCV не смог прочитать изображение: %s", upload_path)
        await status.edit_text(messages.ERROR_READ_IMAGE)
        return
    except Exception:
        log.exception("Непредвиденная ошибка при распознавании: %s", upload_path)
        await status.edit_text(messages.ERROR_UNEXPECTED)
        return
    finally:
        if not settings.keep_uploads:
            upload_path.unlink(missing_ok=True)

    if result.is_empty:
        await status.edit_text(messages.ERROR_OCR_EMPTY)
        return

    response = messages.format_result(
        text=result.text,
        elapsed=result.elapsed_seconds,
        avg_confidence=result.average_confidence,
    )
    await status.edit_text(response)
    log.info(
        "Распознан текст: user=%s, символов=%d, время=%.2fс, conf=%s",
        message.from_user.id if message.from_user else "?",
        len(result.text),
        result.elapsed_seconds,
        f"{result.average_confidence:.2f}" if result.average_confidence is not None else "n/a",
    )
