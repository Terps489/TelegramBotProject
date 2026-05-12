# OCR Telegram Bot

Telegram-бот для распознавания текста на изображениях. Русский и английский языки.

## Возможности

- Принимает изображения как **фото** и как **документ**.
- Предобработка через OpenCV: grayscale → шумоподавление → бинаризация (Otsu) → масштабирование.
- Распознавание через EasyOCR.
- Возвращает текст, время обработки и среднюю уверенность модели.
- Команды: `/start`, `/help`, `/about`.
- Обработка ошибок: не-изображение, превышение размера, пустой результат, повреждённый файл, отсутствие токена.
- Конфигурация через `.env`, токен не хранится в коде.

## Стек

- Python 3.10+
- aiogram 3.x (long polling)
- EasyOCR
- OpenCV, NumPy
- python-dotenv
- pytest

## Структура

```
ocr-telegram-bot/
├── bot/                 # Telegram-бот (aiogram)
├── ocr/                 # распознавание и предобработка
├── config/              # настройки (.env)
├── data/
│   ├── samples/         # тестовые изображения
│   ├── uploads/         # временные загрузки (gitignore)
│   └── results/         # результаты (gitignore)
├── tests/               # unit-тесты
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
└── run.py
```

## Установка

```bash
git clone https://github.com/<user>/ocr-telegram-bot.git
cd ocr-telegram-bot

python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # на Windows: copy .env.example .env
```

В файле `.env` укажите токен бота, полученный у [@BotFather](https://t.me/BotFather):

```
BOT_TOKEN=123456789:AA...
```

## Запуск

```bash
python run.py
```

После старта в Telegram-клиенте найдите своего бота и отправьте `/start`.

## Тесты

```bash
pytest -v
```

Тесты не требуют интернета и моделей EasyOCR — тяжёлые зависимости подменяются моками.

## Конфигурация

| Переменная        | По умолчанию | Описание |
|-------------------|--------------|----------|
| `BOT_TOKEN`       | —            | Токен Telegram-бота (обязательно) |
| `OCR_LANGUAGES`   | `ru,en`      | Языки EasyOCR через запятую |
| `USE_GPU`         | `false`      | Использовать CUDA-GPU |
| `MAX_FILE_SIZE_MB`| `20`         | Лимит размера входящего файла |
| `KEEP_UPLOADS`    | `false`      | Сохранять загруженные изображения |

## Лицензия

Учебный проект.
