# ITMO University AI Assistant

Intelligent assistant providing comprehensive and accurate information about ITMO University using advanced AI technologies.

## Overview

This project implements an AI-powered assistant that provides reliable and well-thought-out answers about ITMO University. By combining Google Search API and Mistral AI, the system ensures accurate, ethical, and university-aligned responses.

### Key Features

- **Bilingual Support**: Handles queries in both Russian and English
- **Verified Information**: Uses official sources through targeted Google Search
- **Ethical Response System**: Validates queries for appropriateness and relevance
- **Source Transparency**: Provides links to information sources
- **Smart Information Synthesis**: Combines data from multiple sources for comprehensive answers

### Technology Stack

- **Mistral AI**: 
  - Query validation and ethics check
  - Bilingual support (Russian/English)
  - Information synthesis from multiple sources
  - Response generation aligned with university values
  
- **Google Search API**: 
  - Targeted search within official domains:
    - *.itmo.ru/*
    - *.minobrnauki.gov.ru/*
  - Ensures information accuracy and reliability
  - Real-time data from official sources

# FastAPI Baseline Service
Это пример базового приложения, которое реализует API для получения запроса и возврата ответа.
Приложение написано на FastAPI, разворачивается при помощи docker-compose.

## Сборка
Для запуска выполните команду:

```bash
docker-compose up -d
```
Она соберёт Docker-образ, а затем запустит контейнер.

После успешного запуска контейнера приложение будет доступно на http://localhost:8080.

## Проверка работы
Отправьте POST-запрос на эндпоинт /api/request. Например, используйте curl:

```bash
curl --location --request POST 'http://localhost:8080/api/request' \
--header 'Content-Type: application/json' \
--data-raw '{
  "query": "В каком городе находится главный кампус Университета ИТМО?\n1. Москва\n2. Санкт-Петербург\n3. Екатеринбург\n4. Нижний Новгород",
  "id": 1
}'
```
В ответ вы получите JSON вида:

```json
{
  "id": 1,
  "answer": 1,
  "reasoning": "Из информации на сайте",
  "sources": [
    "https://itmo.ru/ru/",
    "https://abit.itmo.ru/"
  ]
}
```

id будет соответствовать тому, что вы отправили в запросе,
answer (в базовой версии) всегда будет 5.
## Кастомизация
Чтобы изменить логику ответа, отредактируйте функцию handle_request в main.py.
Если нужно использовать дополнительные библиотеки, добавьте их в requirements.txt и пересоберите образ.


Чтобы остановить сервис, выполните:

```bash
docker-compose down
```
