# 🔌 MCP Серверы (GitHub, Exa, DeepSeek, Obsidian, Colab)

> **Теги:** #mcp #deepseek #obsidian #github #colab #интеграция #конфигурация

---

## 🎯 Назначение установленных MCP серверов

Для глубокой автоматизации, отладки кода, поиска в технической литературе и синхронизации базы знаний настроены серверы **Model Context Protocol (MCP)**:

### 1. 🧠 DeepSeek MCP Server (`deepseek-mcp`)
Подключен к локальному прокси ForgetMeAI (`http://127.0.0.1:9655`), использующему бесплатный авторизационный токен.
- **Путь к серверу:** `C:\Users\Yaroslav\.gemini\mcp\deepseek-mcp\server.py`
- **Доступные инструменты:**
  1. `deepseek_ask` — быстрые ответы и генерация кода через модель `deepseek-chat` (V3 / V4.1-Flash).
  2. `deepseek_reason` — глубокое рассуждение (Chain-of-Thought / DeepThink) через модель `deepseek-r1`.
  3. `deepseek_debug_code` — углубленный аудит багов, анализ стек-трейсов и поиск первопричин ошибок.
  4. `deepseek_web_search` — веб-поиск информации в реальном времени с цитированием через `deepseek-chat-search`.

### 2. 📓 Obsidian MCP Server (`mcp-obsidian`)
Позволяет агенту напрямую взаимодействовать с базой знаний Obsidian:
- **Хранилище проекта:** `C:\Users\Yaroslav\OneDrive\Desktop\diches\проект\проект`
- **Инструменты:**
  - `search_notes` — поиск заметок по названию и регулярным выражениям.
  - `read_notes` — чтение содержимого нескольких заметок одновременно.

### 3. ☁️ Google Colab MCP Server (`colab-mcp`)
Управление браузерной сессией и средами выполнения Google Colab для удаленного обучения моделей на мощных GPU NVIDIA (Tesla T4, A100).

### 4. 🐙 GitHub MCP Server (`@modelcontextprotocol/server-github`)
Управление репозиторием `yaroslikch-glitch/pricing_of_a_unique_cast_product`: коммиты, ветки, релизы, управление файлами.

---

## ⚙️ Файл конфигурации (`mcp_config.json`)

Конфигурационный файл расположен по адресу:
`C:\Users\Yaroslav\.gemini\config\mcp_config.json`

```json
{
  "mcpServers": {
    "deepseek": {
      "command": "python",
      "args": ["C:\\Users\\Yaroslav\\.gemini\\mcp\\deepseek-mcp\\server.py"],
      "disabled": false
    },
    "obsidian": {
      "command": "npx",
      "args": [
        "-y",
        "mcp-obsidian",
        "C:\\Users\\Yaroslav\\OneDrive\\Desktop\\diches\\проект\\проект",
        "C:\\Users\\Yaroslav\\OneDrive\\Desktop\\diches\\сайт для сервака\\сервак"
      ],
      "disabled": false
    },
    "colab-mcp": {
      "command": "uv",
      "args": ["run", "colab-mcp"],
      "cwd": "C:\\Users\\Yaroslav\\.gemini\\mcp\\colab-mcp",
      "timeout": 30000,
      "disabled": false
    },
    "puppeteer": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-puppeteer"],
      "disabled": false
    }
  }
}
```

---

## 🔗 Связанные заметки
- [[00 - Главная (MOC)|🏠 Главная страница]]
- [[03 - Детекция ЕСКД (YOLOv8)|📐 Детекция ЕСКД (YOLOv8)]]
- [[07 - Инструкции по развертыванию и запуску|🚀 Инструкции по развертыванию]]
- [[09 - План работ и Задачи (Roadmap)|✅ План работ и Задачи]]
