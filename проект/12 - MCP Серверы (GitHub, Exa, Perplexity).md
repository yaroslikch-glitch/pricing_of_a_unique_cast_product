# 🔌 MCP Серверы (GitHub, Exa)

> **Теги:** #mcp #github #exa #интеграция #конфигурация

---

## 🎯 Назначение установленных MCP серверов

Для расширения возможностей поиска технической документации, методичек ЮУрГУ и управления репозиторием настроены серверы протокола **Model Context Protocol (MCP)**:

1. **GitHub MCP (`@modelcontextprotocol/server-github`)**:
   - Управление репозиторием проекта прямо из чата (просмотр веток, коммитов, создание Issue, Pull Request).
   - Поиск примеров кода и аналогичных проектов по распознаванию чертежей и анализу ЕСКД.
2. **Exa MCP (`exa-mcp-server`)**:
   - Нейросетевой семантический поиск по PDF-документам, университетским репозиториям и научным статьям.
   - Идеален для расширения датасета учебниками по машиностроению и литейному производству.

*(Perplexity MCP исключён из активной конфигурации, так как требует платную подписку на API).*

---

## ⚙️ Файл конфигурации (`mcp_config.json`)

Конфигурационный файл расположен по глобальному пути:
`C:\Users\Yarchik\.gemini\config\mcp_config.json`

```json
{
  "mcpServers": {
    "github": {
      "command": "C:\\Program Files\\nodejs\\npx.cmd",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "PATH": "C:\\Program Files\\nodejs;%PATH%"
      }
    },
    "exa": {
      "command": "C:\\Program Files\\nodejs\\npx.cmd",
      "args": ["-y", "exa-mcp-server"],
      "env": {
        "PATH": "C:\\Program Files\\nodejs;%PATH%"
      }
    }
  }
}
```

---

## 🔑 Безопасная настройка API Ключей (токены доступа)

> [!IMPORTANT]
> Никогда не отправляйте приватные ключи и токены в текст чата! Задайте их через переменные среды Windows (User Environment Variables). Они подхватятся автоматически.

### 1. GitHub Personal Access Token (`GITHUB_PERSONAL_ACCESS_TOKEN`)
1. Зайдите на GitHub: **Settings -> Developer Settings -> Personal access tokens (Tokens classic)**.
2. Сгенерируйте токен с правами `repo` (чтение/запись репозиториев).
3. В терминале PowerShell выполните команду:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("GITHUB_PERSONAL_ACCESS_TOKEN", "ваш_токен_ghp_...", "User")
   ```

### 2. Exa API Key (`EXA_API_KEY`)
1. Зарегистрируйтесь на [dashboard.exa.ai](https://dashboard.exa.ai/api-keys) (есть бесплатный лимит запросов).
2. Скопируйте ключ и сохраните в переменные среды:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("EXA_API_KEY", "ваш_ключ_exa_...", "User")
   ```

### 3. Perplexity API Key (`PERPLEXITY_API_KEY`)
1. Получите ключ в [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api).
2. Задайте в PowerShell:
   ```powershell
   [System.Environment]::SetEnvironmentVariable("PERPLEXITY_API_KEY", "pplx-...", "User")
   ```

---

## 🔗 Связанные заметки
- [[00 - Главная (MOC)|🏠 Главная страница]]
- [[07 - Инструкции по развертыванию и запуску|🚀 Инструкции по развертыванию]]
- [[10 - Сбор датасета и Парсинг методичек (Miner)|🌐 Сбор датасета и Парсинг методичек]]
