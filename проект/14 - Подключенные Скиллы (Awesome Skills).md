# ⚡ Подключенные Скиллы (Awesome Skills)

> **Источник:** [Awesome Skills](https://awesomeskill.ai/)  
> **Расположение в проекте:** `.agents/skills/`  
> **Теги:** #skills #awesomeskills #recsys #ui_ux #planning #архитектура

---

## 🎯 Выбранные и установленные навыки для проекта

С каталога **Awesome Skills** отобраны и установлены 4 мощных специализированных навыка, идеально подходящих под задачи сервиса ценообразования отливок БВК:

### 1. 🥇 `recsys-pipeline-architect` (Архитектор рекомендательных систем)
* **Путь:** [`.agents/skills/recsys-pipeline-architect/SKILL.md`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/.agents/skills/recsys-pipeline-architect/SKILL.md)
* **Прямая польза для проекта:**
  Наш модуль подбора аналогов (`analogs/`) — это по сути классическая рекомендательная система (RecSys). 
  Скилл задает строгий шестиступенчатый фреймворк:
  $$\text{Source} \rightarrow \text{Hydrator} \rightarrow \text{Filter} \rightarrow \text{Scorer} \rightarrow \text{Selector} \rightarrow \text{SideEffect}$$
  1. **Source:** База выполненных отливок завода БВК (`gallery.npz`).
  2. **Hydrator:** Подгрузка марки стали, группы отливки, истории брака и маршрутных карт.
  3. **Filter:** Отсечение технологически несовместимых деталей (например, мелкие крепежи против массивных корпусов).
  4. **Scorer:** Мультимодальный скоринг (косинусная близость векторов DINOv2 + дельта масс + совпадение термообработки).
  5. **Selector:** Отбор Top-3 наиболее похожих деталей-аналогов.
  6. **SideEffect:** Расчет разницы в себестоимости и предупреждение технолога о рисках.

---

### 2. 🎨 `ui-ux-pro-max` (Интеллектуальный UI/UX дизайн)
* **Путь:** [`.agents/skills/ui-ux-pro-max/SKILL.md`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/.agents/skills/ui-ux-pro-max/SKILL.md)
* **Прямая польза для проекта:**
  Содержит 50 готовых дизайн-стилей, 21 цветовую палитру и правила типографики для промышленных корпоративных систем.
  Помогает превратить интерфейс [`web_app.py`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/web_app.py) и интерактивный отчет [`report.html`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/report.html) из учебного прототипа в стильный, премиальный заводской дашборд уровня корпорации КОНАР.

---

### 3. 📋 `planning-with-files` (Файловое планирование в стиле Manus)
* **Путь:** [`.agents/skills/planning-with-files/SKILL.md`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/.agents/skills/planning-with-files/SKILL.md)
* **Прямая польза для проекта:**
  Внедряет методику ведения структурированных планов (`task_plan.md`, `findings.md`) при решении комплексных задач. Предотвращает потерю контекста между сессиями.

---

### 4. 🌐 `web-design-guidelines` (Стандарты веб-интерфейсов Vercel Labs)
* **Путь:** [`.agents/skills/web-design-guidelines/SKILL.md`](file:///C:/Users/Yarchik/OneDrive/Desktop/проект/.agents/skills/web-design-guidelines/SKILL.md)
* **Прямая польза для проекта:**
  Аудит верстки интерфейса, контрастность, читаемость таблиц калькулятора и отзывчивость элементов управления.

---

## 🔗 Связанные заметки
- [[00 - Главная (MOC)|🏠 Главная страница]]
- [[02 - Архитектура и Пайплайн системы|⚙️ Архитектура системы]]
- [[04 - Поиск аналогов и Компьютерное зрение|🔍 Поиск аналогов]]
- [[11 - Калькулятор себестоимости и Веб-сервис (Web UI)|💰 Веб-сервис и Калькулятор]]
