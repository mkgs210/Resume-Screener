import re
import fitz  # pymupdf
from tqdm import tqdm
from glob import glob
import pandas as pd

# ======================
# Предварительные списки
# ======================
# (Для простоты: часть списков укорочена, можно расширять по необходимости.)

# Примеры технологий / библиотек
TECH_KEYWORDS = [
    'yolo', 'ignite', 'oracle pl/sql', 'jupyter', 'theory of probability',
    'sklearn', 'pandas', 'github', 'gitlab', 'kaggle', 'bitbucket', 'linkedin',
    'theory of probability', 'kafka', 'статистический анализ', 'nats', 'onnx',
    'разработка по', 'prod2vec', 'predictive modeling', 'optuna', 'hyperopt',
    'работа в команде', 'nlp', 'sql', 'data visualization', 'c++', 'skimage',
    'компьютерная лингвистика', 'fastapi', 'web scraping', 'eda', 'matplotlib',
    'amazon aws', 'impala', 'jupyter notebook', 'seaborn', 'elasticsearch',
    'django framework', 'a/b testing', 'коммуникабельность', 'javascript', 'rfe',
    'bash', 'analytical skills', 'lightgbm', 'wsl', 'аналитические исследования',
    'forecasting', 'математический анализ', 'java', 'deep learning', 'r', 'mysql',
    'statistical testing', 'ensemble methods', 'docker', 'time series analysis',
    'statistics', 'bert4rec', 'data science', 'cv', 'анализ данных',
    'аналитическое мышление', 'unsupervised learning', 'surprise',
    'python scientific stack', 'plotly', 'git', 'dvc', 'haskell',
    'ensembles of algorithms', 'pillow', 'численные исследования', 'r shiny',
    'xgboost', 'spark', 'hive', 'data scientist', 'dlib', 'reinforcement learning',
    'jenkins', 'scikit-learn', 'colab', 'ibm', 'linux', 'fbprophet',
    'natural language processing', 'tensor flow', 'data analysis', 'ms excel',
    'flask', 'scala', 'android', 'pymystem', 'gui', 'postgresql', 'business english',
    'cuda', 'математическая статистика', 'big data', 'beautifulsoup', 'boosting',
    'c#', 'keras', 'monte carlo models', 'agile', 'aws', 'nltk', 'tensorflow',
    'apache spark', 'numpy', 'tableau', 'phd', 'opencv2', 'neural networks',
    'lightning', 'catboost', 'обучаемость', 'data mining', 'alm', 'networkx',
    'pytest', 'hadoop', 'a/b тесты', 'mlflow', 'classification', 'ооп', 'ml',
    'machine learning', 'azure cloud', 'decomposition', 'pyspark', 'speech',
    'regression', 'presentation skills', 'a/b тестирование', 'confluence',
    'seasborn', 'tidyverse', 'scipy', 'computer vision', 'teamplayer', 'python',
    'catalyst', 'комбинаторная оптимизация', 'flutter', 'oracle', 'dash',
    'multiprocessing', 'time management', 'opencv', 'pytorch', 'clickhouse',
    'clustering', 'jira', 'kubernetes', 'anaconda', 'python machine learning stack'
]

# A/B тесты (ключевые слова)
AB_TEST_KEYWORDS = [
    'a/b test',
    'a/b тест',
    'a/b testing',
    'ab testing',
    'a/b эксперименты'
]

# Стат. термины
STAT_KEYWORDS = [
    'p-value',
    'pvalue',
    't-test',
    'anova',
    'гипотеза',
    'hypothesis testing',
    'p-значение',
    'z-test',
    'chi-square',
    'mann-whitney',
    'wilcoxon'
]

# Топовые университеты
UNIVERSITIES = [
    # МГУ
    "мгу", "moscow state university", "lomonosov msu", "msu",
    
    # СПбГУ
    "спбгу", "saint petersburg state university", "spbu",
    
    # ВШЭ
    "вшэ", "высшая школа экономики", "hse", "higher school of economics",
    
    # МФТИ
    "мфти", "moscow institute of physics and technology", "mipt",
    
    # МГТУ / Бауманка / BMSTU
    "мгту", "бауманка", "bmstu", "bauman", "bauman moscow state technical university",
    
    # МГИМО
    "мгимо", "mgimo", "moscow state institute of international relations",
    
    # ИТМО
    "итмо", "itmo", "university itmo", 
    "information technologies mechanics and optics",  # иногда пишут расшифровку
    
    # МИФИ
    "мифи", "mephi", "moscow engineering physics institute",
    
    # СПбПУ (Политех Петра Великого)
    "спбпу", "spbstu", "peter the great st. petersburg polytechnic university",
    
    # УФУ (УрФУ)
    "уфу", "урфу", "urfu", "urals federal university", "ural federal university",
    
    # МАИ
    "маи", "mai", "moscow aviation institute",
    
    # ЮФУ
    "юфу", "sfedu", "southern federal university",
    
    # ЛЭТИ
    "лэти", "leti", "etu", "saint petersburg electrotechnical university",
    
    # ТПУ
    "тпу", "tpu", "tomsk polytechnic university",
    
    # Иннополис
    "иннополис", "innopolis university", "innopolis",
    
    # Станкин
    "станкин",  # иногда встречается "mstu «stan»", но крайне редко
    # Можно добавить при желании
    
    # МЭИ
    "мэи", "mpei", "moscow power engineering institute",
    
    # ННГУ (Университет Лобачевского)
    "ннигу", "лобачевского", "lobachevsky university", "unn", 
    "nizhny novgorod state university",
    
    # МИСиС
    "мисис", "misis", "nust misis", "national university of science and technology misis",
    
    # КФУ
    "кфу", "kfu", "kazan federal university",
    
    # НГУ
    "нгу", "nsu", "novosibirsk state university",
    
    # МИРЭА
    "мирэа", "mirea", "rtu mirea",  # Российский технологический университет
    
    # СПбГУТ
    "спбгут",  # Санкт-Петербургский университет телекоммуникаций
    # Можно добавить англ. вариант, но он редко встречается
    
    # МИЭТ
    "миэт", "miet", "moscow institute of electronic technology",
    
    # ИГТУ (Иркутский гос. тех. университет) / ИРНИТУ
    "игту", "ирниту", "istu", "irkutsk national research technical university",
    
    # МТУСИ
    "мтуси", "mtuci", "moscow technical university of communications and informatics",
    
    # МИИТ (РУТ)
    "миит", "rutransport", "russian university of transport", 
    # иногда mgups, но реже
    
    # ВГУ (Воронежский государственный университет)
    "вгу", "vgu", "voronezh state university"
]

# Топовые курсы / школы
TOP_COURSES = [
    'шад',
    'skillfactory',
    'skypro',
    'karpov.courses',
    'ods',
    'stepik',
    'deep learning school',
    'coursera',
    'edx',
    'udacity',
    'datacamp',
    'яндекс.практикум',
    'praktikum.yandex',
    'geekbrains',
    'skillbox',
    'нетология'
]

# Топовые журналы
JOURNALS = [
    'elibrary',
    'scopus',
    'ринц',
    'web of science',
    'ieee xplore',
    'springerlink',
    'sciencedirect',
    'pubmed'
]

# Типовые job titles (укороченный список)
JOB_TITLES = [
    'data scientist',
    'data analyst',
    'machine learning engineer',
    'ml engineer',
    'deep learning engineer',
    'devops engineer',
    'devops',
    'software engineer',
    'software developer',
    'backend engineer',
    'frontend engineer',
    'fullstack engineer',
    'mobile developer',
    'ios developer',
    'android developer',
    'qa engineer',
    'quality assurance engineer',
    'product manager',
    'project manager',
    'scrum master',
    'team lead',
    'tech lead',
    'research scientist',
    'research engineer',
    'system analyst',
    'bi analyst'
]

# Бренды-компании
BRAND_NAMES = [
    'google',
    'amazon',
    'facebook',
    'meta',
    'microsoft',
    'apple',
    'tesla',
    'yandex',
    'яндекс',
    'сбер',
    'tinkoff',
    'gazprom',
    'epam',
    'luxoft',
    'uber',
    'netflix',
    'booking'
]

# Заголовки, по которым будем пытаться понять структуру резюме
SECTION_HEADERS = [
    'experience', 'work experience', 'опыт работы',
    'education', 'образование',
    'skills', 'навыки',
    'projects', 'проекты',
    'achievements', 'достижения',
    'publications', 'публикации',
    'certificates', 'сертификаты',
    'summary', 'обо мне',
    'languages', 'языки'
]

# Регулярка для зарплаты
# Ищем упоминания чисел + (руб/р/₽/$/доллар/usd):
SALARY_REGEX = re.compile(
    r'(\d+[\s\-]*(k|тыс|тысяч)?[\s\-]*(р|руб|₽|\$|доллар|usd))', re.IGNORECASE
)

# ======================
# Функции для подсчёта
# ======================

def get_text_and_doc(pdf_path):
    """Читаем PDF и возвращаем (text, doc).
       text: полный текст
       doc: объект fitz.Document (для кол-ва страниц)
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text, doc

def normalize_text(t: str) -> str:
    """Приводим текст к нижнему регистру для упрощения поиска."""
    return t.lower()

def count_occurrences(keywords, text_norm):
    """Подсчитываем, сколько ключевых слов из списка встретилось в тексте (хотя бы по одному разу).
       Возвращаем целое число: сколько ключей вообще найдено."""
    count = 0
    for kw in keywords:
        if kw in text_norm:
            count += 1
    return count

def count_multiple_occurrences(keywords, text_norm):
    """Подсчитываем суммарное количество всех вхождений (не только уникальных) для ряда ключей.
       Например, если text = "a/b test a/b test" и keywords=["a/b test"], то получим 2."""
    total = 0
    for kw in keywords:
        total += text_norm.count(kw)
    return total

def extract_numeric_features(text: str, doc: fitz.Document) -> dict:
    """Извлекает ВСЕ требуемые числовые фичи из текста резюме."""
    text_norm = normalize_text(text)
    
    # 1. Метрики объёма
    length_chars = len(text)
    length_words = len(text.split())
    num_pages = doc.page_count if doc else 1  # если doc=None, считаем 1
    
    # 2. Ссылки
    all_links = re.findall(r'http[s]?://\S+', text, flags=re.IGNORECASE)
    num_all_links = len(all_links)
    num_github_links = sum('github.com' in link.lower() for link in all_links)
    num_kaggle_links = sum('kaggle.com' in link.lower() for link in all_links)
    num_linkedin_links = sum('linkedin.com' in link.lower() for link in all_links)

    # 3. Технологии (из списка TECH_KEYWORDS)
    tech_count = count_occurrences(TECH_KEYWORDS, text_norm)
    
    # Плотность
    skill_density = 0.0
    if length_words > 0:
        skill_density = tech_count / length_words

    # 4. A/B тесты, статистические термины
    ab_test_count = count_multiple_occurrences(AB_TEST_KEYWORDS, text_norm)
    stat_count = count_multiple_occurrences(STAT_KEYWORDS, text_norm)

    # 5. Университеты, курсы, журналы
    uni_count = count_occurrences(UNIVERSITIES, text_norm)
    course_count = count_occurrences(TOP_COURSES, text_norm)
    journals_count = count_occurrences(JOURNALS, text_norm)

    # 6. Зарплата (ищем совпадения по SALARY_REGEX)
    # Возвращаем 1, если нашли хотя бы одно совпадение
    has_salary = 1 if SALARY_REGEX.search(text_norm) else 0

    # 7. Проверка на экспорт с hh.ru
    #    Часто в pdf написано: "Печатная версия резюме на hh.ru" или "hh.ru/resume"
    is_hh_export = 1 if ("hh.ru" in text_norm or "резюме на hh.ru" in text_norm) else 0

    # 8. Проверка на экспорт с LinkedIn
    #    Часто в pdf написано: "LinkedIn" или "linkedin export"
    is_linkedin_export = 1 if ("linkedin" in text_norm and "export" in text_norm) else 0
    # Или более простой вариант:
    # is_linkedin_export = 1 if "linkedin" in text_norm else 0

    # 9. experience_mentions: количество раз упоминается "N лет" или "N years"
    #    Можно сделать простенькую регулярку
    pattern_years_ru = re.findall(r'\d+\+?\s*лет', text_norm)
    pattern_years_en = re.findall(r'\d+\+?\s*years', text_norm)
    experience_mentions = len(pattern_years_ru) + len(pattern_years_en)

    # 10. Job titles (из списка), бренды
    jobtitle_count = count_occurrences(JOB_TITLES, text_norm)
    brands_count = count_occurrences(BRAND_NAMES, text_norm)

    # 11. Смотрим, сколько стандартных разделов (Experience, Education и т.д.) реально есть
    num_struct_sections = 0
    for header in SECTION_HEADERS:
        if header in text_norm:
            num_struct_sections += 1

    return {
        "length_chars": length_chars,
        "length_words": length_words,
        "num_pages": num_pages,
        "num_all_links": num_all_links,
        "num_github_links": num_github_links,
        "num_kaggle_links": num_kaggle_links,
        "num_linkedin_links": num_linkedin_links,
        "tech_count": tech_count,
        "skill_density": skill_density,
        "ab_test_count": ab_test_count,
        "stat_count": stat_count,
        "uni_count": uni_count,
        "course_count": course_count,
        "journals_count": journals_count,
        "has_salary": has_salary,
        "is_hh_export": is_hh_export,
        "is_linkedin_export": is_linkedin_export,
        "experience_mentions": experience_mentions,
        "jobtitle_count": jobtitle_count,
        "brands_count": brands_count,
        "num_struct_sections": num_struct_sections
    }

def main():
    df = pd.read_csv('./interns_preprocessed/interns_preprocessed.csv').dropna().reset_index(drop=True).sort_values('Резюме')
    # Колонки под фичи
    feature_columns = [
        "length_chars", "length_words", "num_pages",
        "num_all_links", "num_github_links", "num_kaggle_links", "num_linkedin_links",
        "tech_count", "skill_density", "ab_test_count", "stat_count",
        "uni_count", "course_count", "journals_count", 
        "has_salary", "is_hh_export", "is_linkedin_export", 
        "experience_mentions", "jobtitle_count", "brands_count", 
        "num_struct_sections"
    ]
    
    # Подготовим DataFrame для фич
    features_df = pd.DataFrame(columns=feature_columns)
    rows = []

    for i, pdf_path in enumerate(tqdm(df['Резюме'], total=len(df))):
        pdf_path = pdf_path.replace('\\', '\\\\')
        try:
            text, doc = get_text_and_doc(pdf_path)
        except Exception as e:
            print(f"Ошибка при чтении {pdf_path}: {e}")
            # Если файл не читается, ставим пустые значения или нули
            row = {col: 0 for col in feature_columns}
            rows.append(row)
            continue

        # Извлекаем фичи
        feats = extract_numeric_features(text, doc)
        rows.append(feats)

        # Закрываем doc
        doc.close()
    
    # Превращаем список в DataFrame
    features_df = pd.DataFrame(rows, columns=feature_columns)

    # При желании можно склеить с исходным df по индексу
    result_df = pd.concat([df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)

    # Сохраняем
    result_df.to_csv('./simple_features_interns.csv', index=False)
    print("Готово! Файл simple_features_interns.csv сохранён.")

if __name__ == "__main__":
    main()
