import os, cv2, csv, argparse, datetime
from ultralytics import YOLO
from tqdm import tqdm
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from types import SimpleNamespace


def get_valid_image_paths(source_dir):
    """Сканирует директорию на наличие валидных изображений."""
    image_paths = []
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    
    files = os.listdir(source_dir)
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in allowed_extensions and cv2.imread(os.path.join(source_dir, file)) is not None:
            image_paths.append(os.path.join(source_dir, file))    
    return image_paths 
    


def analyze_image_metrics(detections, image_area, model_names, target_classes):
    """Анализирует результат детекции и возвращает метрики."""
    # Инициализируем счётчики для каждого целевого класса
    object_counts = {transport_cls: 0 for transport_cls in target_classes}
    # Список для хранения площадей всех bounding box'ов
    total_bbox_area = 0

    for box in detections:
        class_id = int(box.cls[0].item())
        class_name = model_names[class_id]
        
        # Если класс объекта входит в целевые классы, увеличиваем счётчик
        if class_name in target_classes:
            object_counts[class_name] += 1
            
            # Вычисляем площадь bounding box'а            
            xmin, ymin, xmax, ymax = [float(value) for value in box.xyxy[0].tolist()]
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            total_bbox_area += area
    # Считаем общее количество транспортных средств
    total_vehicles = sum(object_counts.values())
    # Вычисляем плотность - долю изображения, занятую объектами
    density = total_bbox_area / image_area if image_area > 0 else 0
    # Вычисляем среднюю площадь bounding box'а
    avg_bbox_area = total_bbox_area / total_vehicles if total_vehicles > 0 else 0

    # Формируем отчёт с метриками
    report = {
        'total_vehicles': total_vehicles,
        'density': round(density, 4),
        'avg_bbox_area': round(avg_bbox_area, 2),
        **object_counts  # Добавляем счётчики по каждому целевому классу
    }

    return report 


def find_highlight_examples(all_reports, top_n=3):
    if len(all_reports) < top_n:
        print(f"  -> Найдено всего {len(all_reports)} отчётов, будут использованы все.")
        return {r['filename']: r for r in all_reports}

    most_crowded = sorted(all_reports, key=lambda r: r['total_vehicles'], reverse=True)[:top_n]
    most_dense = sorted(all_reports, key=lambda r: r['density'], reverse=True)[:top_n]

    top_examples = {r['filename']: r for r in most_crowded}

    top_examples.update({r['filename']: r for r in most_dense})

    return top_examples 


def generate_visualizations(model, examples_to_visualize, source_dir, output_dir, conf):
    annotated_examples = []

    os.makedirs(output_dir, exist_ok=True)

    for filename, report in examples_to_visualize.items():
        original_path = os.path.join(source_dir, filename)

        results = model(original_path, conf=conf, verbose=False)
        annotated_img = results[0].plot()

        save_path = os.path.join(output_dir, f"highlight_{filename}")
        cv2.imwrite(save_path, annotated_img)
        
        stats_text = (
            f"Имя файла: {report['filename']}\n"
            f"Тип сцены: {report.get('scene_type', 'N/A')}\n"
            f"Всего ТС: {report['total_vehicles']}\n"
            f"  - Легковые автомобили: {report.get('car', 0)}\n"
            f"  - Грузовики: {report.get('truck', 0)}\n"
            f"Плотность объектов: {report['density']:.2%}\n"
            f"Средний размер объекта: {report['avg_bbox_area']:.0f} пикс."
        )
        
        annotated_examples.append({
            'path': save_path,
            'report': report,
            'stats': stats_text
        })
 
    return annotated_examples 


def classify_scene(report, thresholds):
    count = report['total_vehicles']
    density = report['density']

    # Empty scene - нет транспортных средств
    if count == 0:
        return 'Empty'

    # Traffic Jam - много машин и высокая плотность (пробка)
    if count >= thresholds['jam_count'] and density >= thresholds['jam_density']:
        return 'Traffic Jam'

    # Single Big Object - аномалия: 1-2 объекта создают высокую плотность
    if count <= 2 and density > thresholds['single_density']:
        return 'Single Big Object'

    # Heavy Traffic - много машин, но не пробка
    if count >= thresholds['heavy_count']:
        return 'Heavy Traffic'

    # Sparse Traffic - все остальные случаи (несколько машин, свободное движение)
    return 'Sparse Traffic' 

def test_get_valid_image_paths():
    test_dir = "test_sandbox"
    
    # Файлы, которые должны быть найдены
    expected_files = ["image1.jpg", "image2.PNG"]
    # Файлы, которые должны быть проигнорированы
    ignored_files = ["corrupted.jpg", "document.txt", "archive.zip"]
    

    # Вызываем тестируемую функцию
    result_paths = get_valid_image_paths(test_dir)
    
    # Проверяем результаты с помощью assert
    assert len(result_paths) == len(expected_files), \
        f"ОШИБКА: Ожидалось {len(expected_files)} файла, но найдено {len(result_paths)}"

    result_filenames = sorted([os.path.basename(p) for p in result_paths])
    expected_filenames = sorted(expected_files)
    assert result_filenames == expected_filenames, \
        f"ОШИБКА: Имена файлов не совпадают. Найдено: {result_filenames}, Ожидалось: {expected_filenames}"

def test_analyze_image_metrics():
    # box.cls[0] и box.xyxy[0] должны быть похожи на тензоры
    fake_detections = [
        # Первый бокс: 'car'
        SimpleNamespace(cls=np.array([0]), xyxy=np.array([[10, 10, 110, 110]])),
        # Второй бокс: 'truck'
        SimpleNamespace(cls=np.array([1]), xyxy=np.array([[50, 50, 250, 250]])),
        # Третий бокс: еще один 'car'
        SimpleNamespace(cls=np.array([0]), xyxy=np.array([[20, 20, 70, 70]])),
        # Четвёртый бокс: неизвестный класс (id=5), должен быть проигнорирован
        SimpleNamespace(cls=np.array([5]), xyxy=np.array([[0, 0, 5, 5]]))
    ]
    
    # Эмулируем остальные входные данные
    image_area = 1000 * 1000  # 1,000,000 пикселей
    model_names = {0: 'car', 1: 'truck', 5: 'person'} # Словарь имен
    target_classes = ['car', 'truck']

    result_metrics = analyze_image_metrics(fake_detections, image_area, model_names, target_classes)

    assert result_metrics['car'] == 2, f"ОШИБКА: Ожидалось 2 машины, но найдено {result_metrics['car']}"
    assert result_metrics['truck'] == 1, f"ОШИБКА: Ожидалось 1 грузовик, но найдено {result_metrics['truck']}"

    assert result_metrics['total_vehicles'] == 3, f"ОШИБКА: Ожидалось 3 ТС, но найдено {result_metrics['total_vehicles']}"

    # Площади: (100*100) + (200*200) + (50*50) = 10000 + 40000 + 2500 = 52500
    # Плотность: 52500 / 1000000 = 0.0525
    expected_density = 0.0525
    assert np.isclose(result_metrics['density'], expected_density), \
        f"ОШИБКА: Ожидалась плотность {expected_density}, но получено {result_metrics['density']}"

def test_classify_scene():
    """(ТЕСТ) Проверяет корректность работы классификатора сцен classify_scene."""
    # Определяем тестовые пороги
    test_thresholds = {
        'jam_count': 10, 'jam_density': 0.3,
        'heavy_count': 5, 'single_density': 0.15,
    }

    # Создаём набор тестовых сценариев (кейсов)
    test_cases = [
        # Имя теста, Входной отчет, Ожидаемый результат
        ("Пустая сцена", {'total_vehicles': 0, 'density': 0.0}, 'Empty'),
        
        ("Явная пробка", {'total_vehicles': 15, 'density': 0.4}, 'Traffic Jam'),
        
        ("Граничный случай пробки (по количеству)", {'total_vehicles': 11, 'density': 0.31}, 'Traffic Jam'),
        
        ("Не пробка (не хватает плотности)", {'total_vehicles': 15, 'density': 0.29}, 'Heavy Traffic'),
        
        ("Не пробка (не хватает количества)", {'total_vehicles': 10, 'density': 0.4}, 'Heavy Traffic'),
        
        ("Аномалия: одна большая фура", {'total_vehicles': 1, 'density': 0.2}, 'Single Big Object'),
         
        ("Аномалия: две большие машины", {'total_vehicles': 2, 'density': 0.16}, 'Single Big Object'),
        
        ("Не аномалия (слишком низкая плотность)", {'total_vehicles': 1, 'density': 0.14}, 'Sparse Traffic'),
        
        ("Плотное движение", {'total_vehicles': 7, 'density': 0.2}, 'Heavy Traffic'),
        
        ("Граничный случай плотного движения", {'total_vehicles': 6, 'density': 0.1}, 'Heavy Traffic'),
        
        ("Свободная дорога (мало машин)", {'total_vehicles': 4, 'density': 0.1}, 'Sparse Traffic'),
    ]

    # Прогоняем все тесты в цикле
    for i, (case_name, report, expected_class) in enumerate(test_cases):
        actual_class = classify_scene(report, test_thresholds)
        
        assert actual_class == expected_class, f"ОШИБКА в кейсе '{case_name}': Ожидалось '{expected_class}', но получено '{actual_class}'"

class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Пробуем использовать красивые шрифты, если они доступны (их мы скачивали в нашу папку ttf/)
        regular_font_path = os.path.join('ttf', 'DejaVuSans.ttf')
        bold_font_path = os.path.join('ttf', 'DejaVuSans-Bold.ttf')

        if os.path.exists(regular_font_path) and os.path.exists(bold_font_path):
            self.add_font('DejaVu', '', regular_font_path)
            self.add_font('DejaVu', 'B', bold_font_path)

            self.font_family = 'DejaVu'
        else:
            # Если шрифты не найдены, используем стандартный
            self.font_family = 'Arial'

    def header(self):
        """Создаёт шапку для каждой страницы"""
        self.set_font(self.font_family, 'B', 15)
        self.cell(0, 10, 'Аналитический отчёт по дорожной обстановке', border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.set_font(self.font_family, '', 8)
        self.cell(0, 5, f'Дата генерации: {datetime.date.today().strftime("%d.%m.%Y")}', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        """Добавляет номера страниц в подвале"""
        self.set_y(-15)
        self.set_font(self.font_family, 'B', 8)
        self.cell(0, 10, f'Страница {self.page_no()}', border=0, new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')

    def chapter_title(self, title):
        """Создаёт заголовок раздела"""
        self.set_font(self.font_family, 'B', 12)
        self.cell(0, 10, title, border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        self.ln(5)

    def chapter_body(self, body):
        """Добавляет основной текст"""
        self.set_font(self.font_family, '', 10)
        self.multi_cell(0, 5, body)
        self.ln()
    
    def add_image_section(self, title, image_path, stats_text):
        """Добавляет секцию с изображением и статистикой"""
        self.add_page()
        self.chapter_title(title)

        # Центрируем изображение на странице
        image_width = 100
        page_width = self.w - 2 * self.l_margin
        x_position = (page_width - image_width) / 2 + self.l_margin
        self.image(image_path, x=x_position, y=None, w=image_width)
        self.ln(5)
        self.set_font(self.font_family, '', 10) 
        self.multi_cell(0, 5, stats_text) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Инструмент для анализа дорожного трафика на основе YOLOv8.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "mode", 
        choices=['experiment', 'report'],
        help="Режим работы скрипта:\n"
             " 'experiment' - запуск одного прогона для сбора CSV-статистики.\n"
             " 'report' - полный цикл анализа с генерацией PDF-отчёта."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.45, 
        help="Порог уверенности (confidence) для детекции. (По умолчанию: 0.45)"
    )

    args = parser.parse_args()

    # Выбор режима работы на основе аргументов

    if args.mode == 'experiment':
        # Определяем константы для этого режима
        TARGET_CLASSES = ['car', 'truck']
        
        # Получаем список валидных изображений
        image_paths = get_valid_image_paths('data')

        model = YOLO('yolov8n.pt')
        
        # Готовимся собирать отчёты со всех изображений
        all_reports = []
        
        # Основной цикл обработки, обёрнутый в tqdm для наглядности
        for path in tqdm(image_paths, desc=f"Анализ [conf={args.conf}]"):
            try:
                # Читаем изображение и получаем его размеры
                image = cv2.imread(path)
                h, w, _ = image.shape
                image_area = h * w
                # Запускаем инференс с заданным `conf`
                results = model(image, conf=args.conf, verbose=False)
                
                # Извлекаем метрики из результатов детекции
                metrics = analyze_image_metrics(results[0].boxes, image_area, model.names, TARGET_CLASSES)
                
                # Дополняем отчёт информацией об изображении
                metrics['filename'] = os.path.basename(path)
                all_reports.append(metrics)
            except Exception as e:
                print(f"Критическая ошибка при обработке файла {path}: {e}")
        
        # Сохранение результатов в CSV-файл
        if all_reports:
            # Создаём папку для экспериментов, если она ещё не существует
            os.makedirs('experiments', exist_ok=True)
            # Имя файла будет отражать параметр, с которым проводился эксперимент
            csv_path = os.path.join('experiments', f'analysis_conf_{args.conf}.csv')

            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                # Заголовки берём из ключей первого словаря в списке
                writer = csv.DictWriter(f, fieldnames=all_reports[0].keys())
                writer.writeheader()
                writer.writerows(all_reports)

    elif args.mode == 'report':
        REPORT_OUTPUT_DIR = 'report_output'
        TARGET_CLASSES = ['car', 'truck']
        SOURCE_DIR = 'data'
        THRESHOLDS = {
            'jam_count': 10, 'jam_density': 0.3,
            'heavy_count': 5, 'single_density': 0.15,
        }

        # Загружаем модель и собираем данные
        model = YOLO('yolov8n.pt')
        image_paths = get_valid_image_paths(SOURCE_DIR)
        
        all_reports = []
        for path in tqdm(image_paths, desc="Анализ изображений"):
            image = cv2.imread(path)
            h, w, _ = image.shape
            image_area = h * w
            results = model(image, conf=args.conf, verbose=False)
            
            metrics = analyze_image_metrics(results[0].boxes, image_area, model.names, TARGET_CLASSES)
            metrics['scene_type'] = classify_scene(metrics, THRESHOLDS)
            metrics['filename'] = os.path.basename(path)
            all_reports.append(metrics)

        if not all_reports:
            exit()
            
        # Курируем данные и создаём визуализации
        top_examples = find_highlight_examples(all_reports)
        annotated_examples = generate_visualizations(model, top_examples, SOURCE_DIR, REPORT_OUTPUT_DIR, args.conf)
        
        # Сборка PDF-отчёта
        pdf = PDFReport()
        pdf.add_page()
        
        # Общая сводка
        pdf.chapter_title("1. Общая сводка по проанализированным данным")
        total_vehicles_found = sum(r['total_vehicles'] for r in all_reports)
        total_cars = sum(r.get('car', 0) for r in all_reports)
        total_trucks = sum(r.get('truck', 0) for r in all_reports)
        scene_counts = {scene: len([r for r in all_reports if r['scene_type'] == scene]) for scene in ['Traffic Jam', 'Heavy Traffic', 'Sparse Traffic', 'Single Big Object', 'Empty']}

        summary_text = (
            f"Всего обработано изображений: {len(all_reports)}\n"
            f"Использованный порог уверенности: {args.conf}\n\n"
            f"ОБЩАЯ СТАТИСТИКА ТРАНСПОРТА:\n"
            f"  - Всего найдено ТС: {total_vehicles_found}\n"
            f"  - Легковые автомобили: {total_cars}\n"
            f"  - Грузовики: {total_trucks}\n\n"
            f"КЛАССИФИКАЦИЯ СЦЕН:\n"
            f"  - Пробка/Затор: {scene_counts['Traffic Jam']} изображений\n"
            f"  - Плотное движение: {scene_counts['Heavy Traffic']} изображений\n"
            f"  - Свободная дорога: {scene_counts['Sparse Traffic']} изображений\n"
            f"  - Аномалии (крупный объект): {scene_counts['Single Big Object']} изображений\n"
            f"  - Пустые сцены: {scene_counts['Empty']} изображений"
        )
        pdf.chapter_body(summary_text)

        # Визуальные примеры
        pdf.chapter_title("2. Примеры показательных сцен")
        sorted_examples = sorted(annotated_examples, key=lambda x: x['report']['density'], reverse=True)
        
        for i, example in enumerate(sorted_examples):
            title = f"Пример #{i+1}: {example['report']['filename']}"
            pdf.add_image_section(title, example['path'], example['stats'])

        # Сохранение файла
        pdf_output_path = os.path.join(REPORT_OUTPUT_DIR, "traffic_analysis_report.pdf")
        pdf.output(pdf_output_path)
    
