from ultralytics import YOLO

model = YOLO('yolov8n.pt') 
source = 'https://ultralytics.com/images/bus.jpg'
results = model.predict(source)
result = results[0]

# Объект boxes содержит всю информацию о найденных рамках
boxes = result.boxes  

# Переберём каждую найденную рамку
for i, box in enumerate(boxes):
    print(f"\n--- Объект №{i+1} ---")

    # Координаты
    # .xyxy: возвращает координаты в формате [x_min, y_min, x_max, y_max]
    # .xywh: возвращает координаты в формате [x_center, y_center, width, height]
    # Данные возвращаются в виде тензора PyTorch, используем .tolist() для конвертации в список Python
    coords_xyxy = box.xyxy[0].tolist()
    print(f"Координаты (xmin, ymin, xmax, ymax): {coords_xyxy}")

    xmin, ymin, xmax, ymax = coords_xyxy
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    print(f"Площадь рамки: {int(area)}")

    # Класс объекта
    # .cls: возвращает ID класса в виде тензора
    class_id = int(box.cls[0].item())
    
    # Чтобы получить имя класса, используем словарь `names` из объекта модели
    class_name = model.names[class_id]
    print(f"Класс: ID={class_id}, Имя='{class_name}'")

    # Уверенность (Confidence)
    # .conf: возвращает уверенность предсказания
    confidence = box.conf[0].item()
    print(f"Уверенность: {confidence:.4f}") # выведем 4 знака после запятой 