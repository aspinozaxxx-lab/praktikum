from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2

# --- Подготовка ---
model = YOLO('yolov8n.pt') 
source = 'https://ultralytics.com/images/bus.jpg'


# Базовый случай (стандартные параметры)
results_base = model.predict(source, conf=0.25, iou=0.7)

# Повышенный conf (оставляем только очень уверенные детекции)
results_high_conf = model.predict(source, conf=0.85, iou=0.7)

# Повышенный iou (более лояльное оставление дубликатов)
results_low_iou = model.predict(source, conf=0.25, iou=0.9)

# Комбинированный случай: низкий conf, но строгий iou
results_combo = model.predict(source, conf=0.1, iou=0.1)

# Конвертируем результаты в RGB-изображения для Matplotlib
img_base = cv2.cvtColor(results_base[0].plot(), cv2.COLOR_BGR2RGB)
img_high_conf = cv2.cvtColor(results_high_conf[0].plot(), cv2.COLOR_BGR2RGB)
img_low_iou = cv2.cvtColor(results_low_iou[0].plot(), cv2.COLOR_BGR2RGB)
img_combo = cv2.cvtColor(results_combo[0].plot(), cv2.COLOR_BGR2RGB)

# Отрисовка
fig, axs = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Влияние параметров `conf` и `iou` на результат детекции', fontsize=20)

# Ячейка (0, 0): Базовый случай
axs[0, 0].imshow(img_base)
axs[0, 0].set_title("Базовый случай\nconf=0.25, iou=0.7", fontsize=14)
axs[0, 0].axis('off')

# Ячейка (0, 1): Повышенный conf
axs[0, 1].imshow(img_high_conf)
axs[0, 1].set_title("Повышаем `conf` до 0.9\n(Исчезают неуверенные детекции)", fontsize=14)
axs[0, 1].axis('off')

# Ячейка (1, 0): Пониженный iou
axs[1, 0].imshow(img_low_iou)
axs[1, 0].set_title("Повышаем `iou` до 0.9\n(Лояльное оставление дубликатов)", fontsize=14)
axs[1, 0].axis('off')

# Ячейка (1, 1): Комбинированный случай
axs[1, 1].imshow(img_combo)
axs[1, 1].set_title("Низкий `conf`=0.1, но строгий `iou`=0.1\n(Много кандидатов, но без дублей)", fontsize=14)
axs[1, 1].axis('off')

# Отображаем всё вместе
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Оставляем место для общего заголовка
plt.show() 