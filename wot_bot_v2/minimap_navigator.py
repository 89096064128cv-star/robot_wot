import cv2
import numpy as np
import mss
import pydirectinput
import time
import math

# --- НАСТРОЙКИ ---
# Жесткие координаты миникарты для 1920x1080
MINIMAP_X = 1578
MINIMAP_Y = 743
MINIMAP_W = 325
MINIMAP_H = 325

MONITOR = {"top": MINIMAP_Y, "left": MINIMAP_X, "width": MINIMAP_W, "height": MINIMAP_H}
sct = mss.mss()

# База маршрутов. 
# Точки указаны в пикселях ОТНОСИТЕЛЬНО левого верхнего угла миникарты.
# Если миникарта 325x325, то центр это (162, 162).
ROUTES = {
    "sand_river": {
        "top_spawn": (160, 160),    # Пример: ехать в центр
        "bottom_spawn": (160, 160)  # Пример: ехать в центр
    },
    # Сюда вы будете добавлять другие карты
}

# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def find_my_tank():
    """Ищет белую стрелку на миникарте и возвращает её центр (x, y)"""
    img = np.array(sct.grab(MONITOR))
    # Переводим в BGR
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Ищем белый цвет (стрелка игрока)
    lower_white = np.array([220, 220, 220])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(frame, lower_white, upper_white)

    # Убираем шум (эрозия и дилатация)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Берем самый большой контур (стрелка игрока)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 10: 
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
    return None

def navigate_to_point(my_pos, target_pos):
    """Вычисляет угол и нажимает клавиши для движения к цели"""
    if my_pos is None or target_pos is None:
        return

    my_x, my_y = my_pos
    target_x, target_y = target_pos

    # Вычисляем угол от танка к цели
    angle = math.atan2(target_y - my_y, target_x - my_x)
    angle_deg = math.degrees(angle)

    # Логика поворота (Упрощенная)
    # В WOT 0 градусов - это вправо. -90 - вверх. 90 - вниз. 180/-180 - влево.
    # Мы будем крутить танк, пока цель не окажется "впереди" (т.е. вверху миникарты).
    
    # Если цель справа (от -45 до 45)
    if -45 < angle_deg < 45:
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('d') # Поворачиваем направо
        time.sleep(0.05)
        pydirectinput.keyUp('d')
    # Если цель внизу (от 45 до 135)
    elif 45 <= angle_deg < 135:
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('d') # Крутимся дальше направо, чтобы развернуться
        time.sleep(0.1)
        pydirectinput.keyUp('d')
    # Если цель слева (от -135 до -45)
    elif -135 < angle_deg <= -45:
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a') # Поворачиваем налево
        time.sleep(0.05)
        pydirectinput.keyUp('a')
    # Если цель слева (сзади)
    else:
        pydirectinput.keyDown('w')
        pydirectinput.keyDown('a') 
        time.sleep(0.1)
        pydirectinput.keyUp('a')

def test_navigation():
    """Тестовый режим: едем к точке"""
    print("Запуск навигации. Нажмите Ctrl+C для остановки.")
    time.sleep(2)
    
    map_name = "sand_river" # Имя карты
    
    try:
        while True:
            my_pos = find_my_tank()
            
            if my_pos:
                # Определяем спавн: если мы в верхней половине (Y < 162), едем в нижнюю точку.
                # Если в нижней (Y > 162), едем в верхнюю.
                if my_pos[1] < 162:
                    target = ROUTES[map_name]["bottom_spawn"]
                    print(f"Спавн: Верхний. Едем в нижнюю точку.")
                else:
                    target = ROUTES[map_name]["top_spawn"]
                    print(f"Спавн: Нижний. Едем в верхнюю точку.")
                    
                navigate_to_point(my_pos, target)
            else:
                print("Не нашел свой танк на миникарте.")
                
            time.sleep(0.05) # 20 раз в секунду
            
    except KeyboardInterrupt:
        print("\nОстановка навигации.")
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')

# === ГЛАВНЫЙ БЛОК ===
if __name__ == "__main__":
    print("=" * 50)
    print("Навигатор по мини-карте.")
    print("=" * 50)
    
    # Для теста просто запускаем навигацию
    test_navigation()