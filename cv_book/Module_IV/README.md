# Список заданий для выполнения
В папке data/tram располагаются клипы. C использованием угловых детекторов Хариса и методов проекционной геометрии необходимо:
- Найти ключевые точки в кадре с использованием детектора SIFT, найти общие точки между кадрами с использованием дескрипторов точек, выделить точки лежащие на земле и найти перемещение объекта
- Найти ключевые точки в кадре с использованием детектора SIFT, найти общие точки между кадрами с использованием дескрипторов точек, использовать метод RANSAC для устранения выбросов
- Оценить движется ли объект в кадре исходя из анализа ключевых точек
- Оценить неподвижен ли объект в кадре исходя из анализа ключевых точек, во время стоянки трамвая
- Используя детекторы ORB и SIFT для получения ключевых точек сравнить точность и надежность сопоставления на последовательных кадрах, а также оценить производительность вычислений
- На основе анализа поведения ключевых точек между кадрамиЮ оценить является ли область плоской (например дорога)  
