# Данные для повторной отрисовки Plane20

- `scene_00000.npz` — исходная синтетическая сцена, центры и маски сом.
- `activity.npz` — использованные трассы всех сом по кадрам, сдвиги и
  индексы доноров.
- `assignments.csv` — соответствие между синтетическими сомами и
  донорскими ROI.
- `generation.json` — параметры рендера, включая подгонку яркости и
  шума, начальное состояние генератора шума и индексы кадров сравнения.
- `reference_median.tiff` — числовая калибровочная матрица, используемая
  при восстановлении яркости сом.

Профиль тканей и фоновых полей читается из распакованного
`plane20_synthetic.zip`. Чтобы вычислить один кадр итоговой синтетики
в исходных единицах яркости, после установки зависимостей и подключения
базового архива выполните из корня комплекта.
Достаточно зависимостей из `requirements-render.txt`.

```bash
python - <<'PYTHON'
import sys
from pathlib import Path

import tifffile

sys.path.insert(0, "analysis/plane20_gcamp8f_response")
from plot_render_components import prepare_render, render_layers

state = prepare_render(Path("results/Plane20"))
image, *_ = render_layers(state, source_index=2)
tifffile.imwrite("synthetic_frame_0002.tiff", image.astype("float32"))
PYTHON
```

Индексы кадров начинаются с нуля. Исходный фильм и полный банк доноров
включены в метаданные как имена источников. Для отрисовки синтетики их
заменяют сохранённые калибровочные данные и назначенные трассы. Для нового
запуска калибровки используются входные TIFF.
