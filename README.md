# BebraFish · 3D-синтетика Plane20

Генератор создаёт объём ткани с сомами и снимает его на заданных глубинах.
Сцена определяется профилем и seed. У каждой сомы постоянные ID, центр XYZ,
полуоси, поворот в XY и рисунок свечения. Маска строится по сечению
эллипсоида. Оптический сигнал может быть виден и за пределами этого сечения.

## Быстрый запуск

Из корня репозитория, Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m simcalc_zebrafish_pipeline.generate_volume --profile demo --seed 42 \
  --n-planes 5 --z-step-um 10 --output outputs/demo_z10
python -m simcalc_zebrafish_pipeline.generate_volume --profile demo --seed 42 \
  --n-planes 9 --z-step-um 5 --output outputs/demo_z5
```

Обе команды снимают одну сцену 128 × 128 px с глубины 0 мкм. Глубины
0, 10, 20, 30 и 40 мкм общие для обоих стеков. Укажите новый каталог
для каждого запуска. Генерация работает на CPU.

| Профиль | Форма по умолчанию (ZYX) | Доступные глубины | Источник ткани |
| --- | --- | --- | --- |
| `demo` | `(5, 128, 128)` | 0–40 мкм | Процедурная |
| `nominal` | `(31, 288, 1728)` | 0–300 мкм | Процедурная |
| `real18-22` | `(5, 1376, 320)` | 0–40 мкм | Медианы TIFF 18–22 |

По умолчанию CLI использует `nominal`, шаг Z 10 мкм и пиксель XY
0,576 × 0,576 мкм. Параметры `--n-planes`, `--z-start-um` и
`--z-step-um` переопределяют сетку съёмки. Все запрошенные глубины должны
входить в диапазон профиля. В Python можно вызвать `sample_scene(profile, seed)`
один раз и передавать эту сцену в `render_stack(scene, acquisition)`
с разными параметрами съёмки.

## Результаты

| Файл | Содержимое |
| --- | --- |
| `image.ome.tif` | Стек uint16 для просмотра, размеры пикселя и позиции плейнов в OME-XML |
| `clean_adu.npy` | Чистый сигнал float32 до округления в uint16, форма ZYX |
| `instances.npz` | Каталог сом `cells`, полные разреженные маски, `shape_zyx`, `z_positions_um`, `metadata_json` и пики изолированного сигнала `cell_peak_adu` до сборки слоя |

Для строки каталога `i` возьмите диапазон
`instance_voxel_index[instance_offsets[i]:instance_offsets[i+1]]`.
Каждый индекс кодирует `(z * Y + y) * X + x`. Повтор одного индекса в
диапазонах разных сом означает перекрытие. Центр пикселя `(y, x)` находится
в `((y + 0.5) * dy, (x + 0.5) * dx)` от края изображения. Если TIFF-ридер
убрал одиночную ось Z, восстановите форму из `shape_zyx`.

Проверка читает результаты с диска и сравнивает каталог, маски и сигнал
на общих глубинах двух съёмок.

```bash
python -m unittest discover -s tests -p test_volume.py
```

## Сравнение с реальными плейнами 18–22

Для этого профиля нужны `data/Plane_Aligned_18.tiff` … `Plane_Aligned_22.tiff`.
Эталон сохраняет точные временные медианы, карты ткани и исходные полигоны
без поворота или изменения размера TIFF.

```bash
python -m simcalc_zebrafish_pipeline.volume_reference build \
  --annotations annotations/Plane20_median_registered_uint16_frame0_776e568f5575.json \
  --output outputs/volume_reference

python -m simcalc_zebrafish_pipeline.generate_volume --profile real18-22 \
  --reference outputs/volume_reference/reference.npz --seed 42 \
  --output outputs/volume_plane20_legacy_shell

python -m simcalc_zebrafish_pipeline.volume_reference compare \
  --reference outputs/volume_reference/reference.npz \
  --synthetic outputs/volume_plane20_legacy_shell \
  --output outputs/volume_comparison_legacy_shell
```

Сравнение пишет два PNG с XY/XZ и `comparison.json`. Плейн 18 принят
за 0 мкм. Масштаб 0,576 мкм/px по обеим осям — предположение о сохранении
шага после экспорта. TIFF не содержит физического масштаба. Измеренные
значения можно передать при построении эталона через `--dx-um`, `--dy-um`
и `--scale-source`.

Профиль `real18-22` использует исходную `shell_patch` Plane20 для XY и
оптический спад яркости по Z. `--emission-model ellipsoid` включает
излучение объёмной оболочки. `--median-style structured` перераспределяет
часть света в диффузный фон и применяется по умолчанию.
`--median-style baseline` даёт второй режим медианного рендера.

Высота сом, объёмная плотность и осевая оптика пока заданы моделью.
Совпадение двух съёмок на одной глубине проверяет воспроизводимость.
Сходство соседних слоёв оценивается отдельно по реальным TIFF.

## Документация

- [План и исходные измерения для 3D-генератора](docs/plan_3d_generator.md)
- [2D-генерация Plane16, разметка и эксперименты Plane20](docs/2d_workflows.md)
- [Обучение и форматы данных Plane20](training/plane20/README.md)
