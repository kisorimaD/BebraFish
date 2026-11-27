# 3D-синтетика Plane20

Все команды запускаются из корня репозитория после [установки зависимостей](../README.md#установка-и-быстрый-запуск).

## Эллипсоиды и съёмка по глубине

| Профиль | Стек по умолчанию, ZYX | Доступные глубины | Ткань |
| --- | --- | --- | --- |
| `demo` | 5 × 128 × 128 | 0–40 мкм | Процедурная |
| `nominal` | 31 × 288 × 1728 | 0–300 мкм | Процедурная |
| `real18-22` | 5 × 1376 × 320 | 0–40 мкм | Медианы TIFF 18–22 |

`generate_volume` по умолчанию использует `nominal`, шаг Z 10 мкм и
XY 0,576 мкм/px. Опции `--n-planes`, `--z-start-um` и `--z-step-um`
задают сетку съёмки. В Python `sample_scene(profile, seed)` создаёт
сцену, а `render_stack(scene, acquisition)` снимает её. Стек сохраняется
в `image.ome.tif`, чистый сигнал float32 — в `clean_adu.npy`, а каталог,
разреженные маски и `cell_peak_adu` — в `instances.npz`.

### Сравнение с TIFF 18–22

Подготовьте `data/Plane_Aligned_18.tiff` … `data/Plane_Aligned_22.tiff`.

```bash
python -m simcalc_zebrafish_pipeline.volume_reference build \
  --annotations annotations/Plane20_median_registered_uint16_frame0_776e568f5575.json \
  --output outputs/volume_reference

python -m simcalc_zebrafish_pipeline.generate_volume --profile real18-22 \
  --reference outputs/volume_reference/reference.npz --seed 42 \
  --output outputs/volume_plane20_structured

python -m simcalc_zebrafish_pipeline.volume_reference compare \
  --reference outputs/volume_reference/reference.npz \
  --synthetic outputs/volume_plane20_structured \
  --output outputs/volume_plane20_comparison
```

Эталон хранит временные медианы, карту ткани и контуры в исходной
сетке TIFF. Плейн 18 соответствует Z = 0 мкм. XY 0,576 мкм/px
принято по исходной съёмке. Измеренный экспортный масштаб задаётся
через `--dx-um`, `--dy-um` и `--scale-source` при построении эталона.

`real18-22` по умолчанию сочетает исходный XY-профиль оболочки Plane20
с осевым спадом и структурным фоном. Параметры `--emission-model`
и `--median-style` позволяют проверить другие варианты свечения и фона.

### Вклад сом вне сечения

```bash
python -m simcalc_zebrafish_pipeline.generate_volume --profile real18-22 \
  --reference outputs/volume_reference/reference.npz --seed 42 \
  --emission-model uniform-shell --median-style baseline \
  --output outputs/volume_uniform_shell
python -m tools.compare_soma_haze --reference outputs/volume_reference/reference.npz \
  --structured outputs/volume_plane20_structured --uniform outputs/volume_uniform_shell \
  --output outputs/soma_haze_pilot
```

Сравнение проверяет общую геометрию и разбивает свет на вклады сом
внутри и вне сечения. `comparison.png` показывает кропы на общей шкале,
`calibrated.ome.tif` содержит стек, `report.json` — измерения.
Высота сом и осевая PSF здесь заданы моделью.

## Рост сом при контакте

```bash
python -m simcalc_zebrafish_pipeline.generate_growth_masks demo \
  --output outputs/soma_growth_demo
python -m simcalc_zebrafish_pipeline.generate_growth_masks render \
  --scene outputs/soma_growth_demo/group/deform/scene.npz \
  --n-planes 5 --z-start-um 4.8 --z-step-um 1.6 --pixel-um 0.2 \
  --output outputs/group_fluorescence
```

Демо сравнивает одиночную сому, пару и плотную группу. Центры
сдвигаются при контакте, а общие плоскости ограничивают рост поверхностей.
`growth.png` показывает стадии, `packing_comparison.png` — XY- и
XZ-сечения и статистику зазоров. В `scene.npz` лежат постоянные ID и
замкнутые треугольные сетки всех стадий. Команда `reslice` сохраняет
маски без вычисления света.

### Полный Plane20

```bash
python -m simcalc_zebrafish_pipeline.generate_growth_field \
  --reference outputs/volume_reference/reference.npz \
  --output outputs/soma_growth_full_plane20
```

Модель строит локальный слой толщиной 18,432 мкм. Z = 9,216 мкм в нём
соответствует Plane20. `--density-per-um3` задаёт плотность кандидатов,
`--target-volume-scale` масштабирует целевые объёмы после сэмплирования
центров, `--geometry-only` сохраняет сцену и её маски.

Для отрисовки оболочек в стиле исходного Plane20 используйте
`render_growth_plane20` с `--scene`, `--profile`, `--z-um 9.216` и
`--output`. Для эксперимента с постоянным фоном и вкладом сом вне
сечения используйте `render_growth_offplane` с теми же аргументами.
Обе команды требуют экспортированный медианный профиль. Его экспорт
через `training.plane20.generate_median --export-profile` использует
локальные результаты калибровки, описанные в [руководстве Plane20](../training/plane20/README.md).

Инструмент `python -m tools.diagnose_plane20_packing --help` показывает
опции отчёта по форме контуров и зазорам между сомами.
