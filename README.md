# BebraFish · 3D-синтетика сом

BebraFish генерирует синтетические изображения микроскопии и маски
отдельных сом зебрафиш. Одна 3D-сцена с постоянными ID переснимается
на разных глубинах. Геометрические маски строятся по сечению сом,
а рендер учитывает расфокусированный свет.

## Установка и быстрый запуск

Из корня репозитория, Python 3.10+. Рендер работает на CPU.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m simcalc_zebrafish_pipeline.generate_volume --profile demo --seed 42 \
  --n-planes 5 --z-step-um 10 --output outputs/demo_z10
python -m simcalc_zebrafish_pipeline.generate_volume --profile demo --seed 42 \
  --n-planes 9 --z-step-um 5 --output outputs/demo_z5
python -m unittest discover -s tests -p 'test_volume.py'
python -m unittest discover -s tests -p 'test_soma_growth.py'
```

Две команды снимают одну сцену 128 × 128 px с общего начала. На глубинах
0, 10, 20, 30 и 40 мкм совпадают каталог, маски и чистый сигнал. Для
каждого запуска укажите новый каталог результата.

## Синтетика по данным Zenodo Figure 5l,m

Распакуйте архив в `data/Figure 5l,m/`. Генератор читает исходные HDF5 с
временными средними и разметкой Cellpose. Первый прогон использует одну
рыбу. [Данные Zenodo](https://zenodo.org/records/19598932)

```bash
reference='data/Figure 5l,m/2026-02-10_10-12-11_fish001_setup0_arena0_HN_preprocessed_data.h5'

python -m simcalc_zebrafish_pipeline.generate_zenodo \
  --reference "$reference" --seed 42 --fill-contacts --smaller-somata \
  --split-largest-fraction 0.62 --output outputs/zenodo_fish001

python -m simcalc_zebrafish_pipeline.close_contact_gaps \
  --source outputs/zenodo_fish001 --output outputs/zenodo_closed_gaps

python -m simcalc_zebrafish_pipeline.render_zenodo_count_fit \
  --source outputs/zenodo_closed_gaps --refine-appearance \
  --output outputs/zenodo_final
```

- Первый шаг выращивает 3D-сомы с учётом соседей. Размеры, формы и
  пространственная плотность опираются на Cellpose. В плотных зонах
  сомы растут до общих границ, а в наиболее крупных ячейках добавляются
  центры.
- Второй шаг заполняет щели в плотных зонах. Свободные точки получают
  ID ближайшей сомы в пределах 3,5 мкм в 3D.
- Третий шаг приводит число сом к Cellpose на четырёх исходных
  глубинах. Он объединяет соприкасающиеся маски и убирает избыток
  сом в разреженных областях. Рендер использует фон из сглаженного
  реального кадра, объёмный свет от сом и локальную подгонку контраста
  контуров и соседних центров.

Результат для указанного файла — четыре кадра 799 × 799 px, XY
0,315 мкм/px, относительные глубины 5, 15, 25 и 35 мкм. Масштаб XY
читается из HDF5. В наборе также есть записи с шагом 0,270 мкм/px.

В `outputs/zenodo_final/` лежат `comparison_full.png` и
`comparison_crops.png`, итоговый `image.ome.tif` и `instances.npz`.
`closed_gaps.ome.tif` показывает отрисовку до подгонки числа сом.
`summary.json` содержит параметры рендера и число масок на каждом плейне.
Параметр `--workers 3` ускоряет оптический рендер на Linux.

### Пересъёмка масок

Передайте ту же сцену и сохранённые правила заполнения и объединения.
Число сом калибруется на исходных четырёх глубинах. Между ними маски
получаются из той же объёмной модели.

```bash
python -m simcalc_zebrafish_pipeline.generate_growth_masks reslice \
  --scene outputs/zenodo_fish001/scene.npz \
  --dense-gap-fill outputs/zenodo_final/dense_gap_fill.npz \
  --soma-id-map outputs/zenodo_final/soma_id_map.npz \
  --n-planes 7 --z-start-um 5 --z-step-um 5 --pixel-um 0.315 \
  --output outputs/zenodo_masks_z5.npz
```

Координаты массивов идут в порядке ZYX. Строке каталога сомы `i`
соответствует диапазон `instance_offsets[i]:instance_offsets[i+1]` в
`instance_voxel_index`. Плоский индекс равен `(z * Y + y) * X + x`.
OME-TIFF хранит физический масштаб пикселей и позиции плейнов.

## Остальные сценарии

- [3D Plane20, рост сом и эксперименты с засветкой](docs/3d_plane20.md)
- [План 3D-генератора и исходные измерения](docs/plan_3d_generator.md)
- [2D-генерация Plane16, разметка и эксперименты Plane20](docs/2d_workflows.md)
- [Обучение и форматы данных Plane20](training/plane20/README.md)

Локальные исходники и результаты сохраняются в `data/` и `outputs/`.
Обе папки исключены из Git.
