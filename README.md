# CITCA Traning

model 학습에 필요한 코드만 독립적으로 모은 구성입니다.
교차검증의 validation/test 평가는 정상적인 학습 및 최종 성능 확인을 위해 유지합니다.

## 포함 파일

```text
training_only/
├── main.py
├── setting.py
├── train.py
├── utils.py
├── requirements.txt
├── datasets/
│   └── fmri_dataset.py
└── models/
    └── citca.py
```

- `data_path`: 각 원소가 `node_feat`, `label`을 갖는 NumPy array
- `text_path`: `text_embeddings`가 `[subjects, sentences, 768]`인 NumPy array

## 실행

```bash
python -m pip install -r requirements.txt
python main.py \
  --gpu_ids 0 \
  --data_path /path/to/abide_aal116.npy \
  --text_path /path/to/text_its.npy \
  --output_dir ./output \
  --n_folds 5 \
  --epochs 100
```

각 fold의 best validation checkpoint와 전체 metric JSON이 `output_dir`에
저장됩니다. 실행 가능한 전체 옵션은 `python main.py --help`로 확인할 수 있습니다.
