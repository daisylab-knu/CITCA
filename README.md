# CITCA: Cross-Attention Between Intrinsic Neural Timescales and Functional Connectivity for Autism Spectrum Disorder Diagnosis

**Abstract** 기존의 휴지상태 기능적 자기공명영상 기반 자폐스펙트럼장애 진단 연구는 주로 기능적 연결성에 의존하여 정적 뇌 네트워크 특징을 학습해 왔으나, 뇌 신호의 시간적 특성과 동적 신경 활동 정보를 충분히 반영하지 못하는 한계가 있다. 본 연구는 이러한 한계를 보완하기 위해 기능적 연결성과 내재적 신경 시간척도를 교차 모달 어텐션으로 통합하는 CITCA 모델을 제안한다. 실험 결과, CITCA는 단일 정보 기반 모델 및 기존 융합 방식보다 향상된 진단 성능을 달성하였으며, 정적 연결 정보와 시간적 신경 특성의 통합이 ASD 진단에 효과적임을 확인하였다.

## 포함 파일

```text
CITCA/
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
