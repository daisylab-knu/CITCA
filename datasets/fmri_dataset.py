import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


class FMRIDataset(Dataset):
    def __init__(self, npy_path: str, text_path: str, threshold: float = 0.1):
        self.data = np.load(npy_path, allow_pickle=True)
        self.threshold = threshold
        self.num_rois = self.data[0]["node_feat"].shape[0]

        text_data = np.load(text_path, allow_pickle=True).item()
        if "text_embeddings" not in text_data:
            raise KeyError(f"'text_embeddings' not found in text cache: {text_path}")
        self.text_embeddings = torch.tensor(
            text_data["text_embeddings"], dtype=torch.float32
        )
        if self.text_embeddings.ndim != 3 or self.text_embeddings.shape[-1] != 768:
            raise ValueError(
                "text_embeddings must have shape [num_subjects, num_sentences, 768], "
                f"got {tuple(self.text_embeddings.shape)}"
            )

        if "sub_ids" in text_data:
            text_ids = [str(value) for value in text_data["sub_ids"]]
            graph_ids = [str(item.get("id", i)) for i, item in enumerate(self.data)]
            if len(graph_ids) != len(text_ids):
                raise ValueError(
                    f"Graph/Text sample count mismatch: {len(graph_ids)} vs {len(text_ids)}"
                )
            if graph_ids != text_ids:
                index_by_id = {subject_id: i for i, subject_id in enumerate(text_ids)}
                try:
                    order = [index_by_id[subject_id] for subject_id in graph_ids]
                except KeyError as error:
                    raise ValueError(
                        f"Missing subject id in text cache: {error.args[0]}"
                    ) from error
                self.text_embeddings = self.text_embeddings[order]

        if self.text_embeddings.shape[0] != len(self.data):
            raise ValueError(
                "text_embeddings subject count mismatch: "
                f"{self.text_embeddings.shape[0]} vs {len(self.data)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        corr_matrix = np.asarray(sample["node_feat"], dtype=np.float32)
        adjacency = corr_matrix.copy()
        np.fill_diagonal(adjacency, 0.0)
        adjacency[np.abs(adjacency) < self.threshold] = 0.0
        rows, columns = np.where(adjacency != 0.0)

        raw_label = int(sample["label"])
        label = 0 if raw_label == 2 else raw_label
        return Data(
            x=torch.tensor(corr_matrix, dtype=torch.float32),
            edge_index=torch.tensor(np.vstack([rows, columns]), dtype=torch.long),
            edge_weight=torch.tensor(
                np.abs(adjacency[rows, columns]), dtype=torch.float32
            ),
            y=torch.tensor(label, dtype=torch.long),
            text_tokens=self.text_embeddings[index],
        )


def get_full_dataset(npy_path: str, text_path: str, threshold: float = 0.1):
    return FMRIDataset(npy_path, text_path, threshold)
