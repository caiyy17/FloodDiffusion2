import os
import random
from typing import Dict, List

import numpy as np
import torch
from lightning.pytorch.utilities import rank_zero_info
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.initialize import instantiate


class LengthMismatchError(Exception):
    pass


class GenerateDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.num_samples = cfg.data.num_samples
        self.dim = cfg.data.dim
        self.token_dim = cfg.data.token_dim
        if self.split == "train":
            self.dataset = []
        elif self.split == "val":
            self.dataset = []
        elif self.split == "test":
            self.num_samples = cfg.data.num_samples
            self.generate_text()
            self.dataset = self._load_file_list()
        rank_zero_info(f"Loaded {len(self.dataset)} samples from {split} dataset.")

    def _load_file_list(self) -> List[str]:
        """Load a list of npy file paths from a text file."""
        dataset = []
        for i in range(self.num_samples):
            data = {}
            data["name"] = f"sample_{i}"
            data["dataset"] = "generate"
            data["text"] = self.pool_text[i]
            text_length = self.pool_length[i]
            text_end = [sum(text_length[:j+1]) for j in range(len(text_length))]
            data["feature_text_end"] = [int(t * self.cfg.data.feature_fps + 0.5) for t in text_end]
            data["token_text_end"] = [int(t * self.cfg.data.token_fps + 0.5) for t in text_end]
            data["feature_length"] = data["feature_text_end"][-1]
            data["token_length"] = data["token_text_end"][-1]
            data["feature"] = np.zeros((data["feature_length"], self.dim))
            data["token"] = np.zeros((data["token_length"], self.token_dim))
            dataset.append(data)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return data

    def generate_text(self):
        self.pool_text = [
            # 0: Basketball defensive drill expanded to about 1 minute
            [
                "get into a low defensive stance.",
                "shuffle to the left.",
                "shuffle to the right.",
                "run forward a few steps.",
                "jump block.",
                "get into a defensive stance.",
                "walk backward.",
                "take a short sprint forward.",
                "shot a basketball.",
            ],
            # Basic Locomotion (Walking/Running variants)
            ["walk forward.", "turn left.", "walk forward."],
            ["walk forward.", "turn right.", "walk forward."],
            ["run forward.", "stop running.", "walk."],
            ["walk backwards.", "turn around.", "walk forward."],
            ["jog forward.", "jump.", "jog forward."],
            ["walk in circle.", "sit down.", "stand up."],
            ["crawl forward.", "stand up.", "walk."],
            ["tiptoe forward.", "stand up.", "look around."],
            ["walk sideways.", "turn around.", "walk forward."],
            ["march forward.", "salute.", "march forward."],
            # Fitness & Exercises (Whole body)
            ["do pushups.", "stand up.", "jumping jacks."],
            ["do squats.", "stand straight.", "kick."],
            ["lunges.", "switch legs.", "stand straight."],
            ["plank.", "stand up.", "rest."],
            ["yoga pose.", "breathe deep.", "relax."],
            ["burpees.", "high knees.", "stand straight."],
            ["sit ups.", "lie down.", "stand up."],
            ["stretch arms.", "touch toes.", "stand straight."],
            ["side plank.", "switch side.", "stand up."],
            ["mountain climber.", "stand up.", "jump."],
            # Dance & Artistic (Big movements)
            ["dance ballet.", "spin around.", "pose."],
            ["breakdance.", "freeze.", "stand up."],
            ["waltz step.", "twirl.", "bow."],
            ["hip hop dance.", "spin.", "pose."],
            ["salsa step.", "turn around.", "pose."],
            ["contemporary dance.", "leap.", "turn around."],
            ["disco move.", "raise arm.", "spin."],
            ["robot dance.", "freeze.", "move arms."],
            ["moonwalk.", "spin.", "pose."],
            ["tap dance.", "bow.", "stand straight."],
            # Combat & Martial Arts (No weapons)
            ["punch forward.", "dodge.", "kick."],
            ["block attack.", "counter punch.", "step back."],
            ["karate chop.", "kick high.", "bow."],
            ["boxing stance.", "jab.", "uppercut."],
            ["kickboxing.", "knee strike.", "step back."],
            ["shadow box.", "hook.", "dodge."],
            ["wrestling stance.", "tackle.", "stand straight."],
            ["capoeira move.", "spin kick.", "pose."],
            ["tai chi.", "slow push.", "relax."],
            ["judo throw.", "stand up.", "bow."],
            # Daily/Casual (Whole body only - Removed uncommon sleep/roll actions)
            ["sit on chair.", "cross legs.", "stand up."],
            ["kneel down.", "pray.", "stand up."],
            ["lean against wall.", "stand still.", "walk away."],
            ["squat down.", "look at ground.", "stand up."],
            ["jump for joy.", "clap hands.", "laugh."],
            ["wave hello.", "bow deep.", "walk away."],
            ["shrug shoulders.", "shake head.", "turn around."],
            ["stumble.", "regain balance.", "walk on."],
            ["look around.", "scratch head.", "stand still."],
            ["pace back and forth.", "stand still.", "look up."],
            # Sports (No equipment/ball interactions)
            ["swim stroke.", "turn head.", "swim."],
            ["climb ladder.", "look down.", "climb up."],
            ["run fast.", "long jump.", "celebrate."],
            ["run fast.", "high jump.", "celebrate."],
            ["sprinter start.", "run fast.", "stop running."],
            ["skating motion.", "spin.", "pose."],
            ["skiing motion.", "turn.", "stop skiing."],
            ["gymnastics roll.", "stand up.", "pose."],
            ["hurdle jump.", "run.", "stop running."],
            ["basketball defense.", "shuffle.", "jump block."],
            # Single Short Instructions (20 samples)
            ["walk forward."],
            ["run fast."],
            ["jump."],
            ["sit down."],
            ["stand up."],
            ["turn around."],
            ["bow."],
            ["wave."],
            ["clap hands."],
            ["kick."],
            ["punch."],
            ["squat."],
            ["spin."],
            ["dance."],
            ["jog."],
            ["march."],
            ["crawl."],
            ["salute."],
            ["shrug."],
            ["pose."],
            # Single Long/Complex Instructions (20 samples)
            ["perform a complex ballet routine with multiple spins and leaps."],
            ["do a vigorous high intensity interval training workout."],
            ["act out a scene of searching for something lost on the ground."],
            ["demonstrate a series of powerful karate kicks and punches."],
            ["perform a fluid tai chi sequence with slow deliberate movements."],
            ["dance energetically to an upbeat hip hop song."],
            ["mime climbing a steep and difficult mountain face."],
            ["act like a robot malfunctioning and shutting down."],
            ["perform a contemporary dance expressing deep sorrow."],
            ["move like a zombie walking slowly and stumbling."],
            ["shadow box against an imaginary opponent with fast combos."],
            ["practice a yoga flow moving from downward dog to cobra."],
            ["act as if walking through a strong wind storm."],
            ["perform a celebratory touchdown dance."],
            ["mime playing a drum solo with high energy."],
            ["act like sneaking carefully through a laser grid."],
            ["perform a traditional folk dance with skipping steps."],
            ["act out being extremely cold and shivering while walking."],
            ["demonstrate a warm up routine with stretching and jogging."],
            ["perform a breakdance sequence with floor work and freezes."],
        ]

        self.pool_length = [
            # 0: Basketball defensive drill sequence (sum ≈ 300)
            [3, 4, 4, 4, 4, 3, 4, 5, 4],
            # Basic Locomotion
            [4, 3, 4],
            [4, 3, 4],
            [3, 2, 4],
            [4, 2, 4],
            [4, 2, 4],
            [6, 2, 4],
            [5, 3, 4],
            [4, 2, 3],
            [4, 2, 4],
            [4, 2, 3],
            # Fitness
            [6, 3, 6],
            [5, 3, 3],
            [5, 5, 2],
            [8, 3, 4],
            [6, 4, 4],
            [5, 3, 5],
            [6, 4, 3],
            [4, 4, 3],
            [5, 5, 3],
            [6, 3, 3],
            # Dance
            [6, 4, 3],
            [6, 3, 4],
            [5, 4, 3],
            [5, 4, 3],
            [5, 3, 3],
            [6, 3, 5],
            [4, 3, 4],
            [5, 3, 4],
            [5, 3, 3],
            [4, 2, 3],
            # Combat
            [3, 3, 3],
            [3, 3, 3],
            [4, 4, 3],
            [4, 3, 3],
            [4, 4, 3],
            [4, 3, 3],
            [4, 4, 4],
            [5, 4, 3],
            [6, 6, 4],
            [4, 4, 4],
            # Daily/Casual
            [4, 6, 3],
            [4, 4, 8],
            [4, 5, 4],
            [4, 6, 4],
            [4, 6, 4],
            [4, 4, 3],
            [3, 3, 3],
            [3, 4, 4],
            [3, 3, 4],
            [3, 3, 4],
            # Sports (No equipment)
            [5, 3, 5],
            [5, 3, 5],
            [4, 3, 4],
            [4, 4, 4],
            [3, 6, 3],
            [5, 4, 3],
            [5, 4, 3],
            [4, 4, 3],
            [4, 4, 3],
            [5, 5, 3],
            # Single Short Instructions (20 samples)
            [6],
            [4],
            [2],
            [5],
            [4],
            [3],
            [3],
            [2],
            [2],
            [2],
            [2],
            [4],
            [3],
            [6],
            [5],
            [4],
            [5],
            [3],
            [2],
            [2],
            # Single Long Instructions (20 samples) - longer duration for complex actions
            [12],
            [12],
            [10],
            [10],
            [12],
            [10],
            [10],
            [8],
            [12],
            [10],
            [10],
            [12],
            [8],
            [8],
            [8],
            [10],
            [10],
            [8],
            [10],
            [12],
        ]


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None

    output = {}
    keys = batch[0].keys()

    for key in keys:
        if key in ["feature", "token"]:
            # Pad sequences
            items = [
                torch.from_numpy(b[key]) if isinstance(b[key], np.ndarray) else b[key]
                for b in batch
            ]
            output[key] = torch.nn.utils.rnn.pad_sequence(
                items, batch_first=True, padding_value=0
            )
        elif key in ["feature_length", "token_length"]:
            # Stack scalars
            output[key] = torch.tensor([b[key] for b in batch])
        else:
            # Default to list
            output[key] = [b[key] for b in batch]
    return output


if __name__ == "__main__":
    cfg = OmegaConf.load("configs/default.yaml")
    dataset = GenerateDataset(cfg, split="val")
    print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.data.train_bs, shuffle=True, collate_fn=collate_fn
    )
    for idx, data in tqdm(enumerate(dataloader)):
        pass
