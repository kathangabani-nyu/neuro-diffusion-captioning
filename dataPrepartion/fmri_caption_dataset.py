#!/usr/bin/env python3
"""
FmriCaptionDataset: Combines NSD fMRI data with Qwen2-VL generated captions.

This dataset:
1. Loads preprocessed fMRI data (from NSD or cache)
2. Loads corresponding Qwen2-VL captions
3. Implements image-disjoint train/val/test splits
4. Tokenizes captions using Qwen2-VL tokenizer
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import nibabel
import nilearn.signal
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

# #region agent log
import os
DEBUG_LOGGING_ENABLED = os.getenv("FMRI_DEBUG_LOGGING", "false").lower() == "true"
def _debug_log(location, message, data, hypothesis_id):
    if not DEBUG_LOGGING_ENABLED:
        return  # Disabled by default for performance
    try:
        log_path = '/scratch/kdg7224/fmri_caption_project/debug.log'
        with open(log_path, 'a') as f:
            import json, time
            f.write(json.dumps({"sessionId":"debug-session","runId":"run1","hypothesisId":hypothesis_id,"location":location,"message":message,"data":data,"timestamp":int(time.time()*1000)}) + '\n')
    except Exception as e:
        pass  # Silently fail to not break training
# #endregion

# Replicate DynaDiff's constants
TR_s = 4 / 3


class FmriCaptionDataset(Dataset):
    """
    Dataset combining fMRI brain signals with Qwen2-VL captions.
    
    Supports image-disjoint splits to ensure fair evaluation.
    """
    
    def __init__(
        self,
        nsddata_path: str,
        trial_mapping_path: str,
        captions_path: str,
        subjects: list = [1, 2, 5, 7],
        split: str = "train",  # train, val, test
        tokenizer_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        max_caption_length: int = 64,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
        offset: float = 4.6,
        duration: float = 8.0,
    ):
        """
        Args:
            nsddata_path: Path to NSD dataset root
            trial_mapping_path: Path to trial_mapping.json
            captions_path: Path to qwen_captions.json
            subjects: List of subject IDs to include
            split: Data split (train, val, test)
            tokenizer_name: HuggingFace tokenizer name
            max_caption_length: Maximum caption token length
            train_ratio, val_ratio, test_ratio: Split ratios
            seed: Random seed for reproducibility
            offset, duration: fMRI time window parameters
        """
        self.nsddata_path = Path(nsddata_path)
        self.subjects = subjects
        self.subject_to_idx = {s: i for i, s in enumerate(subjects)}
        self.split = split
        self.offset = offset
        self.duration = duration
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_caption_length = max_caption_length
        
        # Load trial mapping
        with open(trial_mapping_path, 'r') as f:
            self.trial_mapping = json.load(f)
        
        # Load captions
        with open(captions_path, 'r') as f:
            self.captions = json.load(f)
        
        # Create image-disjoint splits
        self.split_indices = self._create_image_disjoint_splits(
            train_ratio, val_ratio, test_ratio, seed
        )
        
        # Build samples list
        self.samples = []
        self._load_samples()
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
    
    def _create_image_disjoint_splits(self, train_ratio, val_ratio, test_ratio, seed):
        """Create image-disjoint splits based on COCO image IDs."""
        # Collect all unique COCO IDs across all subjects
        coco_to_trials = defaultdict(list)
        
        for subj_key, mapping in self.trial_mapping.items():
            for image_id, trial_info in mapping.items():
                coco_id = trial_info.get("coco_id")
                if coco_id is not None:
                    trial_key = f"{subj_key}_trial{int(image_id):04d}"
                    coco_to_trials[coco_id].append((subj_key, image_id, trial_key))
        
        # If no COCO IDs, fall back to NSD image IDs
        if len(coco_to_trials) == 0:
            print("Warning: No COCO IDs found, using NSD image IDs for splitting")
            for subj_key, mapping in self.trial_mapping.items():
                for image_id, trial_info in mapping.items():
                    nsd_image_id = trial_info.get("nsd_image_id")
                    if nsd_image_id is not None:
                        trial_key = f"{subj_key}_trial{int(image_id):04d}"
                        coco_to_trials[nsd_image_id].append((subj_key, image_id, trial_key))
        
        # Shuffle and split images (not trials)
        unique_images = list(coco_to_trials.keys())
        random.seed(seed)
        random.shuffle(unique_images)
        
        n_images = len(unique_images)
        n_train = int(n_images * train_ratio)
        n_val = int(n_images * val_ratio)
        
        train_images = set(unique_images[:n_train])
        val_images = set(unique_images[n_train:n_train + n_val])
        test_images = set(unique_images[n_train + n_val:])
        
        # Assign trials to splits
        splits = {"train": [], "val": [], "test": []}
        for image_id_key, trials in coco_to_trials.items():
            if image_id_key in train_images:
                splits["train"].extend([t[2] for t in trials])  # trial_key
            elif image_id_key in val_images:
                splits["val"].extend([t[2] for t in trials])
            else:
                splits["test"].extend([t[2] for t in trials])
        
        return splits
    
    def _load_samples(self):
        """Load all samples for the current split."""
        split_trials = set(self.split_indices[self.split])
        
        for subj_key, mapping in self.trial_mapping.items():
            subject_id = int(subj_key.replace("subj", ""))
            
            for image_id, trial_info in mapping.items():
                trial_key = f"{subj_key}_trial{int(image_id):04d}"
                
                # Only include if in current split
                if trial_key not in split_trials:
                    continue
                
                # Check if caption exists
                if trial_key not in self.captions:
                    continue
                
                self.samples.append({
                    "subject_id": subject_id,
                    "subject_idx": self.subject_to_idx[subject_id],
                    "image_id": int(image_id),
                    "trial_key": trial_key,
                    "trial_info": trial_info,
                    "caption": self.captions[trial_key],
                })
    
    def _load_fmri_data(self, trial_info, subject_id):
        """Load preprocessed fMRI data for a trial."""
        session = trial_info.get("session", 1)
        run = trial_info.get("run", 1)
        start_idx = trial_info.get("start_idx", 0)
        end_idx = trial_info.get("end_idx", 6)
        
        # Construct paths
        run_id = f"session{session:02d}_run{run:02d}"
        nifti_fp = (
            self.nsddata_path
            / f"nsddata_timeseries/ppdata/subj{subject_id:02d}/func1pt8mm/timeseries/timeseries_{run_id}.nii.gz"
        )
        roi_fp = (
            self.nsddata_path
            / f"nsddata/ppdata/subj{subject_id:02d}/func1pt8mm/roi/nsdgeneral.nii.gz"
        )
        
        if not nifti_fp.exists() or not roi_fp.exists():
            raise FileNotFoundError(f"Missing fMRI files for trial: {nifti_fp}")
        
        # Load and preprocess (same as DynaDiff)
        # Use mmap=False to avoid DataLoader worker issues with memory-mapped files
        nifti = nibabel.load(nifti_fp, mmap=False)
        nifti = nifti.slicer[..., :225]
        roi_np = nibabel.load(roi_fp, mmap=False).get_fdata()
        nifti_data = nifti.get_fdata()[roi_np > 0]
        
        # Ensure we have a regular numpy array (not memory-mapped)
        nifti_data = np.array(nifti_data, dtype=np.float32, copy=True)
        
        # z-score and detrend
        nifti_data = nifti_data.T  # time first
        shape = nifti_data.shape
        nifti_data = nilearn.signal.clean(
            nifti_data.reshape(shape[0], -1),
            detrend=True,
            high_pass=None,
            t_r=TR_s,
            standardize="zscore_sample",
        )
        nifti_data = nifti_data.reshape(shape).T
        
        # Extract time window and ensure contiguous array
        fmri = np.ascontiguousarray(nifti_data[..., start_idx:end_idx], dtype=np.float32)
        
        return fmri
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # #region agent log
        _debug_log(f"{__file__}:{226}", "__getitem__ entry", {"idx": idx}, "A")
        # #endregion
        
        sample = self.samples[idx]
        
        # Load fMRI data
        try:
            fmri = self._load_fmri_data(sample["trial_info"], sample["subject_id"])
            # #region agent log
            _debug_log(f"{__file__}:{232}", "fmri loaded", {"shape": fmri.shape, "dtype": str(fmri.dtype), "is_contiguous": fmri.flags['C_CONTIGUOUS'], "has_shared_memory": hasattr(fmri, 'base') and fmri.base is not None}, "A")
            # #endregion
        except Exception as e:
            print(f"Error loading fMRI for {sample['trial_key']}: {e}")
            # Return zeros as fallback
            fmri = np.zeros((15724, 6), dtype=np.float32)
        
        # Create independent numpy array copy
        fmri_copy = np.array(fmri, dtype=np.float32, copy=True)
        # #region agent log
        has_base = hasattr(fmri_copy, 'base') and fmri_copy.base is not None
        _debug_log(f"{__file__}:{257}", "after np.array copy", {"is_contiguous": bool(fmri_copy.flags['C_CONTIGUOUS']), "has_shared_memory": has_base, "base_id": id(fmri_copy.base) if has_base else None, "orig_id": id(fmri)}, "A")
        # #endregion
        
        # Tokenize caption
        caption = sample["caption"]
        encoded = self.tokenizer(
            caption,
            max_length=self.max_caption_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        # #region agent log
        _debug_log(f"{__file__}:{252}", "tokenizer output", {"input_ids_shape": encoded["input_ids"].shape, "input_ids_storage": str(encoded["input_ids"].storage()) if hasattr(encoded["input_ids"], 'storage') else None, "input_ids_is_contiguous": encoded["input_ids"].is_contiguous()}, "B")
        # #endregion
        
        # Prepare labels (shifted for next-token prediction)
        labels = encoded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Create brain tensor with explicit contiguous copy
        # Use torch.tensor() to ensure completely independent storage
        brain_tensor = torch.tensor(fmri_copy, dtype=torch.float32).contiguous()
        # #region agent log
        _debug_log(f"{__file__}:{281}", "brain tensor created", {"shape": brain_tensor.shape, "is_contiguous": brain_tensor.is_contiguous(), "storage_size": brain_tensor.storage().size() if hasattr(brain_tensor, 'storage') else None}, "A")
        # #endregion
        
        # Create tokenizer tensors with completely independent storage
        # Use clone().detach() as recommended by PyTorch to ensure no shared storage
        input_ids = encoded["input_ids"].squeeze(0).clone().detach().cpu().contiguous()
        attention_mask = encoded["attention_mask"].squeeze(0).clone().detach().cpu().contiguous()
        labels_tensor = labels.squeeze(0).clone().detach().cpu().contiguous()
        # #region agent log
        _debug_log(f"{__file__}:{293}", "tokenizer tensors prepared", {"input_ids_contiguous": input_ids.is_contiguous(), "attention_mask_contiguous": attention_mask.is_contiguous(), "labels_contiguous": labels_tensor.is_contiguous(), "all_cpu": all(t.device.type == 'cpu' for t in [input_ids, attention_mask, labels_tensor])}, "B")
        # #endregion
        
        result = {
            "brain": brain_tensor.cpu().contiguous(),  # Ensure CPU and contiguous
            "subject_id": torch.tensor(sample["subject_idx"], dtype=torch.long),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels_tensor,
            "coco_id": sample["trial_info"].get("coco_id"),
            "trial_key": sample["trial_key"],
            "caption": caption,  # Keep raw caption for evaluation
        }
        # #region agent log
        _debug_log(f"{__file__}:{281}", "__getitem__ exit", {"result_keys": list(result.keys()), "brain_contiguous": result["brain"].is_contiguous(), "brain_device": str(result["brain"].device)}, "E")
        # #endregion
        
        return result


def collate_fn_variable_fmri(batch):
    """
    Custom collate function to handle variable-length fMRI sequences.
    
    Different subjects have different numbers of voxels, so we need to pad
    the fMRI sequences to the maximum length in the batch.
    """
    # Separate all fields
    brains = [item["brain"] for item in batch]  # List of [num_voxels, 6] tensors
    subject_ids = torch.stack([item["subject_id"] for item in batch])
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_masks = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Track original voxel counts for each sample
    original_voxel_counts = torch.tensor([b.shape[0] for b in brains], dtype=torch.long)
    
    # Pad fMRI sequences to max length in batch
    # Each brain is [num_voxels, 6], we need to pad along the first dimension
    max_voxels = max(b.shape[0] for b in brains)
    
    # Pad each brain tensor to [max_voxels, 6]
    padded_brains = []
    for brain in brains:
        if brain.shape[0] < max_voxels:
            # Pad with zeros along the first dimension
            padding = torch.zeros(max_voxels - brain.shape[0], brain.shape[1], dtype=brain.dtype)
            padded_brain = torch.cat([brain, padding], dim=0)
        else:
            padded_brain = brain
        padded_brains.append(padded_brain)
    
    brains_batch = torch.stack(padded_brains)  # [batch_size, max_voxels, 6]
    
    return {
        "brain": brains_batch,
        "subject_id": subject_ids,
        "original_voxel_counts": original_voxel_counts,  # Track original sizes
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "coco_id": [item.get("coco_id") for item in batch],
        "trial_key": [item["trial_key"] for item in batch],
        "caption": [item["caption"] for item in batch],
    }

