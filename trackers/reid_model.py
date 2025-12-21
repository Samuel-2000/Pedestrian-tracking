"""
ReID (Re-Identification) Model pre DeepSORT

Pouziva natreneny OSNet model z torchreid knižnice.
OSNet je state-of-the-art ReID model optimalizovany pre person re-identification.

Referencie:
- DeepSORT paper: https://arxiv.org/abs/1703.07402
- OSNet paper: https://arxiv.org/abs/1905.00953
- torchreid: https://github.com/KaiyangZhou/deep-person-reid
"""

import torch
import cv2
import numpy as np

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("Warning: torchreid not installed. Install with: pip install torchreid")


class ReIDExtractor:
    def __init__(self, device='cpu', model_name='osnet_x1_0'):
        self.device = device
        self.model = None
        self.feat_dim = 512
        
        if TORCHREID_AVAILABLE:
            try:
                self.model = torchreid.models.build_model(
                    name=model_name,
                    num_classes=1,
                    pretrained=True
                )
                self.model.eval()
                
                if torch.cuda.is_available() and device == 'cuda':
                    self.model = self.model.cuda()
                    self.device = 'cuda'
                else:
                    self.device = 'cpu'
                
                print(f"Loaded pre-trained ReID model: {model_name}")
            except Exception as e:
                print(f"Error loading torchreid model: {e}")
                print("Falling back to simple feature extraction")
                self.model = None
        else:
            print("torchreid not available, using simple feature extraction")
    
    def preprocess(self, roi):
        if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
            return None
        
        roi_resized = cv2.resize(roi, (128, 256))
        roi_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        
        roi_norm = roi_rgb.astype(np.float32) / 255.0
        
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        roi_norm = (roi_norm - mean) / std
        
        roi_chw = np.transpose(roi_norm, (2, 0, 1))
        roi_batch = np.expand_dims(roi_chw, axis=0)
        
        tensor = torch.from_numpy(roi_batch).float()
        
        if self.device == 'cuda':
            tensor = tensor.cuda()
        
        return tensor
    
    def extract(self, roi):
        if self.model is None:
            return self._extract_simple_features(roi)
        
        preprocessed = self.preprocess(roi)
        if preprocessed is None:
            return self._extract_simple_features(roi)
        
        try:
            with torch.no_grad():
                features = self.model(preprocessed)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            features_np = features.cpu().numpy()[0]
            return features_np
        except Exception as e:
            return self._extract_simple_features(roi)
    
    def _extract_simple_features(self, roi):
        if roi.size == 0:
            return np.zeros(self.feat_dim, dtype=np.float32)
        
        h, w = roi.shape[:2]
        aspect_ratio = w / h if h > 0 else 1.0
        area = w * h
        
        features = np.zeros(self.feat_dim, dtype=np.float32)
        features[0] = w / 1000.0
        features[1] = h / 1000.0
        features[2] = aspect_ratio
        features[3] = area / 100000.0
        features[4:] = np.random.randn(self.feat_dim - 4).astype(np.float32) * 0.01
        
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features
