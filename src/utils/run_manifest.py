"""
Run Manifest Generation for Reproducibility

Implements Priority 2.2: Creates a run_manifest.json for every experiment containing:
- Git commit hash
- Timestamp
- Full config (objective, power, feature set, calibration mode, split vs mono, seeds)
- Package versions
- Hardware info (CPU/RAM)
- Wall-clock runtime
- Dataset fingerprint

This makes every table/figure traceable to an exact run ID (audit-grade reproducibility).
"""

import json
import os
import sys
import platform
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


def get_git_commit_hash() -> Optional[str]:
    """Get the current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_branch() -> Optional[str]:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_status_dirty() -> bool:
    """Check if git working directory has uncommitted changes."""
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return False


def get_package_versions() -> Dict[str, str]:
    """Get versions of key packages."""
    packages = [
        'numpy', 'pandas', 'scipy', 'scikit-learn',
        'lightgbm', 'xgboost', 'catboost', 'interpret',
        'torch', 'matplotlib', 'yaml', 'pyyaml'
    ]
    
    versions = {'python': sys.version.split()[0]}
    
    for pkg in packages:
        try:
            if pkg == 'yaml':
                import yaml
                versions['pyyaml'] = getattr(yaml, '__version__', 'unknown')
            else:
                mod = __import__(pkg)
                versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[pkg] = 'not_installed'
    
    return versions


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information."""
    info = {
        'platform': platform.platform(),
        'processor': platform.processor(),
        'machine': platform.machine(),
        'python_implementation': platform.python_implementation()
    }
    
    # CPU count
    try:
        info['cpu_count'] = os.cpu_count()
    except Exception:
        info['cpu_count'] = 'unknown'
    
    # Memory (if psutil available)
    try:
        import psutil
        mem = psutil.virtual_memory()
        info['total_memory_gb'] = round(mem.total / (1024**3), 1)
        info['available_memory_gb'] = round(mem.available / (1024**3), 1)
    except ImportError:
        info['total_memory_gb'] = 'psutil_not_installed'
    
    # GPU info (if torch available)
    try:
        import torch
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_device_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(torch.cuda.device_count())
            ]
        else:
            info['cuda_device_count'] = 0
            info['cuda_device_names'] = []
    except ImportError:
        info['cuda_available'] = 'torch_not_installed'
    
    return info


def compute_dataset_fingerprint(df: pd.DataFrame,
                                id_column: Optional[str] = None,
                                file_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Compute a fingerprint/hash of the dataset for reproducibility.
    
    Parameters
    ----------
    df : DataFrame
        The dataset
    id_column : str, optional
        Column to use for record IDs
    file_path : str, optional
        Path to the source file
        
    Returns
    -------
    dict
        Dataset fingerprint
    """
    fingerprint = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns)
    }
    
    # Compute hash of key statistics
    hash_data = []
    for col in df.select_dtypes(include=[np.number]).columns[:10]:  # First 10 numeric cols
        hash_data.append(f"{col}:{df[col].mean():.6f}:{df[col].std():.6f}")
    
    fingerprint['stats_hash'] = hashlib.md5(
        '|'.join(hash_data).encode()
    ).hexdigest()[:16]
    
    # Record IDs if provided
    if id_column and id_column in df.columns:
        ids = df[id_column].astype(str).tolist()
        fingerprint['id_hash'] = hashlib.md5(
            ','.join(sorted(ids)).encode()
        ).hexdigest()[:16]
    
    # File hash if path provided
    if file_path and os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()[:16]
            fingerprint['file_hash'] = file_hash
            fingerprint['file_path'] = str(file_path)
            fingerprint['file_size_mb'] = round(
                os.path.getsize(file_path) / (1024**2), 2
            )
        except Exception:
            pass
    
    return fingerprint


class RunManifest:
    """
    Creates and manages run manifests for experiment reproducibility.
    
    Usage
    -----
    manifest = RunManifest(config=config, output_dir='outputs/')
    manifest.start()  # Records start time
    # ... run experiment ...
    manifest.add_result('cv_metrics', cv_result.outer_metrics)
    manifest.add_result('predictions_path', 'outputs/predictions.csv')
    manifest.end()  # Records end time, saves manifest
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 output_dir: str,
                 experiment_name: Optional[str] = None,
                 dataset_df: Optional[pd.DataFrame] = None,
                 dataset_path: Optional[str] = None):
        """
        Initialize run manifest.
        
        Parameters
        ----------
        config : dict
            Full experiment configuration
        output_dir : str
            Output directory for manifest
        experiment_name : str, optional
            Name for this experiment
        dataset_df : DataFrame, optional
            Dataset for fingerprinting
        dataset_path : str, optional
            Path to dataset file
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.experiment_name = experiment_name or 'experiment'
        self.dataset_df = dataset_df
        self.dataset_path = dataset_path
        
        # Initialize manifest
        self.manifest = {
            'experiment_name': self.experiment_name,
            'created_at': datetime.now().isoformat(),
            'run_id': self._generate_run_id()
        }
        
        self.start_time = None
        self.end_time = None
        
    def _generate_run_id(self) -> str:
        """Generate a unique run ID."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_suffix = hashlib.md5(
            str(np.random.random()).encode()
        ).hexdigest()[:6]
        return f"{timestamp}_{random_suffix}"
    
    def start(self) -> 'RunManifest':
        """Record experiment start time."""
        import time
        self.start_time = time.time()
        self.manifest['started_at'] = datetime.now().isoformat()
        return self
    
    def end(self) -> 'RunManifest':
        """Record experiment end time and save manifest."""
        import time
        self.end_time = time.time()
        self.manifest['ended_at'] = datetime.now().isoformat()
        
        if self.start_time is not None:
            self.manifest['runtime_seconds'] = round(
                self.end_time - self.start_time, 2
            )
            self.manifest['runtime_minutes'] = round(
                (self.end_time - self.start_time) / 60, 2
            )
        
        self._save()
        return self
    
    def add_result(self, key: str, value: Any) -> 'RunManifest':
        """Add a result to the manifest."""
        if 'results' not in self.manifest:
            self.manifest['results'] = {}
        
        # Convert numpy types to Python types
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            value = float(value)
        elif isinstance(value, dict):
            value = self._sanitize_dict(value)
        
        self.manifest['results'][key] = value
        return self
    
    def _sanitize_dict(self, d: Dict) -> Dict:
        """Convert numpy types in dict to Python types."""
        result = {}
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result[k] = float(v)
            elif isinstance(v, dict):
                result[k] = self._sanitize_dict(v)
            elif isinstance(v, pd.DataFrame):
                result[k] = v.to_dict('records')
            else:
                result[k] = v
        return result
    
    def build(self) -> Dict[str, Any]:
        """Build the complete manifest."""
        # Git info
        self.manifest['git'] = {
            'commit_hash': get_git_commit_hash(),
            'branch': get_git_branch(),
            'dirty': get_git_status_dirty()
        }
        
        # Package versions
        self.manifest['package_versions'] = get_package_versions()
        
        # Hardware info
        self.manifest['hardware'] = get_hardware_info()
        
        # Config
        self.manifest['config'] = self._sanitize_dict(self.config)
        
        # Dataset fingerprint
        if self.dataset_df is not None:
            self.manifest['dataset'] = compute_dataset_fingerprint(
                self.dataset_df,
                id_column=None,
                file_path=self.dataset_path
            )
        
        return self.manifest
    
    def _save(self) -> Path:
        """Save manifest to JSON file."""
        # Ensure build is complete
        self.build()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save manifest
        manifest_path = self.output_dir / 'run_manifest.json'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        return manifest_path
    
    def save_to(self, path: str) -> Path:
        """Save manifest to a specific path."""
        self.build()
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.manifest, f, indent=2, default=str)
        
        return path


def create_run_manifest(config: Dict[str, Any],
                        output_dir: str,
                        dataset_df: Optional[pd.DataFrame] = None,
                        dataset_path: Optional[str] = None,
                        experiment_name: Optional[str] = None) -> RunManifest:
    """
    Factory function to create a run manifest.
    
    Parameters
    ----------
    config : dict
        Experiment configuration
    output_dir : str
        Output directory
    dataset_df : DataFrame, optional
        Dataset for fingerprinting
    dataset_path : str, optional
        Path to dataset file
    experiment_name : str, optional
        Experiment name
        
    Returns
    -------
    RunManifest
        Initialized manifest (call .start() to begin)
    """
    return RunManifest(
        config=config,
        output_dir=output_dir,
        experiment_name=experiment_name,
        dataset_df=dataset_df,
        dataset_path=dataset_path
    )


def validate_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Validate a run manifest and check reproducibility.
    
    Parameters
    ----------
    manifest_path : str
        Path to manifest JSON
        
    Returns
    -------
    dict
        Validation results with warnings/errors
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    results = {
        'valid': True,
        'warnings': [],
        'errors': [],
        'manifest': manifest
    }
    
    # Check required fields
    required = ['run_id', 'config', 'git', 'package_versions']
    for field in required:
        if field not in manifest:
            results['errors'].append(f"Missing required field: {field}")
            results['valid'] = False
    
    # Check git status
    if manifest.get('git', {}).get('dirty', False):
        results['warnings'].append(
            "Git working directory was dirty (uncommitted changes)"
        )
    
    if manifest.get('git', {}).get('commit_hash') is None:
        results['warnings'].append("Git commit hash not recorded")
    
    # Check package versions
    pkg_vers = manifest.get('package_versions', {})
    critical_packages = ['numpy', 'pandas', 'lightgbm', 'scikit-learn']
    for pkg in critical_packages:
        if pkg not in pkg_vers or pkg_vers[pkg] == 'not_installed':
            results['warnings'].append(f"Critical package {pkg} version not recorded")
    
    return results
