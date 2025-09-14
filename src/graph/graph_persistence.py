import pandas as pd
import json
import os
import uuid
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional


class GraphPersistenceManager:
    def __init__(self, storage_path: str = "/tmp/graph_storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.versions = {}  # In-memory storage for simplicity

    def save_graph_snapshot(self, correlation_matrix: pd.DataFrame,
                          metadata: Dict[str, Any],
                          version_name: str) -> str:
        """Save a graph snapshot and return version ID"""
        version_id = str(uuid.uuid4())

        # Store in memory for simplicity
        self.versions[version_id] = {
            'correlation_matrix': correlation_matrix,
            'metadata': metadata,
            'version_name': version_name,
            'created_at': datetime.now()
        }

        return version_id

    def load_graph_snapshot(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Load a graph snapshot by version ID"""
        if version_id not in self.versions:
            return None

        return {
            'correlation_matrix': self.versions[version_id]['correlation_matrix'],
            'metadata': self.versions[version_id]['metadata']
        }

    def get_version_history(self) -> List[Dict[str, Any]]:
        """Get list of all versions with metadata"""
        history = []
        for version_id, data in self.versions.items():
            history.append({
                'version_id': version_id,
                'version_name': data['version_name'],
                'created_at': data['created_at'],
                'metadata': data['metadata']
            })
        return history

    def compare_versions(self, version_1: str, version_2: str) -> Optional[Dict[str, Any]]:
        """Compare two graph versions"""
        if version_1 not in self.versions or version_2 not in self.versions:
            return None

        matrix_1 = self.versions[version_1]['correlation_matrix']
        matrix_2 = self.versions[version_2]['correlation_matrix']

        # Simple comparison - calculate differences
        diff_matrix = matrix_1 - matrix_2

        return {
            'changes': 'Different correlations detected',
            'edge_differences': diff_matrix.abs().sum().sum(),
            'statistics': {
                'max_change': diff_matrix.abs().max().max(),
                'mean_change': diff_matrix.abs().mean().mean()
            }
        }

    def export_graph(self, version_id: str, export_path: str) -> bool:
        """Export graph data to file"""
        if version_id not in self.versions:
            return False

        data = self.versions[version_id]

        # Convert data to JSON serializable format
        export_data = {
            'version_id': version_id,
            'version_name': data['version_name'],
            'correlation_matrix': data['correlation_matrix'].to_dict(),
            'metadata': data['metadata'],
            'created_at': data['created_at'].isoformat()
        }

        try:
            with open(export_path, 'w') as f:
                json.dump(export_data, f, default=str)
            return True
        except Exception:
            return False

    def import_graph(self, import_path: str, version_name: str) -> Optional[str]:
        """Import graph data from file"""
        if not os.path.exists(import_path):
            return None

        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)

            # Recreate correlation matrix
            correlation_matrix = pd.DataFrame.from_dict(import_data['correlation_matrix'])
            metadata = import_data['metadata']

            # Save as new version
            return self.save_graph_snapshot(
                correlation_matrix=correlation_matrix,
                metadata=metadata,
                version_name=version_name
            )
        except Exception:
            return None

    def cleanup_old_versions(self, days_to_keep: int,
                           keep_tagged_versions: bool = True) -> Dict[str, Any]:
        """Clean up old versions"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        deleted_count = 0

        versions_to_delete = []
        for version_id, data in self.versions.items():
            # Check if version has timestamp in metadata, otherwise use created_at
            version_date = data['metadata'].get('timestamp', data['created_at'])
            if isinstance(version_date, str):
                # Convert string to datetime if needed
                version_date = datetime.fromisoformat(version_date)

            if version_date < cutoff_date:
                versions_to_delete.append(version_id)

        for version_id in versions_to_delete:
            del self.versions[version_id]
            deleted_count += 1

        return {'deleted_count': deleted_count}

    def create_backup(self, backup_path: str,
                     include_versions: List[str] = None) -> bool:
        """Create backup of graph data"""
        try:
            # Create temporary directory for backup data
            temp_dir = "/tmp/backup_temp"
            os.makedirs(temp_dir, exist_ok=True)

            # Export specified versions
            if include_versions is None:
                include_versions = list(self.versions.keys())

            for version_id in include_versions:
                if version_id in self.versions:
                    export_path = os.path.join(temp_dir, f"{version_id}.json")
                    self.export_graph(version_id, export_path)

            # Create zip file
            with zipfile.ZipFile(backup_path, 'w') as zipf:
                for file_path in os.listdir(temp_dir):
                    zipf.write(os.path.join(temp_dir, file_path), file_path)

            # Clean up temp directory
            shutil.rmtree(temp_dir)
            return True
        except Exception:
            return False

    def restore_from_backup(self, backup_path: str) -> Optional[Dict[str, Any]]:
        """Restore graph data from backup"""
        if not os.path.exists(backup_path):
            return None

        try:
            # Extract backup
            temp_dir = "/tmp/restore_temp"
            os.makedirs(temp_dir, exist_ok=True)

            with zipfile.ZipFile(backup_path, 'r') as zipf:
                zipf.extractall(temp_dir)

            # Import all files
            restored_versions = []
            for file_name in os.listdir(temp_dir):
                if file_name.endswith('.json'):
                    file_path = os.path.join(temp_dir, file_name)
                    version_id = self.import_graph(file_path, f"restored_{file_name}")
                    if version_id:
                        restored_versions.append(version_id)

            # Clean up temp directory
            shutil.rmtree(temp_dir)

            return {'restored_versions': restored_versions}
        except Exception:
            return None

    def delete_version(self, version_id: str) -> bool:
        """Delete a specific version"""
        if version_id in self.versions:
            del self.versions[version_id]
            return True
        return False

    def validate_graph_integrity(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Validate graph data integrity"""
        if version_id not in self.versions:
            return None

        data = self.versions[version_id]
        correlation_matrix = data['correlation_matrix']

        # Simple validation checks
        is_symmetric = correlation_matrix.equals(correlation_matrix.T)
        has_diagonal_ones = all(correlation_matrix.iloc[i, i] == 1.0
                               for i in range(len(correlation_matrix)))
        values_in_range = ((correlation_matrix.abs() <= 1.0).all().all())

        checks_passed = [
            ('is_symmetric', is_symmetric),
            ('diagonal_ones', has_diagonal_ones),
            ('values_in_range', values_in_range)
        ]

        is_valid = all(check[1] for check in checks_passed)

        return {
            'is_valid': is_valid,
            'checks_passed': checks_passed
        }