#!/usr/bin/env python3
"""
Download Enterprise RAG Challenge data from GitHub repository.
Downloads PDFs, questions.json, and dataset.csv with progress tracking.
"""

import os
import hashlib
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json
import time

class ChallengeDataDownloader:
    """Download and manage Enterprise RAG Challenge data."""
    
    def __init__(self, project_root: Path = None):
        """Initialize downloader with project paths."""
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = project_root
        self.base_url = "https://raw.githubusercontent.com/trustbit/enterprise-rag-challenge/main/round1"
        self.pdfs_dir = project_root / "data" / "enterprise" / "pdfs"
        self.data_dir = project_root / "data" / "enterprise"
        
        # Ensure directories exist
        os.makedirs(self.pdfs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def get_pdf_list(self) -> List[Dict[str, Any]]:
        """Get list of PDFs to download from GitHub API."""
        api_url = "https://api.github.com/repos/trustbit/enterprise-rag-challenge/contents/round1/pdfs"
        
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()
            
            pdf_files = []
            for file_info in files:
                if file_info['name'].endswith('.pdf'):
                    pdf_files.append({
                        'name': file_info['name'],
                        'sha1': file_info['name'].split('.')[0],  # Filename is the SHA1
                        'download_url': file_info['download_url'],
                        'size': file_info['size']
                    })
            
            print(f"Found {len(pdf_files)} PDF files to download")
            return pdf_files
            
        except requests.RequestException as e:
            print(f"Error fetching PDF list: {e}")
            return []
    
    def verify_file_integrity(self, file_path: Path, expected_sha1: str) -> bool:
        """Verify downloaded file integrity using SHA1 hash."""
        if not file_path.exists():
            return False
            
        # Calculate SHA1 of downloaded file
        sha1_hash = hashlib.sha1()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha1_hash.update(chunk)
        
        actual_sha1 = sha1_hash.hexdigest()
        return actual_sha1 == expected_sha1
    
    def download_single_pdf(self, pdf_info: Dict[str, Any]) -> Dict[str, Any]:
        """Download a single PDF file."""
        file_path = self.pdfs_dir / pdf_info['name']
        
        # Check if file already exists and is valid
        if file_path.exists():
            if self.verify_file_integrity(file_path, pdf_info['sha1']):
                return {
                    'name': pdf_info['name'],
                    'status': 'already_exists',
                    'path': str(file_path),
                    'size': pdf_info['size']
                }
            else:
                print(f"Existing file {pdf_info['name']} has wrong hash, re-downloading...")
        
        try:
            # Download with progress
            response = requests.get(pdf_info['download_url'], stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    downloaded += len(chunk)
            
            # Verify integrity
            if self.verify_file_integrity(file_path, pdf_info['sha1']):
                return {
                    'name': pdf_info['name'],
                    'status': 'downloaded',
                    'path': str(file_path),
                    'size': pdf_info['size']
                }
            else:
                file_path.unlink()  # Delete corrupted file
                return {
                    'name': pdf_info['name'],
                    'status': 'integrity_failed',
                    'error': 'SHA1 hash mismatch'
                }
                
        except requests.RequestException as e:
            return {
                'name': pdf_info['name'],
                'status': 'download_failed',
                'error': str(e)
            }
    
    def download_pdfs_parallel(self, max_workers: int = 4) -> List[Dict[str, Any]]:
        """Download all PDFs in parallel."""
        pdf_files = self.get_pdf_list()
        if not pdf_files:
            return []
        
        print(f"Starting download of {len(pdf_files)} PDFs with {max_workers} workers...")
        
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_pdf = {
                executor.submit(self.download_single_pdf, pdf_info): pdf_info 
                for pdf_info in pdf_files
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_pdf):
                pdf_info = future_to_pdf[future]
                try:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    
                    status = result['status']
                    if status == 'downloaded':
                        print(f"[{completed}/{len(pdf_files)}] OK Downloaded: {result['name']}")
                    elif status == 'already_exists':
                        print(f"[{completed}/{len(pdf_files)}] OK Exists: {result['name']}")
                    else:
                        print(f"[{completed}/{len(pdf_files)}] FAILED: {result['name']} ({status})")
                        
                except Exception as e:
                    print(f"[{completed+1}/{len(pdf_files)}] ERROR: {pdf_info['name']} - {e}")
                    results.append({
                        'name': pdf_info['name'],
                        'status': 'exception',
                        'error': str(e)
                    })
                    completed += 1
        
        return results
    
    def download_supporting_files(self) -> Dict[str, Any]:
        """Download questions.json and dataset.csv."""
        supporting_files = {
            'questions.json': f"{self.base_url}/questions.json",
            'dataset.csv': f"{self.base_url}/dataset.csv"
        }
        
        results = {}
        for filename, url in supporting_files.items():
            file_path = self.data_dir / filename
            
            try:
                if file_path.exists():
                    print(f"OK {filename} already exists")
                    results[filename] = {'status': 'already_exists', 'path': str(file_path)}
                else:
                    response = requests.get(url)
                    response.raise_for_status()
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    print(f"OK Downloaded: {filename}")
                    results[filename] = {'status': 'downloaded', 'path': str(file_path)}
                    
            except requests.RequestException as e:
                print(f"FAILED to download {filename}: {e}")
                results[filename] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def generate_download_report(self, pdf_results: List[Dict], support_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive download report."""
        successful_pdfs = [r for r in pdf_results if r['status'] in ['downloaded', 'already_exists']]
        failed_pdfs = [r for r in pdf_results if r['status'] not in ['downloaded', 'already_exists']]
        
        total_size = sum(r.get('size', 0) for r in successful_pdfs)
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pdfs': {
                'total': len(pdf_results),
                'successful': len(successful_pdfs),
                'failed': len(failed_pdfs),
                'total_size_mb': round(total_size / (1024 * 1024), 2)
            },
            'supporting_files': support_results,
            'failed_downloads': failed_pdfs if failed_pdfs else []
        }
        
        return report


def main():
    """Main execution function."""
    print("Enterprise RAG Challenge Data Downloader")
    print("=" * 50)
    
    downloader = ChallengeDataDownloader()
    
    # Download PDFs
    print("\n1. Downloading PDF files...")
    pdf_results = downloader.download_pdfs_parallel(max_workers=6)
    
    # Download supporting files
    print("\n2. Downloading supporting files...")
    support_results = downloader.download_supporting_files()
    
    # Generate report
    report = downloader.generate_download_report(pdf_results, support_results)
    
    # Save report
    report_path = downloader.data_dir / "download_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"PDFs: {report['pdfs']['successful']}/{report['pdfs']['total']} successful")
    print(f"Total size: {report['pdfs']['total_size_mb']} MB")
    print(f"Supporting files: {len([r for r in support_results.values() if r['status'] in ['downloaded', 'already_exists']])}/2 successful")
    print(f"Report saved to: {report_path}")
    
    if report['failed_downloads']:
        print(f"\nWARNING: {len(report['failed_downloads'])} downloads failed - check report for details")
    else:
        print("\nSUCCESS: All downloads completed successfully!")


if __name__ == "__main__":
    main()