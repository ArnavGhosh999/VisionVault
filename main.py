#!/usr/bin/env python3
"""
Simple LLM Image Tagging System
Uses BLIP and LLaVA without complex dependencies
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

# Core libraries
import torch
import numpy as np
from PIL import Image, ExifTags

# Vision-Language Models
from transformers import BlipProcessor, BlipForConditionalGeneration

# Simple database (JSON-based)
import sqlite3

# Ollama for LLaVA
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageAnalyzer:
    """Handles image analysis using BLIP and LLaVA"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load BLIP model
        self.load_blip()
        
        # Initialize Ollama client
        self.ollama_client = ollama.Client()
        
    def load_blip(self):
        """Load BLIP model"""
        try:
            logger.info("Loading BLIP model...")
            self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            self.blip_model.to(self.device)
            logger.info("BLIP loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP: {e}")
            raise
    
    def analyze_with_blip(self, image: Image.Image) -> Dict[str, str]:
        """Analyze image using BLIP"""
        results = {}
        
        try:
            # Basic captioning
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=100)
            caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            results["caption"] = caption
            
            # Conditional captioning with different prompts
            prompts = ["a photo of", "this is", "the image shows"]
            for i, prompt in enumerate(prompts):
                try:
                    conditional_inputs = self.blip_processor(
                        image, text=prompt, return_tensors="pt"
                    ).to(self.device)
                    with torch.no_grad():
                        conditional_outputs = self.blip_model.generate(**conditional_inputs, max_length=100)
                    conditional_caption = self.blip_processor.decode(conditional_outputs[0], skip_special_tokens=True)
                    results[f"conditional_{i}"] = conditional_caption
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"BLIP analysis failed: {e}")
            results["error"] = str(e)
        
        return results
    
    def analyze_with_llava(self, image_path: str) -> Dict[str, str]:
        """Analyze image using LLaVA via Ollama"""
        results = {}
        
        models_to_try = ['llava:7b', 'llava:13b', 'llava:latest', 'moondream']
        
        prompts = {
            "description": "Describe this image in detail.",
            "objects": "What objects can you see in this image?",
            "scene": "What type of scene is this?",
            "tags": "List 5-10 keywords for this image, separated by commas."
        }
        
        working_model = None
        try:
            available_models = [model['name'] for model in self.ollama_client.list()['models']]
            for model in models_to_try:
                if model in available_models:
                    working_model = model
                    break
        except:
            working_model = 'llava:7b'
        
        if not working_model:
            return {"error": "No vision models available"}
        
        try:
            for key, prompt in prompts.items():
                response = self.ollama_client.chat(
                    model=working_model,
                    messages=[{
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }]
                )
                results[key] = response['message']['content'].strip()
        except Exception as e:
            logger.warning(f"LLaVA analysis failed: {e}")
            results["error"] = str(e)
        
        return results

class SimpleDatabase:
    """Simple SQLite database for image metadata"""
    
    def __init__(self, db_path: str = "./image_database.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create database and tables"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE,
                file_name TEXT,
                file_hash TEXT,
                created_at TEXT,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                format TEXT,
                blip_analysis TEXT,
                llava_analysis TEXT,
                tags TEXT,
                search_text TEXT
            )
        ''')
        self.conn.commit()
        logger.info("Database initialized")
    
    def insert_image_data(self, image_data: Dict):
        """Insert image data into database"""
        try:
            # Create search text from all analysis
            search_parts = []
            search_parts.append(image_data.get('file_name', ''))
            
            if 'blip_analysis' in image_data:
                for value in image_data['blip_analysis'].values():
                    if isinstance(value, str):
                        search_parts.append(value)
            
            if 'llava_analysis' in image_data:
                for value in image_data['llava_analysis'].values():
                    if isinstance(value, str):
                        search_parts.append(value)
            
            search_text = ' '.join(search_parts).lower()
            
            # Extract simple tags
            tags = set()
            for text in search_parts:
                if isinstance(text, str):
                    words = text.lower().split()
                    meaningful_words = [w for w in words if len(w) > 3 and w.isalpha()]
                    tags.update(meaningful_words[:20])
            
            tags_str = ','.join(list(tags))
            
            # Insert into database
            self.conn.execute('''
                INSERT OR REPLACE INTO images 
                (id, file_path, file_name, file_hash, created_at, file_size, 
                 width, height, format, blip_analysis, llava_analysis, tags, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                image_data['id'],
                image_data['file_path'],
                image_data['file_name'],
                image_data['file_hash'],
                image_data['created_at'],
                image_data['file_size'],
                image_data.get('image_properties', {}).get('width', 0),
                image_data.get('image_properties', {}).get('height', 0),
                image_data.get('image_properties', {}).get('format', ''),
                json.dumps(image_data.get('blip_analysis', {})),
                json.dumps(image_data.get('llava_analysis', {})),
                tags_str,
                search_text
            ))
            self.conn.commit()
            logger.info(f"Inserted: {image_data['file_name']}")
            
        except Exception as e:
            logger.error(f"Failed to insert data: {e}")
    
    def search_images(self, query: str, limit: int = 10) -> List[Dict]:
        """Simple text search in database"""
        try:
            query_lower = query.lower()
            
            # Search in search_text and tags
            cursor = self.conn.execute('''
                SELECT * FROM images 
                WHERE search_text LIKE ? OR tags LIKE ?
                ORDER BY 
                    CASE 
                        WHEN file_name LIKE ? THEN 1
                        WHEN tags LIKE ? THEN 2
                        ELSE 3
                    END
                LIMIT ?
            ''', (f'%{query_lower}%', f'%{query_lower}%', f'%{query_lower}%', f'%{query_lower}%', limit))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'id': row[0],
                    'file_path': row[1],
                    'file_name': row[2],
                    'file_hash': row[3],
                    'created_at': row[4],
                    'file_size': row[5],
                    'width': row[6],
                    'height': row[7],
                    'format': row[8],
                    'blip_analysis': json.loads(row[9]) if row[9] else {},
                    'llava_analysis': json.loads(row[10]) if row[10] else {},
                    'tags': row[11].split(',') if row[11] else [],
                    'search_text': row[12]
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def close(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()

class ImageTaggerSystem:
    """Main system"""
    
    def __init__(self, db_path: str = "./image_database.db"):
        self.analyzer = ImageAnalyzer()
        self.database = SimpleDatabase(db_path)
        self.supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    def is_image_file(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() in self.supported_formats
    
    def get_image_properties(self, image_path: str) -> Dict[str, Any]:
        """Get basic image properties"""
        try:
            image = Image.open(image_path)
            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "size_bytes": os.path.getsize(image_path),
                "aspect_ratio": round(image.width / image.height, 2),
            }
        except Exception as e:
            logger.warning(f"Failed to get properties: {e}")
            return {}
    
    def generate_file_hash(self, file_path: str) -> str:
        """Generate file hash"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def process_single_image(self, image_path: str) -> Dict[str, Any]:
        """Process single image"""
        try:
            logger.info(f"Processing: {Path(image_path).name}")
            
            file_hash = self.generate_file_hash(image_path)
            
            file_info = {
                "id": file_hash,
                "file_path": str(Path(image_path).absolute()),
                "file_name": Path(image_path).name,
                "file_hash": file_hash,
                "created_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(image_path)
            }
            
            file_info["image_properties"] = self.get_image_properties(image_path)
            
            image = Image.open(image_path).convert("RGB")
            
            file_info["blip_analysis"] = self.analyzer.analyze_with_blip(image)
            file_info["llava_analysis"] = self.analyzer.analyze_with_llava(image_path)
            
            return file_info
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {e}")
            return None
    
    def scan_directory(self, root_path: str):
        """Scan directory for images"""
        root_path = Path(root_path)
        
        if not root_path.exists():
            logger.error(f"Directory does not exist: {root_path}")
            return
        
        logger.info(f"Scanning: {root_path}")
        
        image_files = []
        for file_path in root_path.rglob("*"):
            if file_path.is_file() and self.is_image_file(str(file_path)):
                image_files.append(str(file_path))
        
        logger.info(f"Found {len(image_files)} images")
        
        for i, image_path in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}")
            
            image_data = self.process_single_image(image_path)
            if image_data:
                self.database.insert_image_data(image_data)
                
        logger.info("Scanning completed!")
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search images"""
        logger.info(f"Searching: '{query}'")
        results = self.database.search_images(query, limit)
        
        logger.info(f"Found {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"\n{i}. {result.get('file_name', 'Unknown')}")
            print(f"   Path: {result.get('file_path', 'Unknown')}")
            if result.get('tags'):
                print(f"   Tags: {', '.join(result['tags'][:8])}")
            if result.get('blip_analysis', {}).get('caption'):
                print(f"   Caption: {result['blip_analysis']['caption'][:100]}...")
        
        return results
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'database'):
            self.database.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Image Tagging")
    parser.add_argument("--scan", type=str, help="Directory to scan")
    parser.add_argument("--search", type=str, help="Search query")
    parser.add_argument("--db-path", type=str, default="./image_database.db", help="Database path")
    parser.add_argument("--limit", type=int, default=10, help="Search limit")
    
    args = parser.parse_args()
    
    system = ImageTaggerSystem(args.db_path)
    
    try:
        if args.scan:
            system.scan_directory(args.scan)
            print(f"\nFinished scanning {args.scan}")
        elif args.search:
            system.search(args.search, args.limit)
        else:
            print("Simple Image Tagger")
            print("Commands: scan <path>, search <query>, quit")
            
            while True:
                try:
                    cmd = input("\n> ").strip().split(maxsplit=1)
                    if not cmd:
                        continue
                    
                    if cmd[0] == "quit":
                        break
                    elif cmd[0] == "scan" and len(cmd) > 1:
                        system.scan_directory(cmd[1])
                    elif cmd[0] == "search" and len(cmd) > 1:
                        system.search(cmd[1])
                    else:
                        print("Usage: scan <path> | search <query> | quit")
                except KeyboardInterrupt:
                    break
    finally:
        if hasattr(system, 'database'):
            system.database.close()

if __name__ == "__main__":
    main()