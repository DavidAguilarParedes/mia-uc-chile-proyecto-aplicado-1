import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.infrastructure.loaders.pdf_loader import PDFLoader
from src.application.services.dataset_generation_service import DatasetGenerationService

def main():
    # Configuración
    PDF_PATH = "data/1-s2.0-S259015752400539X-main.pdf" # Cambia esto por tu archivo
    OUTPUT_DIR = "datasets"
    TEST_SIZE = 3 # Empieza con pocos para probar localmente
    
    # 1. Instanciar Loader (Usamos tu PDFLoader existente)
    loader = PDFLoader()
    
    # 2. Instanciar Servicio
    service = DatasetGenerationService(loader=loader)
    
    # 3. Ejecutar
    if os.path.exists(PDF_PATH):
        service.run(input_file=PDF_PATH, output_dir=OUTPUT_DIR, test_size=TEST_SIZE)
    else:
        print(f"Error: No se encuentra el archivo {PDF_PATH}")

if __name__ == "__main__":
    main()

