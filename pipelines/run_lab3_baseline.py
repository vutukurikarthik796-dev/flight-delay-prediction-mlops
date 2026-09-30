import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess,sys
print('[INFO] =========================================')
print('[INFO] Starting Lab 3: Baseline ML Training')
print('[INFO] =========================================')
subprocess.run([sys.executable,'src/validate_data.py'],check=True)
subprocess.run([sys.executable,'src/train_models.py'],check=True)
print('[SUCCESS] Lab 3 Baseline Pipeline fully executed!')
