import os; os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import subprocess,sys
print('[INFO] =========================================')
print('[INFO] Starting Lab 5: Production Data Pipeline')
print('[INFO] =========================================')
for s in ['src/validate_data.py','src/preprocess_pipeline.py','src/validate_outputs.py']:
 print(f'[INFO] ---> Executing {s}...'); subprocess.run([sys.executable,s],check=True)
print('[SUCCESS] Lab 5 Production Pipeline fully executed!')
