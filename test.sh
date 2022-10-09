# srccのテスト
sbatch run nasbowl.py srcc --trials 10 -T 750
sbatch run nasbowl.py srcc --trials 10 -T 750 --d_max 800

# accのテスト
sbatch run nasbowl.py acc --trials 10 -T 750
sbatch run nasbowl.py acc --trials 10 -T 750 --d_max 800

# timeのテスト
sbatch run nasbowl.py time -T 1500 
sbatch run nasbowl.py time -T 1500 --d_max 800

# srccのテスト
sbatch run nasbowl.py srcc --trials 10 -T 750 --load_kernel_cache
sbatch run nasbowl.py srcc --trials 10 -T 750 --d_max 800 --load_kernel_cache

# accのテスト
sbatch run nasbowl.py acc --trials 10 -T 750 --load_kernel_cache
sbatch run nasbowl.py acc --trials 10 -T 750 --d_max 800 --load_kernel_cache

# timeのテスト
sbatch run nasbowl.py time -T 1500  --load_kernel_cache
sbatch run nasbowl.py time -T 1500 --d_max 800 --load_kernel_cache