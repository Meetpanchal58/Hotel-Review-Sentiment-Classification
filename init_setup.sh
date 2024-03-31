echo "[$(date)]: START"

echo "[$(date)]: creating env with python 3.9 version"
conda create -n myenv python=3.9

echo "[$(date)]: activating the environment"
conda activate myenv

echo "[$(date)]: installing the dev requirements"
pip install -r requirements_dev.txt

echo "[$(date)]: END"

