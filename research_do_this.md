# DO THIS
```bash
cd kge && rm -rf local/experiments && rm -rf wandb/run-* && git pull
screen -R kge-runner
source ./venv/bin/activate && pip install networkx
kge start svd-experiment/g-svd-lp-1-4e8-d
```
