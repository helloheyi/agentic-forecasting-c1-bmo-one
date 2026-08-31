#this file is used to check optimization process in a separate terminal (but have to stay in the same Coder environment). 
#go down to the project root directory. In the terminal, first run source .venv/bin/activate
#then python sqlliteLogChk1.py  
#One observed sample:
#baa10y_change_5b_univariate: 2 trials
#  trial 0: state=COMPLETE, duration=0:13:11.956033, value=2.9994162690666233
#  trial 1: state=COMPLETE, duration=0:09:32.377041, value=3.01049890004928

#baa10y_change_5b_covariate: 2 trials
#  trial 0: state=COMPLETE, duration=2:32:01.870038, value=3.3394840209640164
#  trial 1: state=RUNNING, duration=None, value=None

from pathlib import Path
import optuna

def _repo_root() -> Path:
    here = Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").exists() and (cand / "aieng-forecasting").is_dir():
            return cand
    return here

root = _repo_root()
db_path = root / "data" / "lgbm_tuning" / "optuna_studies.db"
storage = f"sqlite:///{db_path.resolve().as_posix()}"

for name in ["baa10y_change_5b_univariate", "baa10y_change_5b_covariate"]:
    try:
        study = optuna.load_study(study_name=name, storage=storage)
    except KeyError:
        print(name, "-> not created yet")
        continue
    print(f"\n{name}: {len(study.trials)} trials")
    for t in study.trials:
        dur = (t.datetime_complete - t.datetime_start) if t.datetime_complete else None
        print(f"  trial {t.number}: state={t.state.name}, duration={dur}, value={t.value}")

trialToCheck = 1
study = optuna.load_study(study_name="baa10y_change_5b_covariate", storage=storage)
print(study.trials[trialToCheck].params)
