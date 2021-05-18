python -u processing.py
python -u train.py ProcR 1 2Prong_Contaminated 20 0 0.1 16 2 0
python -u eval.py ProcR 1 2Prong_Contaminated 20 0 0.1 16 2 0
python -u train.py ProcR 1 3Prong_Contaminated 20 0 0.1 16 2 0
python -u eval.py ProcR 1 3Prong_Contaminated 20 0 0.1 16 2 0
python -u plot.py 2
python -u plot.py 3
