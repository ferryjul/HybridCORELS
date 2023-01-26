python3.10 compute_pareto.py --dataset=compas --bbox=random_forest --mode=test 
python3.10 compute_pareto.py --dataset=compas --bbox=random_forest --mode=valid 
python3.10 compute_pareto.py --dataset=compas --bbox=ada_boost --mode=test 
python3.10 compute_pareto.py --dataset=compas --bbox=ada_boost --mode=valid 
python3.10 compute_pareto.py --dataset=compas --bbox=gradient_boost --mode=test 
python3.10 compute_pareto.py --dataset=compas --bbox=gradient_boost --mode=valid 

python3.10 compute_pareto.py --dataset=adult --bbox=random_forest --mode=test --min_acc=0.81 --max_acc=0.85
python3.10 compute_pareto.py --dataset=adult --bbox=random_forest --mode=valid --min_acc=0.81 --max_acc=0.85
python3.10 compute_pareto.py --dataset=adult --bbox=ada_boost --mode=test --min_acc=0.81 --max_acc=0.85
python3.10 compute_pareto.py --dataset=adult --bbox=ada_boost --mode=valid --min_acc=0.81 --max_acc=0.85
python3.10 compute_pareto.py --dataset=adult --bbox=gradient_boost --mode=test --min_acc=0.81 --max_acc=0.85
python3.10 compute_pareto.py --dataset=adult --bbox=gradient_boost --mode=valid --min_acc=0.81 --max_acc=0.85

python3.10 compute_pareto.py --dataset=acs_employ --bbox=random_forest --mode=test --min_acc=0.70 --max_acc=0.75
python3.10 compute_pareto.py --dataset=acs_employ --bbox=random_forest --mode=valid --min_acc=0.70 --max_acc=0.75
python3.10 compute_pareto.py --dataset=acs_employ --bbox=ada_boost --mode=test --min_acc=0.70 --max_acc=0.75
python3.10 compute_pareto.py --dataset=acs_employ --bbox=ada_boost --mode=valid --min_acc=0.70 --max_acc=0.75
python3.10 compute_pareto.py --dataset=acs_employ --bbox=gradient_boost --mode=test --min_acc=0.70 --max_acc=0.75
python3.10 compute_pareto.py --dataset=acs_employ --bbox=gradient_boost --mode=valid --min_acc=0.70 --max_acc=0.75