import pandas as pd 
import numpy as np
import argparse

from rule_mining import generate_rulespace, screen_rules



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='Dataset name. Options: adult, compas', default='compas')
    parser.add_argument("--method", type=str, help='Rule Mining Method. Options: fpgrowth, rf', default='fpgrowth')
    parser.add_argument("--min_support", type=int, help='Minimum Support in Percentage', default=5)
    parser.add_argument("--max_card", type=int, help='Maximum Length', default=2)
    parser.add_argument("--n_rules", type=int, help='Maximum Number of Rules', default=2000)
    args = parser.parse_args()

    df = pd.read_csv(f"data/{args.dataset}.csv", sep = ',')
    X = df.iloc[:, :-1]
    y = np.array(df.iloc[:, -1])
    prediction_name = df.columns[-1]
    print(X.shape)
    
    X, prules, nrules = generate_rulespace(X, y, max_card=args.max_card, neg_columns=True ,method=args.method)

    prules, pRMatrix = screen_rules(prules, X, N_rules=args.n_rules // 2, min_support=args.min_support)
    nrules, nRMatrix = screen_rules(nrules, X, N_rules=args.n_rules // 2, min_support=args.min_support)

    # Reformat rules
    prules = [" && ".join(rule) for rule in prules]
    nrules = [" && ".join(rule) for rule in nrules]
    
    # Save processed data
    df = pd.DataFrame(np.hstack((pRMatrix, y.reshape((-1, 1)))), columns=prules + [prediction_name])
    fname = f"{args.dataset}_pos_mined_{args.method}_{args.min_support}_{args.max_card}_{args.n_rules}"
    df.to_csv(f"data/{fname}.csv", encoding='utf-8', index=False)

    df = pd.DataFrame(np.hstack((nRMatrix, y.reshape((-1, 1)))), columns=nrules + [prediction_name])
    fname = f"{args.dataset}_neg_mined_{args.method}_{args.min_support}_{args.max_card}_{args.n_rules}"
    df.to_csv(f"data/{fname}.csv", encoding='utf-8', index=False)
    print(f"({df.shape[0]}, {len(prules) + len(nrules)})")


if __name__ == "__main__":
    main()
