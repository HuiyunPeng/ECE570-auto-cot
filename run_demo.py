import random
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
from utils import fix_seed
import hdbscan
from sklearn.preprocessing import StandardScaler

# This script performs zero-shot reasoning chain generation and clustering on a specified dataset.
# It includes functionalities to parse command-line arguments, process a corpus of questions and answers,
# generate embeddings using a sentence transformer, cluster the embeddings using HDBSCAN, and construct
# demonstrations based on the clustering results. Additionally, it visualizes the clustering results using PCA.
# The script is designed to support various tasks and allows for debugging and random seed control.

def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")
    parser.add_argument(
        "--task", type=str, default="commonsensqa",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq", "coin_flip"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--max_ra_len", type=int, default=5, help="maximum number of reasoning chains"
    )
    parser.add_argument(
        "--pred_file", type=str, default="log/commonsensqa_zero_shot_cot.log",
        help="use the reasoning chains generated by zero-shot-cot."
    )
    parser.add_argument(
        "--demo_save_dir", type=str, default="demos/commonsensqa", help="where to save the contructed demonstrations"
    )
    parser.add_argument("--random_seed", type=int, default=192, help="random seed")
    parser.add_argument(
        "--encoder", type=str, default="all-MiniLM-L6-v2", help="which sentence-transformer encoder for clustering"
    )
    parser.add_argument(
        "--sampling", type=str, default="center", help="whether to sample the cluster center first"
    )
    parser.add_argument(
        "--debug", type=bool, default=True, help="debug mode"
    )
    args = parser.parse_args()
    return args

def generate_visual(args, clustering_model, corpus_embeddings, save_file):
    y_km = clustering_model.fit_predict(corpus_embeddings)
    pca_model = PCA(n_components=2, random_state=args.random_seed)
    transformed = pca_model.fit_transform(corpus_embeddings)

    # Simple scatter plot without cluster centers
    plt.scatter(x=transformed[:, 0], y=transformed[:, 1], c=y_km, s=50, cmap=plt.cm.Paired, alpha=0.4)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_file+".png", dpi=600)

def construct_demo(args, clustered_dists, rationale, pred_ans, clustered_idx, question, max_ra_len, gold_ans):
    demos = []

    for i in range(len(clustered_dists)):
        print("Cluster ", i+1)
        tmp = list(map(list, zip(range(len(clustered_dists[i])), clustered_dists[i])))
        top_min_dist = sorted(tmp, key=lambda x: x[1], reverse=False)
        if not args.sampling == "center":
            random.shuffle(top_min_dist)
        for element in top_min_dist:
            min_idx = element[0]
            c_rationale = rationale[clustered_idx[i][min_idx]].strip()
            c_pred_ans = pred_ans[clustered_idx[i][min_idx]].strip()

            if len(question[clustered_idx[i][min_idx]].strip().split()) <= 60 \
                and len(c_rationale.replace("\n\n", "\n").split("\n")) <= max_ra_len and c_rationale[-1] == "." and c_pred_ans != "":
                if args.task in ["gsm8k", "multiarith", "singleeq", "addsub", "svamp"]:
                    if not (c_pred_ans.strip() in c_rationale.split(".")[-2] or c_pred_ans.strip() in c_rationale.split()[-10:]):
                        continue
                c_question = question[clustered_idx[i][min_idx]]
                c_rationale = c_rationale.replace("\n\n", "\n").replace("\n", " ").strip()
                c_rationale = " ".join(c_rationale.split())
                if args.debug:
                    c_gold_ans = gold_ans[clustered_idx[i][min_idx]]
                else:
                    c_gold_ans = None
                demo_element = {
                    "question": c_question,
                    "rationale": c_rationale,
                    "pred_ans": c_pred_ans,
                    "gold_ans": c_gold_ans,
                }
                demos.append(demo_element)
                print(c_question)
                print(c_rationale)
                print(c_pred_ans)
                print(c_gold_ans)
                print("")
                break

    demos = {"demo": demos}

    with open(args.demo_save_dir, 'w', encoding="utf-8") as write_f:
        json.dump(demos, write_f, indent=4, ensure_ascii=False)

def process_corpus(args, pred_file):
    corpus = []
    question = []
    rationale = []
    gold_ans = []
    pred_ans = []

    with open(pred_file, "r", encoding="utf-8") as fp:
        answer_seg = ""
        for line in fp:
            if "Q: " in line:
                c_question = line.strip()
            if "A: " in line:
                answer_seg = line
            elif "Therefore" in line and "the answer" in line:
                c_rationale = answer_seg

            elif answer_seg != "":
                answer_seg += line
            if "pred_mode" in line:
                c_pred_ans = line.split(":")[1].strip()
            if "GT :" in line:
                c_gold_ans = line.split(":")[1].strip()

                c_rationale = c_rationale.replace("A: Let's think step by step.", "Let's think step by step.")
                c_question = c_question + "\nA:"

                corpus.append(c_question)
                question.append(c_question)
                rationale.append(c_rationale)
                pred_ans.append(c_pred_ans)
                if args.debug:
                    gold_ans.append(c_gold_ans)
                answer_seg = ""
    return corpus, question, rationale, gold_ans, pred_ans

def clustering(args, corpus_embeddings, corpus):
    min_cluster_size = 5
    if args.task == "multiarith":
        min_cluster_size = 8
    elif args.task == "coin_flip":
        min_cluster_size = 2
    elif args.task == "svamp":
        min_cluster_size = 18

    clustering_model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric='euclidean',
    )
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    # Get the unique cluster labels (excluding noise points marked as -1)
    unique_clusters = sorted(list(set(label for label in cluster_assignment if label != -1)))
    num_actual_clusters = len(unique_clusters)

    # Initialize lists with actual number of clusters
    clustered_sentences = [[] for _ in range(num_actual_clusters)]
    clustered_dists = [[] for _ in range(num_actual_clusters)]
    clustered_idx = [[] for _ in range(num_actual_clusters)]

    # Use probabilities
    dist = clustering_model.probabilities_.reshape(-1, 1)

    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if cluster_id != -1:  # Skip noise points
            # Map cluster_id to a zero-based index
            cluster_idx = unique_clusters.index(cluster_id)
            clustered_sentences[cluster_idx].append(corpus[sentence_id])
            clustered_dists[cluster_idx].append(dist[sentence_id][0])
            clustered_idx[cluster_idx].append(sentence_id)
    return clustered_dists, clustered_idx, clustering_model

def main():
    args = parse_arguments()
    fix_seed(args.random_seed)
    encoder = SentenceTransformer(args.encoder)

    pred_file = args.pred_file
    save_file = args.demo_save_dir
    max_ra_len = args.max_ra_len

    corpus, question, rationale, gold_ans, pred_ans = process_corpus(args, pred_file)

    corpus_embeddings = encoder.encode(corpus)

    scaler = StandardScaler()
    corpus_embeddings = scaler.fit_transform(corpus_embeddings)

    clustered_dists, clustered_idx, clustering_model = clustering(args, corpus_embeddings, corpus)

    construct_demo(args, clustered_dists, rationale, pred_ans, clustered_idx, question, max_ra_len, gold_ans)
    
    generate_visual(args, clustering_model, corpus_embeddings, save_file)

if __name__ == "__main__":
    main()