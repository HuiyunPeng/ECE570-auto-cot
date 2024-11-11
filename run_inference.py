import argparse
from utils import *
import os
from dotenv import load_dotenv

# This script is designed to perform inference using a language model in a zero-shot reasoning context.
# It parses command-line arguments to configure the experiment, including dataset selection, output directory,
# and model parameters. The script loads a pre-trained model and tokenizer, processes a dataset of questions
# and answers, generates responses using the model, and evaluates the accuracy of the predictions against
# the ground truth. The results are logged to a specified output file, and the script supports resuming
# from a specific question ID in case of interruptions. It also includes functionality for answer cleansing
# and managing API rate limits.

def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    
    fix_seed(args.random_seed)
    
    load_dotenv()
    print("OPENAI_API_KEY:")
    print(os.getenv("OPENAI_API_KEY")[0:5] + '**********')
    
    # Initialize decoder class (load model and tokenizer) ...
    decoder = Decoder()
    
    print("setup data loader ...")
    dataloader = setup_data_loader(args)
    print_now()

    demo = create_demo_text(args, cot_flag=True)
    demo2 = create_demo_text2(args)

    total = 0
    correct_list = []
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    with open(args.output_dir, "a") as wp:

        for i, data in enumerate(dataloader):
            if i < args.resume_id - 1:
            # if i < 297:
                continue
            output_line = {}
            
            print('*************************')
            print("{}st data".format(i+1))
                    
            # Prepare question template ...
            x, y = data
            x = "Q: " + x[0] + "\n" + "A:"
            y = y[0].strip()
                        
            output_line["question"] = x
            output_line["gold_ans"] = y

            demo2.append({"role": "user", "content": x})
            demo2.append({"role": "system", "content": args.cot_trigger})
            x = demo + x + " " + args.cot_trigger
            
            # Answer experiment by generating text ...
            max_length = args.max_length_cot
            z = decoder.decode(args, demo2, max_length)

            output_line["rationale"] = z

            pred = z
            print(x + str(pred))

            # Clensing of predicted answer ...
            pred = answer_cleansing(args, pred)
            
            
            output_line["pred_ans"] = pred
            output_line["wrap_que"] = x

            output_json = json.dumps(output_line)
            wp.write(output_json + '\n')

            # Choose the most frequent answer from the list ...
            print("pred : {}".format(pred))
            print("GT : " + y)
            print('*************************')
            
            # Checking answer ...
            correct = (np.array([pred]) == np.array([y])).sum().item()
            correct_list.append(correct)
            total += 1 #np.array([y]).size(0)

            # Calculate and print accuracy in each iteration ...
            accuracy = (sum(correct_list) * 1.0 / total) * 100
            print("Current accuracy : {}".format(accuracy))
            
            if (args.limit_dataset_size != 0) and ((i+1) >= args.limit_dataset_size):
                break

    # Calculate accuracy ...
    accuracy = (sum(correct_list) * 1.0 / total) * 100
    print("accuracy : {}".format(accuracy))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="commonsensqa", choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith",  "strategyqa", "svamp", "singleeq", "coin_flip"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/commonsensqa", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0, help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1], help="minibatch size should be 1 because GPT API takes only 1 input for each request")
    
    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")
    
    parser.add_argument(
        "--output_dir", type=str, default="experiment/commonsensqa", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256, help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32, help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0, help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0, help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )
    parser.add_argument(
        "--llm_model", type=str, default="llama", help="select LLM model"
    )
    
    args = parser.parse_args()
    
    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test.json"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")
        
    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."
    
    return args

if __name__ == "__main__":
    main()