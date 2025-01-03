# Load model directly
import os
import torch
from prompt_utils import get_prompt
import json
import argparse
import utils
from prompt_utils import *
from data_loader import BatchDatasetLoader
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='', type=str)
parser.add_argument("--stem_flan_type", default='', choices=['', 'pot_prompt'], type=str)
parser.add_argument("--dtype", default='bfloat16', type=str)
parser.add_argument("--dataset", required=True, type=str)
parser.add_argument("--form", default='alpaca', type=str)
parser.add_argument("--shots", default=0, type=int)
parser.add_argument("--model_max_length", default=2048, type=int)
parser.add_argument("--cot_backup", action='store_true', default=False)
parser.add_argument("--tiny", action='store_true', default=False)
parser.add_argument("--output_folder", default='outputs', type=str)

args = parser.parse_args()

DTYPES = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}

def print_to_both(message, file):
    # Print to console
    print(message)
    # Write to the file
    print(message, file=file)


def get_seperation_trigger(dataset: str):
    triggers = ['The answer is:', 'The answer is', 'the answer is']
    if dataset == 'gsm8k':
        triggers.append('####')
    return triggers


def run_question_answer(questions: list, groundtruths: list, tasks: list, collect_rerun: bool = False):
    assert len(questions) == len(groundtruths) == len(tasks)
    used_examples = get_examples(tasks, args.shots, args.stem_flan_type)
    prompt_prefixs = [get_prompt(example, args.form) for example in used_examples]
    input_strs = [p[0] + p[1].format(query=q) for p, q in zip(prompt_prefixs, questions)]

    outputs = llm.generate(input_strs, sampling_params)
    outputs = [output.outputs[0].text for output in outputs]

    # We need to collect the values and possibly the rerun questions;
    returned_value = []
    rerun_questions = []
    rerun_groundtruths = []
    rerun_tasks = []
    for output, question, groundtruth, task in zip(outputs, questions, groundtruths, tasks):
        if 'print(' in output:
            output = output.split("### Instruction")[0]
            # output = "\n".join(line.lstrip() for line in output.splitlines())
            output = output.lstrip() 
            if output[:len('Python')].lower()== 'Python'.lower(): output = '#' + output
            tmp = utils.execute_with_timeout(output)
            tmp = 'The answer is' + ' ' + tmp
            answer = utils.answer_clean(args.dataset, get_seperation_trigger(args.dataset), tmp)
        else:
            answer = utils.answer_clean(args.dataset, get_seperation_trigger(args.dataset), output)

        if answer == "" and collect_rerun:
            rerun_questions.append(utils.remove_flan_tag(question, args.stem_flan_type))
            rerun_groundtruths.append(groundtruth)
            rerun_tasks.append(task)
            continue

        returned_value.append((question, output, answer, groundtruth, task))

    if collect_rerun:
        assert len(returned_value) + len(rerun_questions) == len(questions) == len(groundtruths)
        return returned_value, rerun_questions, rerun_groundtruths, rerun_tasks
    else:
        return returned_value


if __name__ == "__main__":
    stop_tokens = ["USER:", "ASSISTANT:",  "### Instruction:", "Response:",
                   "\n\nProblem", "\nProblem", "Problem:", "<|eot_id|>", "####"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=args.model_max_length, stop=stop_tokens)
    llm = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), dtype=args.dtype, trust_remote_code=True)
    tokenizer = llm.get_tokenizer()
    print('Using VLLM, we do not need to set batch size!')

    correct, wrong = 0, 0

    suffix = 'PoT' if 'pot' in args.stem_flan_type.lower() else 'CoT'
    filename = args.model.strip('/').split('/')[-1].replace('-', '_')
    if filename.startswith('checkpoint'):
        filename = args.model.strip('/').split('/')[-2].replace('-', '_') + '__' + filename
    filename = filename + '/' + args.dataset.replace('/', '-')
    filename += '/' + f'{args.shots}shots' + '_' + args.form
    filename += f'_length{args.model_max_length}'
    if args.cot_backup:
        filename += '_CoTBackup'
    filename += '_' + suffix
    output_path = f'outputs-{args.output_folder}/{filename}.jsonl'
    print('Writing the output to', output_path)
    result_path = f'results-{args.output_folder}/{filename}.txt'

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    if not os.path.exists(os.path.dirname(result_path)):
        os.makedirs(os.path.dirname(result_path))

    file_handle = open(output_path, 'w')
    loader = BatchDatasetLoader(args.dataset, -1)

    questions, groundtruths, tasks = loader[0]
    if args.tiny:
        questions, groundtruths, tasks = questions[:20], groundtruths[:20], tasks[:20]
    processed_questions = utils.process_question_with_flan_tag(questions, args.stem_flan_type)

    if args.stem_flan_type == 'pot_prompt' and args.cot_backup:
        # if there is hybrid decoding, we try pot fist and then cot
        returned_values, rerun_questions, rerun_groundtruths, rerun_tasks = run_question_answer(
            processed_questions, groundtruths, tasks, collect_rerun=True)
        if rerun_questions:
            # if things are not working well
            processed_questions = utils.process_question_with_flan_tag(rerun_questions, "")
            tmp = run_question_answer(processed_questions, rerun_groundtruths, rerun_tasks, collect_rerun=False)
            returned_values += tmp
    else:
        # only cot_prompt or pot_prompt, then we don't need to rerun
        returned_values = run_question_answer(processed_questions, groundtruths, tasks, collect_rerun=False)

    for question, output, answer, groundtruth, task in returned_values:
        if isinstance(groundtruth, str):
            groundtruth = [groundtruth]
        if len(answer) > 100: answer=answer[:100]
        if utils.compare_answer_with_groundtruth(answer, *groundtruth):
            correct += 1
        else:
            wrong += 1

        example = {
            'question': question,
            'correct': groundtruth,
            'solution': output,
            'pred': answer,
            'task': task
        }

        file_handle.write(json.dumps(example) + '\n')

    with open(result_path, "a") as f:
        message = f'final accuracy: {correct / (correct + wrong)}'
        print_to_both(message, f)
    file_handle.close()
