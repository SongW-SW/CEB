# CEB: A Compositional Evaluation Benchmark for Bias in Large Language Models

![The framework of CEB.](framework.png)

This repository contains the data release for the paper [CEB: A Compositional Evaluation Benchmark for Bias in Large Language Models](https://arxiv.org/pdf/2407.02408).

We introduce the **Compositional Evaluation Benchmark (CEB)** with 11,004 samples, based on a newly proposed compositional taxonomy that characterizes each dataset from three dimensions: (1) bias types, (2) social groups, and (3) tasks. Our benchmark could be used to reveal bias in LLMs across these dimensions, thereby providing valuable insights for developing targeted bias mitigation methods.

## Dataset

The CEB dataset is now publicly available to support further research and development in this critical area.

**[Dataset Files]**: ./data

**[HugginFace Dataset Link]**: [CEB Dataset](https://huggingface.co/datasets/Song-SW/CEB)

**[Dataset Statistics]**:

| **Dataset**            | **Task Type**   | **Bias Type**   | **Age** | **Gender** | **Race** | **Religion** | **Size** |
|------------------------|-----------------|-----------------|---------|------------|----------|--------------|----------|
| CEB-Recognition-S      | Recognition     | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Selection-S        | Selection       | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Continuation-S     | Continuation    | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Conversation-S     | Conversation    | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Recognition-T      | Recognition     | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Selection-T        | Selection       | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Continuation-T     | Continuation    | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Conversation-T     | Conversation    | Toxicity        | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-Adult              | Classification  | Stereotyping    | No      | Yes        | Yes      | No           | 500      |
| CEB-Credit             | Classification  | Stereotyping    | Yes     | Yes        | No       | No           | 500      |
| CEB-Jigsaw             | Classification  | Toxicity        | No      | Yes        | Yes      | Yes          | 500      |
| CEB-WB-Recognition     | Recognition     | Stereotyping    | No      | Yes        | No       | No           | 792      |
| CEB-WB-Selection       | Selection       | Stereotyping    | No      | Yes        | No       | No           | 792      |
| CEB-SS-Recognition     | Recognition     | Stereotyping    | No      | Yes        | Yes      | Yes          | 960      |
| CEB-SS-Selection       | Selection       | Stereotyping    | No      | Yes        | Yes      | Yes          | 960      |
| CEB-RB-Recognition     | Recognition     | Stereotyping    | No      | Yes        | Yes      | Yes          | 1000     |
| CEB-RB-Selection       | Selection       | Stereotyping    | No      | Yes        | Yes      | Yes          | 1000     |
| CEB-CP-Recognition     | Recognition     | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |
| CEB-CP-Selection       | Selection       | Stereotyping    | Yes     | Yes        | Yes      | Yes          | 400      |


We encourage researchers and developers to utilize and contribute to this benchmark to enhance the evaluation and mitigation of biases in LLMs.


## Configuration
Before running, specify the configurations (e.g., OpenAI API key) in ./src/config/config.py.

## Running

Execute the corresponding bash files in ./script. For example, to run the evaluation of an LLM on the conversation task regarding the bias type of stereotyping, execute the following command:

```
bash run_gen_stereotype_conversation.sh
```

The specific LLM for evaluation should be specified in the same bash file.

## Questions

If you encounter any cases and need help, feel free to contact ```sw3wv@virginia.edu``` and ```pw7nc@virginia.edu```. We are more than willing to help!

## Citation

If you find our work helpful, please kindly consider citing our paper. Thank you so much for your attention!
```
@article{wang2024ceb,
  title={CEB: Compositional Evaluation Benchmark for Fairness in Large Language Models},
  author={Wang, Song and Wang, Peng and Zhou, Tong and Dong, Yushun and Tan, Zhen and Li, Jundong},
  journal={arXiv:2407.02408},
  year={2024}
}
```
