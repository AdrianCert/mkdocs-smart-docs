# State of the Art Document: Plugin for MkDocs to Create an Interactive Chat for Q&A

## 1. Project Overview

In the realm of technical documentation, MkDocs is a popular static site generator that allows users to create beautiful, versioned documentation using Markdown. However, as documentation grows in size and complexity, navigating it can become a challenge for users seeking quick answers to specific questions. To address this, a plugin for MkDocs that integrates an interactive Q&A chat system, based on the content of the documentation, can significantly improve user experience.

The proposed system involves creating an interactive chat that can provide answers to questions posed by users based on the content of the documentation. The key challenge in this project lies in training the model with appropriate data derived from the documentation, especially when the text is arbitrarily structured. This document will present the concept, outline the challenges, and explore the existing state-of-the-art techniques and tools available to solve this problem.

The goal of this project is to create an MKDocs plugin that extent functionality of a statically generated html by offering a possibility for answers questions based on the content of the documentation.

## 2. Current situations

Question-Answering (QA) systems aim to provide direct answers to user queries based on a given corpus of text, such as documentation, articles, or other knowledge sources. There are two main categories of QA systems: Extractive QA and Generative QA, each utilizing different approaches to retrieve or generate answers.

1. Extractive QA models like BERT excel in retrieving precise answers directly from the text, offering high accuracy and efficiency, but are limited to existing content within the document.

2. Generative QA models like GPT can create flexible answers by synthesizing information, but they may generate incorrect or irrelevant responses, especially in ambiguous contexts.

2. Hybrid QA systems, combining extractive and generative approaches, balance accuracy and flexibility, providing robust answers by both retrieving relevant information and generating contextually appropriate responses.

|      Model Type      |                      Strengths (+)                                                         |                                           Weaknesses (−)                                           |
|:--------------------:|:------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:|
| Extractive QA (BERT) | + High accuracy and efficiency<br>+ Fast responses                                         | − Limited to content directly available in the text<br>- Struggle with ambiguous and complex pomps |
| Generative QA (GPT)  | + Flexible and can generate answers from multiple sources<br>+                             | - Can be computationally expensive<br>- Risk of generating incorrect or hallucinated responses     |
| Hybrid QA (RAG)      | + Combines accuracy with flexibility<br>+Better suited for complex, context-driven queries | - More complex and resource-intensive<br>- Slower response time due to dual processes              |


## 3. Concept Overview

The central idea of this project is to develop a plugin that can be integrated with MkDocs to create an interactive chat interface. This chat interface will answer questions based on the content of the documentation, allowing users to ask specific questions and receive instant responses without navigating the entire document manually.

### 3.1 Key Features

**Chatbot Integration**: A real-time, interactive chat interface embedded within the MkDocs-generated site.

**Content Parsing and Understanding**: The chatbot needs to understand the context of the documentation and provide relevant answers.

**Dynamic Data Training**: The ability to train the system using content from various Markdown files, ensuring the answers are tailored to the specific content.

### 3.2 How it Works

**Content Parsing**: The plugin will extract the content of the MkDocs site (from the generated Markdown files or HTML).

**Data Preprocessing**: The content will be transformed into a format suitable for training a language model, such as splitting text into smaller chunks and creating a knowledge base.

**Question-Answer Generation**: Using Natural Language Processing (NLP) techniques, the chatbot will generate answers to the questions posed by the user by matching the query against the knowledge base created from the documentation.

**Chatbot Interface**: The user interface will allow real-time interactions between the user and the system, responding with the most relevant information.

### 3.3. Deployment View

- **Client-Server Model**: Questions are sent to a backend server hosting the model, which processes the query and returns an answer.

- **Client-Side Model**: The model is converted to ONNX format and executed directly in the browser using WebAssembly or ONNX.js, minimizing server dependency.

### 3.4 Dataset working

- Extract content from Markdown documentation files to generate a structured representation (e.g., sections, subsections, and paragraphs).

- Preprocess content to ensure it is ready for training and inference.

- Create QA / A SQuAD (Stanford Question Answering Dataset) entry

```json
{
  "data": [
    {
      "title": "Document",
      "paragraphs": [
        {
          "context": "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
          "qas": [
            {
              "question": "Q1",
              "id": "ID1",
              "answers": [
                {
                  "text": "A1",
                  "answer_start": 23
                }
              ],
              "is_impossible": false
            }
          ]
        }
      ]
    }
  ]
}
```

### 3.5 Heuristics Evaluation

In the context of QA systems, heuristic scoring could involve several factors that measure how well an answer aligns with the expected or ideal response following criteria based on:

- Relevance
- Accuracy
- Completeness
- Contextual Understanding
- Coherence
- Grammar and Spelling
- Reusability
- etc.

### 3.6 Fine-tunning

Fine-tuning is a critical step in transfer learning, where knowledge gained from one domain (e.g., general text understanding) is transferred to a more specific domain (e.g., answering questions based on technical documentation).

1. Select a Pre-trained Model
2. Prepare Task-Specific Dataset
3. Fine-Tuning the Model
4. Evaluate and Test

## 4.  Challenges and Problem Adoption

The primary challenge in this project is to create training data from arbitrary text (Markdown documentation) to train a model that can answer questions effectively. Text in documentation is typically non-conversational, containing structured information, headers, and explanations, which complicates the process of creating a suitable training dataset.

### 3.1 Challenge 1: Parsing and Structuring Documentation

Documentation often follows a hierarchical structure, with headings, subheadings, lists, and code blocks, which does not align with conversational data. Extracting meaningful content from such arbitrary text structures requires:

**Content Segmentation**: Identifying which sections of the documentation are relevant for specific queries.
**Contextual Representation**: Mapping unstructured text into a context that a chatbot can understand.

### 4.2 Challenge 2: Generating Training Data

Unlike conversational datasets, documentation does not naturally come with question-answer pairs. A solution to this challenge might involve:

**Automatic Question Generation (AQG)**: Developing techniques that automatically generate questions from documentation content. For example, questions could be generated by summarizing key points or by transforming facts into question formats.

**Manual Supervision**: Curating a small set of high-quality question-answer pairs manually and then using semi-supervised learning to expand the dataset.

### 4.3 Challenge 3: Language Model Adaptation

The chatbot needs to answer questions based on the context and content of the documentation. This requires the language model to:

**Understand Technical Terms**: Documentation often contains domain-specific jargon and abbreviations that the model needs to comprehend.

**Contextual Recall**: Unlike traditional QA systems, where responses are based on a fixed dataset, here the model needs to "understand" the ongoing conversation and the context of the documentation. This is especially difficult with non-standard formatting like code snippets, tables, and hyperlinks.

### 4.4 Challenge 4: Real-Time Interaction

For the plugin to be useful, it must function in real-time. The time it takes to retrieve a response should be as short as possible to ensure an interactive experience. Techniques such as:

**Efficient Search Algorithms**: Using vector-based search (e.g., semantic search using embeddings) can help speed up the response process by narrowing down the relevant content before processing.

**Caching Mechanisms**: To further improve response time, caching previous queries and their answers can be employed.

## 5. Proposed Solution

To address the problem of creating interactive Q&A for MkDocs documentation, the following steps outline the approach:

1. **Content Parsing**: Extract and preprocess content from MkDocs-generated HTML or Markdown files. This will involve identifying relevant sections, paragraphs, headers, and code snippets.

2. **Automatic Question Generation (AQG)**: Use AQG models to generate a set of questions based on the documentation.

3. **Training the Language Model**: Fine-tune an existing language model (such as BERT or GPT) on the documentation content and the generated question-answer pairs.

4. **Integration with MkDocs**: Develop the plugin to embed a chatbot interface on the MkDocs site, capable of responding in real-time based on the trained model.

5. **Optimization for Real-time Use**: Implement vector-based search and caching strategies to ensure fast response times during user interaction.
