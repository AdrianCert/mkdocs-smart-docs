import collections
import contextlib
import json
import logging
import typing
from functools import cached_property
from pathlib import Path

import spacy
import torch
from spacy.cli import download as spacy_download
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline,
)

QAPair = collections.namedtuple("QAPair", ["question", "answer"])


class Model:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        qa_data: typing.List[QAPair],
    ):
        self.model: PreTrainedModel = model
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.qa_data: typing.List[QAPair] = qa_data
        self.qa_pipeline = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    @classmethod
    def guess_bases(cls, model_path: Path):
        if "t5" in model_path.name:
            return T5ForConditionalGeneration, T5Tokenizer
        if "roberta" in model_path.name:
            return AutoModelForQuestionAnswering, AutoTokenizer
        return None

    @classmethod
    def load(cls, path: Path):
        try:
            model_cls, tokenizer_cls = cls.guess_bases(path)
        except Exception:
            raise ValueError(f"Cannot determine model type for {path}")
        try:
            qa_data = json.loads(path.joinpath("qa-dataset.json").read_text())
        except Exception:
            qa_data = []

        return cls(
            model=model_cls.from_pretrained(path),
            tokenizer=tokenizer_cls.from_pretrained(path),
            qa_data=qa_data,
        )

    @cached_property
    def corpus(self):
        return "\n".join([QAPair(*qa).answer for qa in self.qa_data])

    def answer(self, question: str):
        res = self.qa_pipeline(question=question, context=self.corpus)
        return res["answer"]

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        path.joinpath("qa-dataset.json").write_text(json.dumps(self.qa_data))


class DocumentProcessor:
    def __init__(self):
        self.questions_generated = []

    def generate_questions(self, text):
        """Generates questions based on a given text."""
        for res in self.mentor_model(f"generate questions: {text}"):
            yield res["generated_text"]

    def split_into_sentences(self, document):
        """Splits the document into sentences. You can replace this with a more advanced NLP sentence splitter."""
        # Basic splitting based on punctuation
        return document.split(".")

    def create_squad_entry(self, question, context):
        """Creates a SQuAD format entry."""
        return {
            "title": "document",
            "paragraphs": [
                {
                    "context": context,
                    "qas": [
                        {
                            "question": question,
                            "id": f"{hash(question)}",  # Generate a unique ID based on the question text
                            "answers": [
                                {
                                    "text": context,
                                    "answer_start": context.find(context),
                                }
                            ],
                            "is_answer_absent": False,  # Assuming answers are always present
                        }
                    ],
                }
            ],
        }

    def digest_document_to_squad(self, document):
        """Processes a document into multiple SQuAD-formatted QA pairs."""
        sentences = self.split_into_sentences(document)
        squad_data = []

        for sentence in sentences:
            for question in self.generate_questions(sentence):
                qa_entry = self.create_squad_entry(question, sentence)
                squad_data.append(qa_entry)
                self.questions_generated.append(
                    (question, sentence)
                )  # Store for later use

        return squad_data

    def digest_documents_to_squad(self, documents):
        """Processes multiple documents into SQuAD format."""
        all_squad_data = []
        for document in documents:
            all_squad_data.extend(self.digest_document_to_squad(document))
        return all_squad_data


class QuestionAnsweringSupervisor:
    def __init__(self, logger=None):
        self.configs = {}
        self.mentor_model_name = "valhalla/t5-small-qa-qg-hl"
        self.mentor_model = pipeline(
            "text2text-generation",
            model=self.mentor_model_name,
            tokenizer=T5Tokenizer.from_pretrained(self.mentor_model_name),
        )
        self.questions_generated: typing.List[QAPair] = []
        self.logger = logger or logging.getLogger(__name__)

    @cached_property
    def nlp_model(self):
        with contextlib.suppress(Exception):
            return spacy.load("en_core_web_sm")
        spacy_download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

    def split_into_sentences(self, text):
        return [sent.text.strip() for sent in self.nlp_model(text).sents]

    def generate_questions(self, text):
        for res in self.mentor_model(f"generate questions: {text}"):
            yield res["generated_text"]

    def digest_document(self, document):
        sentences = self.split_into_sentences(document)
        for sentence in sentences:
            for question in self.generate_questions(sentence):
                self.questions_generated.append(
                    QAPair(
                        question,
                        sentence,
                    )
                )
        return len(self.questions_generated)

    def digest_documents(self, documents):
        return sum([self.digest_document(document) for document in documents])

    def fine_tune(self, epochs=10, batch_size=4):
        input_text = [
            f"question: {qa.question} context: {qa.answer}"
            for qa in self.questions_generated
        ]
        target_text = [qa.answer for qa in self.questions_generated]

        input_encoded = self.tokenizer(
            input_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        target_encoded = self.tokenizer(
            target_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        data_set = torch.utils.data.TensorDataset(
            input_encoded.input_ids,
            target_encoded.input_ids,
        )
        data_provider = torch.utils.data.DataLoader(
            data_set,
            batch_size=batch_size,
        )

        # Fine-tune the model
        optimizer = torch.optim.AdamW(self.trained_model.parameters(), lr=5e-5)
        self.trained_model.train()
        for epoch in range(epochs):
            for batch in data_provider:
                optimizer.zero_grad()
                input_ids, target_ids = batch
                outputs = self.trained_model(
                    input_ids=input_ids,
                    labels=target_ids,
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            self.logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    def train(
        self,
        *documents,
        epochs=10,
        batch_size=4,
    ):
        self.configs["epochs"] = epochs
        self.documents = documents

        self.trained_model_name = "t5-small"
        self.trained_model = T5ForConditionalGeneration.from_pretrained(
            self.trained_model_name
        )
        self.tokenizer = T5Tokenizer.from_pretrained(self.trained_model_name)

        self.digest_documents(documents)
        self.fine_tune(epochs=epochs, batch_size=batch_size)

        return Model(
            model=self.trained_model,
            tokenizer=self.tokenizer,
            qa_data=self.questions_generated,
        )
