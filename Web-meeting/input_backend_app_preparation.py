import pandas as pd
import sqlite3
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datetime
import numpy as np
import time
import accelerate
from transformers import AutoModel, AutoTokenizer
import streamlit as st
import collections

model_name = "cl-tohoku/bert-base-japanese-v2"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def make_llm_output(messages, max_new_tokens=2048):
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

class MeetingAssistant:
    def __init__(self, db_path = 'meeting_topics.db'):
        self.db_path = db_path
        self._create_table()

    def _create_table(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS topics (
                meeting_id TEXT PRIMARY KEY,
                topic_text TEXT NOT NULL,
                keywords TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_topic(self, meeting_id, topic_text, keywords=None):
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            keywords_str = ','.join(keywords) if keywords else ''
            cursor.execute("INSERT OR REPLACE INTO topics (meeting_id, topic_text,keywords)VALUES(?, ?, ?)",
                            (meeting_id, topic_text, keywords_str))
            conn.commit()
            conn.close()

    def get_topic(self, meeting_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT topic_text, keywords FROM topics WHERE meeting_id = ?", (meeting_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            topic_text, keywords_str = result
            keywords = keywords_str.split(',') if keywords_str else []
            return {"topic_text": topic_text, "keywords": keywords}
        return None

topic_manager = MeetingAssistant()
topic_manager.add_topic("meeting_001", "新製品Xのマーケティング戦略について議論", ["新製品X", "マーケティング", "戦略"])

class TextEmbedder:
    def __init__(self, model_name="cl-tohoku/bert-base-japanese-whole-word-masking"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_embedding(self,text):
        inputs = self.tokenizer(text, return_tensors ="pt", truncation=True, padding=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)


        attention_mask = inputs['attention_mask']
        expanded_mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * expanded_mask, 1)
        sum_mask = torch.clamp(expanded_mask.sum(1), min=1e-9)
        mean_embedding =sum_embeddings / sum_mask

        return mean_embedding.squeeze(0).cpu().numpy()

    def calculate_similarity(self, embedding1, embedding2):
        from sklearn.metrics.pairwise import cosine_similarity
        return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

class TopicDeviationDetector:
    def __init__(self, topic_embedding: np.ndarray, topic_info: dict, similarity_threshold: float =0.7, consecutive_deviations_needed: int =2, cooldown_period_seconds: int =10):
        self.topic_embedding = topic_embedding
        self.topic_info = topic_info
        self.similarity_threshold = similarity_threshold
        self.consecutive_deviations_needed = consecutive_deviations_needed
        self.consecutive_deviations_count = 0
        self.last_notification_time = 0
        self.cooldown_period_seconds = cooldown_period_seconds
        self.text_embedder = TextEmbedder()

    def process_utterance(self, current_utterance_text):
        current_time = time.time()

        if current_time - self.last_notification_time < self.cooldown_period_seconds:
            # 類似度スコアも返すように変更
            # （クールダウン中でも類似度自体は計算可能なので、ここでは仮の0を返しています。
            # 必要であれば、ここで改めて類似度を計算して返してください。）
            utterance_embedding = self.text_embedder.get_embedding(current_utterance_text)
            similarity = self.text_embedder.calculate_similarity(self.topic_embedding, utterance_embedding)
            return False, "クールダウン中", similarity

        utterance_embedding = self.text_embedder.get_embedding(current_utterance_text)
        similarity = self.text_embedder.calculate_similarity(self.topic_embedding, utterance_embedding)

        print(f"今のお話: '{current_utterance_text}'")
        print(f"テーマとの似ている度合い: {similarity:.4f} (ボーダーライン: {self.similarity_threshold})")

        if similarity < self.similarity_threshold:
            self.consecutive_deviations_count += 1
            if self.consecutive_deviations_count >= self.consecutive_deviations_needed:
                self.last_notification_time = current_time
                self.consecutive_deviations_count = 0
                return True, f"話が逸れています！ 今のお話は'{current_utterance_text[:30]}...'ですが、テーマは'{self.topic_info['topic_text']}'です！", similarity
            else:
                return False, f"お題がら派生しています。 (あと {self.consecutive_deviations_needed - self.consecutive_deviations_count}回話が逸れた場合は注意します)", similarity
        else:
            self.consecutive_deviations_count = 0
            return False, "", similarity
