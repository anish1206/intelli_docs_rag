import json
import hashlib
import redis


class ChatMemory:
    def __init__(
        self,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
    ):
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            decode_responses=True,
        )

        # Used to store the conversation history
        self.history_key = "chat:history"

        # Prefix used for cached question-answer responses
        self.cache_prefix = "chat:cache:"

    # =========================================================
    # QUESTION KEY
    # =========================================================

    def _question_key(self, question: str) -> str:
        """
        Converts a question into a stable Redis key.

        The same question will always generate the same key.
        """

        normalized_question = " ".join(question.lower().split())

        question_hash = hashlib.sha256(
            normalized_question.encode("utf-8")
        ).hexdigest()

        return f"{self.cache_prefix}{question_hash}"

    # =========================================================
    # CONVERSATION MEMORY
    # =========================================================

    def add_message(self, role: str, content: str):
        """
        Stores one message in the conversation history.
        """

        message = {
            "role": role,
            "content": content,
        }

        self.redis.rpush(
            self.history_key,
            json.dumps(message)
        )

    def get_history(self):
        """
        Returns the complete conversation history.
        """

        messages = self.redis.lrange(
            self.history_key,
            0,
            -1
        )

        return [
            json.loads(message)
            for message in messages
        ]

    def clear_history(self):
        """
        Deletes the current conversation history.
        """

        self.redis.delete(self.history_key)

    # =========================================================
    # ANSWER CACHE
    # =========================================================

    def cache_answer(self, question: str, response: dict):
        """
        Stores the complete RAG response in Redis.

        Example response:

        {
            "question": "...",
            "answer": "...",
            "sources": [...]
        }
        """

        key = self._question_key(question)

        self.redis.set(
            key,
            json.dumps(response)
        )

    def get_cached_answer(self, question: str):
        """
        Returns the complete cached response.

        Returns:
            dict  -> cache hit
            None  -> cache miss
        """

        key = self._question_key(question)

        cached_response = self.redis.get(key)

        if cached_response is None:
            return None

        return json.loads(cached_response)

    # =========================================================
    # CACHE MANAGEMENT
    # =========================================================

    def clear_cache(self):
        """
        Deletes all cached question-answer responses.

        Only deletes keys starting with chat:cache:
        """

        keys = self.redis.keys(
            f"{self.cache_prefix}*"
        )

        if keys:
            self.redis.delete(*keys)