import json
import hashlib
import redis
import time


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

    # =========================================================
    # SESSION MANAGEMENT
    # =========================================================

    def register_session(self, session_id: str, title: str):
        """Registers a new session with a title and timestamp."""
        timestamp = time.time()
        # Add to sorted set of sessions, score is timestamp
        self.redis.zadd("chat:sessions", {session_id: timestamp})
        # Store title
        self.redis.set(f"chat:{session_id}:title", title)

    def get_all_sessions(self):
        """Returns all sessions sorted by most recent first."""
        session_ids = self.redis.zrevrange("chat:sessions", 0, -1)
        sessions = []
        for sid in session_ids:
            title = self.redis.get(f"chat:{sid}:title") or "New Chat"
            timestamp = self.redis.zscore("chat:sessions", sid)
            sessions.append({
                "id": sid,
                "title": title,
                "timestamp": timestamp
            })
        return sessions

    # =========================================================
    # QUESTION KEY
    # =========================================================

    def _question_key(self, question: str, session_id: str = "default") -> str:
        """
        Converts a question into a stable Redis key for a specific session.
        """

        normalized_question = " ".join(question.lower().split())

        question_hash = hashlib.sha256(
            normalized_question.encode("utf-8")
        ).hexdigest()

        return f"chat:{session_id}:cache:{question_hash}"

    # =========================================================
    # CONVERSATION MEMORY
    # =========================================================

    def add_message(self, role: str, content: str, session_id: str = "default"):
        """
        Stores one message in the conversation history of a session.
        """

        message = {
            "role": role,
            "content": content,
        }

        self.redis.rpush(
            f"chat:{session_id}:history",
            json.dumps(message)
        )

    def get_history(self, session_id: str = "default"):
        """
        Returns the complete conversation history for a session.
        """

        messages = self.redis.lrange(
            f"chat:{session_id}:history",
            0,
            -1
        )

        return [
            json.loads(message)
            for message in messages
        ]

    def clear_history(self, session_id: str = "default"):
        """
        Deletes the current conversation history for a session.
        """

        self.redis.delete(f"chat:{session_id}:history")

    # =========================================================
    # ANSWER CACHE
    # =========================================================

    def cache_answer(self, question: str, response: dict, session_id: str = "default"):
        """
        Stores the complete RAG response in Redis for a session.
        """

        key = self._question_key(question, session_id)

        self.redis.set(
            key,
            json.dumps(response)
        )

    def get_cached_answer(self, question: str, session_id: str = "default"):
        """
        Returns the complete cached response for a session.
        """

        key = self._question_key(question, session_id)

        cached_response = self.redis.get(key)

        if cached_response is None:
            return None

        return json.loads(cached_response)

    # =========================================================
    # CACHE MANAGEMENT
    # =========================================================

    def clear_cache(self, session_id: str = "default"):
        """
        Deletes all cached question-answer responses for a session.
        """

        keys = self.redis.keys(
            f"chat:{session_id}:cache:*"
        )

        if keys:
            self.redis.delete(*keys)