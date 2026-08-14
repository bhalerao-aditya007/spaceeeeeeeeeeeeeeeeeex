"""
Redis Client with In-Memory Fallback Broker.
Enables full multi-agent functionality and dashboard pub/sub
even when an external Redis server process is not running.
"""

import os
import time
import json
import queue
import threading
import redis


class InMemoryPubSub:
    def __init__(self, broker):
        self.broker = broker
        self.q = queue.Queue()
        self.channels = set()

    def subscribe(self, *channels):
        for ch in channels:
            ch_str = ch.decode('utf-8') if isinstance(ch, bytes) else str(ch)
            self.channels.add(ch_str)
            self.broker._add_subscriber(ch_str, self.q)

    def listen(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            yield item

    def close(self):
        for ch in self.channels:
            self.broker._remove_subscriber(ch, self.q)
        self.q.put(None)


class InMemoryRedisBroker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.subscribers = {}
                cls._instance.sub_lock = threading.Lock()
            return cls._instance

    def _add_subscriber(self, channel, q):
        with self.sub_lock:
            if channel not in self.subscribers:
                self.subscribers[channel] = set()
            self.subscribers[channel].add(q)

    def _remove_subscriber(self, channel, q):
        with self.sub_lock:
            if channel in self.subscribers:
                self.subscribers[channel].discard(q)

    def ping(self):
        return True

    def publish(self, channel, message):
        ch_str = channel.decode('utf-8') if isinstance(channel, bytes) else str(channel)
        msg_str = message.decode('utf-8') if isinstance(message, bytes) else (message if isinstance(message, str) else json.dumps(message))

        data = {
            "type": "message",
            "channel": ch_str.encode('utf-8'),
            "data": msg_str.encode('utf-8')
        }
        with self.sub_lock:
            queues = list(self.subscribers.get(ch_str, []))
        for q in queues:
            q.put(data)
        return len(queues)

    def pubsub(self):
        return InMemoryPubSub(self)

    @classmethod
    def from_url(cls, *args, **kwargs):
        return cls()


def get_redis_client(host="localhost", port=6379, db=0, url=None, **kwargs):
    """
    Attempts to connect to a real Redis server.
    If unreachable, returns an InMemoryRedisBroker instance.
    """
    redis_url = url or os.environ.get("REDIS_URL")
    try:
        if redis_url:
            r = redis.Redis.from_url(redis_url, socket_timeout=0.2, socket_connect_timeout=0.2, **kwargs)
        else:
            r = redis.Redis(host=host, port=port, db=db, socket_timeout=0.2, socket_connect_timeout=0.2, **kwargs)
        r.ping()
        return r
    except Exception:
        return InMemoryRedisBroker()
