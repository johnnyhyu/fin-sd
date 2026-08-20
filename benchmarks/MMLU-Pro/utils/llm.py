from utils.config import LLMConfig
from tqdm.asyncio import tqdm_asyncio
import asyncio
import math
import random
from openai import AsyncOpenAI, RateLimitError, APIStatusError
import aiolimiter
from loguru import logger

class LLM:

    def __init__(self, config: LLMConfig):
        self.config = config
        # Every model is served via an OpenAI-compatible endpoint: OpenRouter for
        # hosted models (including Anthropic/Gemini/DeepSeek via their OpenRouter
        # slugs) and a local vLLM server for self-hosted models. config validation
        # (utils/config.py) guarantees base_url is one of those two.
        # max_retries=0: our own _async_generate_with_limiter owns retry/backoff,
        # so disable the SDK's hidden internal retries (they'd compound our loop
        # and hide long stalls). timeout bounds each attempt at the httpx layer;
        # asyncio.wait_for below is the authoritative hard cap in case a keepalive
        # stream keeps resetting the httpx read timeout.
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.request_timeout,
            max_retries=0,
        )
        # The rate limiter and concurrency semaphore both bind to the event loop
        # they are first used in. Because batch_generate() spins up a *new* loop
        # via asyncio.run() on every call, a single LLM object reused across calls
        # (e.g. retriever query optimization then inference, or inference then
        # evaluation) would otherwise carry a semaphore/limiter bound to an
        # already-closed loop and raise "bound to a different event loop". Build
        # them lazily, keyed by the running loop, so each asyncio.run() gets a
        # fresh pair.
        self._limiters: dict[int, aiolimiter.AsyncLimiter] = {}
        self._semaphores: dict[int, asyncio.Semaphore] = {}

    @property
    def limiter(self) -> aiolimiter.AsyncLimiter:
        loop_id = id(asyncio.get_running_loop())
        limiter = self._limiters.get(loop_id)
        if limiter is None:
            limiter = self._limiters[loop_id] = aiolimiter.AsyncLimiter(1, 60 / self.config.rpm)
        return limiter

    @property
    def semaphore(self) -> asyncio.Semaphore:
        loop_id = id(asyncio.get_running_loop())
        semaphore = self._semaphores.get(loop_id)
        if semaphore is None:
            semaphore = self._semaphores[loop_id] = asyncio.Semaphore(self.config.max_concurrency)
        return semaphore

    def __getattr__(self, name):
        if hasattr(self.config, name):
            return getattr(self.config, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def apply_chat_template(self, system_inputs: list[str], user_inputs: list[str]):
        prompts = []
        for system_input, user_input in zip(system_inputs, user_inputs):
            if self.config.support_system_role:
                messages = [{"role": "system", "content": system_input}, {"role": "user", "content": user_input} ]
            else:
                messages = [ {"role": "user", "content": system_input + "\n" + user_input}]
            prompts.append(messages)
        return prompts
    
    @staticmethod
    def _mean_token_entropy(logprobs):
        """Mean per-token predictive entropy (in nats) computed over the returned
        `top_logprobs` distribution at each generated position. The truncated
        top-k distribution is renormalized to sum to 1 before computing entropy.
        Returns None when the provider did not return token logprobs."""
        if not logprobs:
            return None
        content = logprobs.get("content") if isinstance(logprobs, dict) else None
        if not content:
            return None
        entropies = []
        for tok in content:
            tops = tok.get("top_logprobs") or []
            if not tops:
                continue
            probs = [math.exp(t["logprob"]) for t in tops]
            z = sum(probs)
            if z <= 0:
                continue
            probs = [p / z for p in probs]
            entropies.append(-sum(p * math.log(p) for p in probs if p > 0))
        if not entropies:
            return None
        return sum(entropies) / len(entropies)

    def _format_openai_response(self, response):
        if response:
            response = response.model_dump()
            content = response["choices"][0]["message"]["content"]
            completion_tokens = response["usage"]["completion_tokens"]
            reasoning_content = response["choices"][0]["message"].get("reasoning_content", None)
            mean_entropy = self._mean_token_entropy(response["choices"][0].get("logprobs"))
            return { "output": content, "reasoning_content": reasoning_content, "completion_tokens": completion_tokens, "mean_entropy": mean_entropy, "raw_response": response }
        else: return { "output": None, "reasoning_content": None, "completion_tokens": None, "mean_entropy": None, "raw_response": None }

    async def _async_openai_generate(self, prompt: list[dict]):
        response = await asyncio.wait_for(
            self.client.chat.completions.create(
                model=self.config.model_id,
                messages=prompt,
                **self.config.sampling_args,
            ),
            timeout=self.config.request_timeout,
        )
        return self._format_openai_response(response)

    @property
    def _async_generate(self):
        return self._async_openai_generate

    @staticmethod
    def _retry_after(error: Exception) -> float | None:
        """Extract the Retry-After header (seconds) from a rate-limit error, if present."""
        response = getattr(error, "response", None)
        if response is None:
            return None
        retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except ValueError:
            return None

    async def _async_generate_with_limiter(self, prompt: list[dict]):
        async with self.semaphore:
            async with self.limiter:
                for i in range(self.config.max_retries):
                    try:
                        return await self._async_generate(prompt)
                    except Exception as e:
                        # Exponential backoff with jitter; honor Retry-After on 429s.
                        backoff = min(2 ** i, 60) + random.uniform(0, 1)
                        if isinstance(e, (RateLimitError, APIStatusError)):
                            retry_after = self._retry_after(e)
                            if retry_after is not None:
                                backoff = retry_after + random.uniform(0, 1)
                        logger.error(
                            f"Error generating response: {e} | retry {i + 1} of "
                            f"{self.config.max_retries}, backing off {backoff:.1f}s"
                        )
                        if i < self.config.max_retries - 1:
                            await asyncio.sleep(backoff)
                        continue
        logger.error(f"Failed to generate response after {self.config.max_retries} retries")
        return self._format_openai_response(None)
    
    async def async_generate(self, prompt: list[dict]):
        return await self._async_generate_with_limiter(prompt)
    
    async def async_batch_generate(self, prompts: list[list[dict]], desc: str = "Generating"):
        async_responses = [ self.async_generate(prompt) for prompt in prompts ]
        responses = await tqdm_asyncio.gather(*async_responses, desc=desc)
        return responses

    def batch_generate(self, prompts: list[list[dict]], desc: str = "Generating"):
        return asyncio.run(self.async_batch_generate(prompts, desc))

    async def async_batch_generate_samples(
        self, prompts: list[list[dict]], num_samples: int, desc: str = "Sampling"
    ):
        """Draw `num_samples` independent completions per prompt. Returns a list
        parallel to `prompts`, each entry being a list of `num_samples` responses."""
        flat = [prompt for prompt in prompts for _ in range(num_samples)]
        responses = await self.async_batch_generate(flat, desc=desc)
        return [
            responses[i * num_samples:(i + 1) * num_samples]
            for i in range(len(prompts))
        ]

    def batch_generate_samples(
        self, prompts: list[list[dict]], num_samples: int, desc: str = "Sampling"
    ):
        return asyncio.run(self.async_batch_generate_samples(prompts, num_samples, desc))

