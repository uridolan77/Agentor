import logging
import json
import os
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import tiktoken

logger = logging.getLogger(__name__)



class TokenCounter:
    """Count tokens for different models."""

    def __init__(self):
        """Initialize the token counter."""
        self.encoders = {}

    def _get_encoder(self, model: str):
        """Get the encoder for a model.

        Args:
            model: The model name

        Returns:
            The encoder
        """
        if model not in self.encoders:
            try:
                self.encoders[model] = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to cl100k_base for unknown models
                self.encoders[model] = tiktoken.get_encoding("cl100k_base")

        return self.encoders[model]

    def count_tokens(self, text: str, model: str = "gpt-3.5-turbo") -> int:
        """Count the number of tokens in a text.

        Args:
            text: The text to count tokens for
            model: The model to use for counting

        Returns:
            The number of tokens
        """
        encoder = self._get_encoder(model)
        return len(encoder.encode(text))


@dataclass
class CostData:
    """Data about the cost of an LLM request."""
    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    timestamp: float


class CostTracker:
    """Track the cost of LLM requests."""

    def __init__(self, pricing_file: Optional[str] = None):
        """Initialize the cost tracker.

        Args:
            pricing_file: Path to a JSON file with pricing information
        """
        self.pricing = self._load_pricing(pricing_file)
        self.costs: List[CostData] = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_by_model: Dict[str, int] = {}
        self.tokens_by_model: Dict[str, Dict[str, int]] = {}
        self.cost_by_model: Dict[str, float] = {}

    def _load_pricing(self, pricing_file: Optional[str]) -> Dict[str, Dict[str, float]]:
        """Load pricing information from a file.

        Args:
            pricing_file: Path to a JSON file with pricing information

        Returns:
            Pricing information
        """
        # Default pricing (per 1000 tokens)
        default_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
            "text-embedding-ada-002": {"input": 0.0001, "output": 0.0},
            "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
            "text-embedding-3-large": {"input": 0.00013, "output": 0.0},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125}
        }

        # Load custom pricing if provided
        if pricing_file and os.path.exists(pricing_file):
            try:
                with open(pricing_file, "r") as f:
                    custom_pricing = json.load(f)

                # Merge custom pricing with default pricing
                for model, prices in custom_pricing.items():
                    default_pricing[model] = prices

            except Exception as e:
                logger.error(f"Error loading pricing file: {e}")

        return default_pricing

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> Dict[str, Any]:
        """Calculate the cost of an LLM request.

        Args:
            model: The model used
            input_tokens: The number of input tokens
            output_tokens: The number of output tokens

        Returns:
            Cost information
        """
        # Get pricing for the model
        model_pricing = self.pricing.get(model, {"input": 0.01, "output": 0.01})

        # Calculate costs
        input_cost = (input_tokens / 1000) * model_pricing["input"]
        output_cost = (output_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost

        # Create cost data
        cost_data = CostData(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=total_cost,
            timestamp=time.time()
        )

        # Track the cost
        self.costs.append(cost_data)
        self.total_cost += total_cost
        self.total_tokens += input_tokens + output_tokens

        # Update the model-specific stats
        self.requests_by_model[model] = self.requests_by_model.get(model, 0) + 1

        if model not in self.tokens_by_model:
            self.tokens_by_model[model] = {"input": 0, "output": 0}

        self.tokens_by_model[model]["input"] += input_tokens
        self.tokens_by_model[model]["output"] += output_tokens

        self.cost_by_model[model] = self.cost_by_model.get(model, 0.0) + total_cost

        # Return cost information
        return {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }

    def get_total_cost(self) -> float:
        """Get the total cost of all tracked requests.

        Returns:
            The total cost
        """
        return self.total_cost

    def get_stats(self) -> Dict[str, Any]:
        """Get the current cost statistics.

        Returns:
            A dictionary with the cost statistics
        """
        return {
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "requests_by_model": self.requests_by_model,
            "tokens_by_model": self.tokens_by_model,
            "cost_by_model": self.cost_by_model
        }

    def get_cost_summary(self) -> Dict[str, Any]:
        """Get a summary of costs.

        Returns:
            Cost summary
        """
        if not self.costs:
            return {
                "total_cost": 0.0,
                "total_tokens": 0,
                "models": {}
            }

        # Calculate summary
        total_tokens = sum(cost.input_tokens + cost.output_tokens for cost in self.costs)
        model_costs = {}

        for cost in self.costs:
            if cost.model not in model_costs:
                model_costs[cost.model] = {
                    "count": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_cost": 0.0
                }

            model_costs[cost.model]["count"] += 1
            model_costs[cost.model]["input_tokens"] += cost.input_tokens
            model_costs[cost.model]["output_tokens"] += cost.output_tokens
            model_costs[cost.model]["total_cost"] += cost.total_cost

        return {
            "total_cost": self.total_cost,
            "total_tokens": total_tokens,
            "models": model_costs
        }

    def reset(self):
        """Reset the cost tracker."""
        self.costs = []
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_by_model = {}
        self.tokens_by_model = {}
        self.cost_by_model = {}


class CostAwareLLM:
    """A wrapper around an LLM that tracks costs."""

    def __init__(self, llm, cost_tracker: CostTracker, token_counter: TokenCounter):
        """Initialize the cost-aware LLM.

        Args:
            llm: The LLM to wrap
            cost_tracker: The cost tracker
            token_counter: The token counter
        """
        self.llm = llm
        self.cost_tracker = cost_tracker
        self.token_counter = token_counter

    async def generate(self, request):
        """Generate a response, tracking the cost.

        Args:
            request: The LLM request

        Returns:
            The LLM response
        """
        # Count the input tokens
        input_tokens = self.token_counter.count_tokens(request.prompt, request.model)

        # Generate the response
        response = await self.llm.generate(request)

        # Count the output tokens
        output_tokens = self.token_counter.count_tokens(response.text, request.model)

        # Calculate the cost
        cost = self.cost_tracker.calculate_cost(
            model=request.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens
        )

        # Add the cost to the response metadata
        if response.metadata is None:
            response.metadata = {}

        response.metadata["cost"] = cost

        # Log the cost
        logger.info(
            f"LLM request cost: ${cost['total_cost']:.6f} "
            f"({input_tokens} input tokens, {output_tokens} output tokens)"
        )

        return response

    async def generate_batch(self, requests):
        """Generate responses for multiple requests, tracking costs.

        Args:
            requests: The LLM requests

        Returns:
            The LLM responses
        """
        # Check if the LLM supports batch generation
        if hasattr(self.llm, "generate_batch"):
            # Generate responses
            responses = await self.llm.generate_batch(requests)

            # Track costs for each response
            for i, response in enumerate(responses):
                # Skip failed requests
                if isinstance(response, Exception):
                    continue

                # Count tokens
                input_tokens = self.token_counter.count_tokens(requests[i].prompt, requests[i].model)
                output_tokens = self.token_counter.count_tokens(response.text, requests[i].model)

                # Calculate cost
                cost = self.cost_tracker.calculate_cost(
                    model=requests[i].model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens
                )

                # Add cost to metadata
                if response.metadata is None:
                    response.metadata = {}

                response.metadata["cost"] = cost

            return responses
        else:
            # Fall back to individual generation
            return await asyncio.gather(*[self.generate(request) for request in requests])


class BudgetLimitExceeded(Exception):
    """Exception raised when a budget limit is exceeded."""

    def __init__(self, budget: float, current_cost: float):
        """Initialize the exception.

        Args:
            budget: The budget limit
            current_cost: The current cost
        """
        self.budget = budget
        self.current_cost = current_cost
        super().__init__(f"Budget limit of ${budget:.2f} exceeded (current cost: ${current_cost:.2f})")


class BudgetLimitedLLM:
    """A wrapper around an LLM that enforces a budget limit."""

    def __init__(self, llm, budget: float):
        """Initialize the budget-limited LLM.

        Args:
            llm: The LLM to wrap
            budget: The budget limit
        """
        self.llm = llm
        self.budget = budget
        self.cost_tracker = CostTracker()

    async def generate(self, request):
        """Generate a response, enforcing the budget limit.

        Args:
            request: The LLM request

        Returns:
            The LLM response

        Raises:
            BudgetLimitExceeded: If the budget limit would be exceeded
        """
        # Check if we have a cost tracker in the LLM
        if not hasattr(self.llm, "cost_tracker"):
            raise ValueError("The wrapped LLM must have a cost_tracker attribute")

        # Get the current cost
        current_cost = self.llm.cost_tracker.get_total_cost()

        # Check if we're already over budget
        if current_cost >= self.budget:
            raise BudgetLimitExceeded(self.budget, current_cost)

        # Generate the response
        response = await self.llm.generate(request)

        # Get the updated cost
        updated_cost = self.llm.cost_tracker.get_total_cost()

        # Check if we're now over budget
        if updated_cost > self.budget:
            logger.warning(
                f"Budget limit of ${self.budget:.2f} exceeded after request "
                f"(current cost: ${updated_cost:.2f})"
            )

        return response


# Import asyncio at the end to avoid circular imports
import asyncio
