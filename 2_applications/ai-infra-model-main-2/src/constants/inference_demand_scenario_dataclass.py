from dataclasses import dataclass
from typing import Dict, List

TOKENS_PER_WORD = 1.3
TOKENS_PER_SENTENCE = TOKENS_PER_WORD * 15
TOKENS_PER_PARAGRAPH = TOKENS_PER_SENTENCE * 5
TOKENS_PER_PAGE = TOKENS_PER_PARAGRAPH * 5

@dataclass
class InferenceDemandPerUser:
    """Represents inference demand pattern for a specific model type"""
    category: str  # 'agent', 'chat', 'small', or 'tiny'
    num_prompts: float
    avg_input_tokens: float
    avg_output_tokens: float
    model_used: str

    def daily_tokens(self):
        """Calculate total tokens used daily with this model"""
        return (self.avg_input_tokens + self.avg_output_tokens) * self.num_prompts

@dataclass
class InferenceDemandScenario:
    """Represents a complete scenario configuration for inference demand patterns"""
    name: str
    patterns: List[InferenceDemandPerUser]
    num_users: float  # Number of users in millions
    monthly_subscription_price: float  # Monthly subscription price per user in USD

# TODO ChatGPT power users, perplexity users, etc.

# Define the default scenario
DEFAULT_SCENARIO = InferenceDemandScenario(
    name="Chat Experience",
    patterns=[
        InferenceDemandPerUser(
            category="agent",
            num_prompts=5,
            avg_input_tokens=10000,
            avg_output_tokens=1000,
            model_used="openai/o1"
        ),
        InferenceDemandPerUser(
            category="chat",
            num_prompts=15,
            avg_input_tokens=10 * TOKENS_PER_PARAGRAPH,
            avg_output_tokens=3 * TOKENS_PER_PARAGRAPH,
            model_used="anthropic/claude-3.5-sonnet"
        ),
        InferenceDemandPerUser(
            category="small",
            num_prompts=25,
            avg_input_tokens=500,
            avg_output_tokens=50,
            model_used="anthropic/claude-3.5-haiku"
        ),
        InferenceDemandPerUser(
            category="tiny",
            num_prompts=40,
            avg_input_tokens=500,
            avg_output_tokens=50,
            model_used="meta-llama/llama-3.2-1b-instruct"
        )
    ],
    num_users=20.0,  # 20 million users
    monthly_subscription_price=20.0  # $20 per month
)

# Define the coding agent scenario
CODING_AGENT_SCENARIO = InferenceDemandScenario(
    name="Coding Agent",
    patterns=[
        InferenceDemandPerUser(
            category="agent",
            num_prompts=15,
            avg_input_tokens=12 * TOKENS_PER_PAGE,
            avg_output_tokens=3 * TOKENS_PER_PAGE,
            model_used="anthropic/claude-3.7-sonnet:thinking"
        ),
        InferenceDemandPerUser(
            category="chat",
            num_prompts=50,
            avg_input_tokens=8 * TOKENS_PER_PARAGRAPH,
            avg_output_tokens=3 * TOKENS_PER_PARAGRAPH,
            model_used="anthropic/claude-3.7-sonnet"
        ),
        InferenceDemandPerUser(
            category="small",
            num_prompts=100,
            avg_input_tokens=500,
            avg_output_tokens=100,
            model_used="anthropic/claude-3.5-haiku"
        ),
        InferenceDemandPerUser(
            category="tiny",
            num_prompts=500,
            avg_input_tokens=500,
            avg_output_tokens=50,
            model_used="meta-llama/llama-3.2-1b-instruct"
        )
    ],
    num_users=7.5,  # 7.5 million users
    monthly_subscription_price=20.0  # $50 per month
)

# Define all available scenarios
SCENARIOS: Dict[str, InferenceDemandScenario] = {
    "Default": DEFAULT_SCENARIO,
    "Coding Agent": CODING_AGENT_SCENARIO
}
