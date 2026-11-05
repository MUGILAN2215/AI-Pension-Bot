from dotenv import load_dotenv
load_dotenv()

from langsmith import evaluate
from src.api.service import RagService

agent = RagService.from_env().agent


def target(inputs):
    result = agent.ask(inputs["question"], history=[])
    return {
        "route": result.route,
        "rewritten": result.rewritten_question,
        "answer": result.answer.text,
        "calculator": result.calculator,
        "steps": result.steps,
    }


def route_correct(outputs, reference_outputs):
    return outputs["route"] == reference_outputs["route"]


evaluate(
    target,
    data="pension_test_50",
    evaluators=[route_correct],
    experiment_prefix="full-answers-client",
    max_concurrency=1,
)