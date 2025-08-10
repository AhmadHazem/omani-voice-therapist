from Agent import Therpaist
from Agent import GPT4o
from Agent import ClaudeSonnet3_7
from Agent import Notify

from logger import logger
from sentence_transformers import SentenceTransformer, util

import numpy as np
import asyncio

therapist_GPT = Therpaist(GPT4o)
therapist_claude = Therpaist(ClaudeSonnet3_7)
model_comparer = SentenceTransformer("all-MiniLM-L6-v2")

def validate_response(gpt_analysis : dict, claude_analysis : dict) -> float:
    model_comparer = SentenceTransformer("all-MiniLM-L6-v2")
    similarity_scores = []
    logger.info(f"GPT Therapist analysis:{gpt_analysis}")
    logger.info(f"Claude Therapist analysis: {claude_analysis}")
    for key in gpt_analysis.keys():
        embeddings = model_comparer.encode([str(gpt_analysis[key]), str(claude_analysis[key])], show_progress_bar= False)
        similarity_scores.append(util.cos_sim(embeddings[0], embeddings[1]))
    similarity_score = float(np.average(similarity_scores))
    return np.average(similarity_score)


async def therapist_chat(prompt: str, timeout: float = 10.0, max_retries: int = 3, session_id: str = "abc123", emergency_contacts=[]):
    try:
        retries = 1
        final_analysis = None

        logger.info("Phase 1: Sending Prompt to the AI Therapists......")
        gpt_task = asyncio.create_task(therapist_GPT.invoke_step1(prompt))
        claude_task = asyncio.create_task(therapist_claude.invoke_step1(prompt))

        # Wait up to timeout for both tasks to complete
        done, pending = await asyncio.wait(
            [gpt_task, claude_task],
            timeout=timeout,
            return_when=asyncio.ALL_COMPLETED
        )

        if len(done) == 2:
            logger.info("Phase 1: Analysis Finished For Both Therapists")
            gpt_result = await gpt_task
            claude_result = await claude_task
            score = validate_response(gpt_result, claude_result)
            logger.info(f"Similarity Score is : {score}")
            if score > 0.5 or (not gpt_result['requires_analysis'] and not claude_result['requires_analysis']) or retries == max_retries:
                final_analysis = gpt_result
            else:
                retries += 1

        elif gpt_task in done:
            logger.info("Phase 1: Analysis Finished For Only GPT")
            for task in pending:
                task.cancel()
            final_analysis = await gpt_task

        elif claude_task in done:
            logger.info("Phase 1: Analysis Finished For Only Claude")
            for task in pending:
                task.cancel()
            final_analysis = await claude_task

        logger.info("Phase 2: Adapting the response")
        if final_analysis['severity'] == "CRIT":
            logger.info("User Prompt situation is critical; Notifying Authorities .......")
            Notify(emergency_contacts)

        tokens = await therapist_GPT.invoke_step2(final_analysis, session_id=session_id)
        for token in tokens:
            yield token

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        yield "عفوا لقد حدث خطأ حاول من جديد"
