# app/services/prompts.py

SYSTEM_PROMPT = (
    "You are an expert meeting/video summarizer. "
    "You create concise, faithful summaries of spoken content."
)

CHUNK_SUMMARY_USER_PROMPT_TEMPLATE = (
    "This is segment {idx} of {num_chunks} from a video transcript.\n"
    "Your task: Write a concise summary of the key points of this segment.\n\n"
    "Transcript segment:\n{chunk}\n\n"
    "Guidelines:\n"
    "- Focus only on important information.\n"
    "- Preserve factual accuracy.\n"
    "- Use 3–7 bullet points or short paragraphs.\n"
)

FINAL_SUMMARY_USER_PROMPT_TEMPLATE = (
    "You are given summaries of different segments of a single YouTube video.\n"
    "Combine them into ONE final summary that a busy person can read quickly "
    "and still understand the main message of the video.\n\n"
    "Segment summaries:\n"
    "{merged}\n"
    "Guidelines:\n"
    "- Group related ideas.\n"
    "- Avoid redundancy.\n"
    "- Keep it within 3–6 short paragraphs or 5–10 bullets.\n"
)
