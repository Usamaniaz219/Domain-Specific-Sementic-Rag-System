from enum import Enum

class FileType(str, Enum):
    TXT = "text"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    MD = "markdown"

class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"

class LLMProvider(str, Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    BEDROCK = "bedrock"
    MOCK = "mock"