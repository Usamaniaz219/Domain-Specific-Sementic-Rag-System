# import logging
# import os
# from typing import List, Dict, Any, Optional
# from tenacity import retry, stop_after_attempt, wait_exponential
# from dotenv import load_dotenv

# from config.settings import settings
# from config.constants import LLMProvider
# from utils.logger import get_logger
# from utils.cache import get_cache, set_cache
# import torch
# import hashlib
# import json

# logger = get_logger(__name__)

# load_dotenv()

# class LLMGenerator:
#     """Handles answer generation using various LLM providers"""
    
#     def __init__(self):
#         self.provider = settings.LLM_PROVIDER
#         self.model_name = settings.LLM_MODEL
#         self.max_tokens = settings.MAX_TOKENS
#         self.temperature = settings.TEMPERATURE
        
#         # Initialize based on provider
#         if self.provider == LLMProvider.OPENAI:
#             self._init_openai()
#         elif self.provider == LLMProvider.ANTHROPIC:
#             self._init_anthropic()
#         elif self.provider == LLMProvider.HUGGINGFACE:
#             self._init_huggingface()
#         elif self.provider == LLMProvider.BEDROCK:
#             self._init_bedrock()
#         else:
#             raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
#         logger.info(f"LLM generator initialized with {self.provider}:{self.model_name}")
    
#     def _init_openai(self):
#         """Initialize OpenAI client"""
#         try:
#             import openai
#             self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#         except ImportError:
#             logger.error("openai package is required for OpenAI provider")
#             raise
    
#     def _init_anthropic(self):
#         """Initialize Anthropic client"""
#         try:
#             import anthropic
#             self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
#         except ImportError:
#             logger.error("anthropic package is required for Anthropic provider")
#             raise
    
#     def _init_huggingface(self):
#         """Initialize HuggingFace client"""
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
#             self.pipeline = pipeline(
#                 "text-generation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=0 if torch.cuda.is_available() else -1
#             )
#         except ImportError:
#             logger.error("transformers and torch are required for HuggingFace provider")
#             raise
    
#     def _init_bedrock(self):
#         """Initialize AWS Bedrock client"""
#         try:
#             import boto3
#             self.client = boto3.client('bedrock-runtime', region_name=os.getenv("AWS_REGION"))
#         except ImportError:
#             logger.error("boto3 is required for AWS Bedrock provider")
#             raise
    
#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
#     def generate_answer(self, question: str, context: str, 
#                        conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
#         """Generate answer based on question and context"""
#         # Check cache first
#         cache_key = f"llm_answer:{hashlib.md5(f'{question}_{context}'.encode()).hexdigest()}"
#         cached = get_cache(cache_key)
#         if cached:
#             return cached
        
#         # Prepare prompt based on provider
#         if self.provider == LLMProvider.OPENAI:
#             response = self._generate_openai(question, context, conversation_history)
#         elif self.provider == LLMProvider.ANTHROPIC:
#             response = self._generate_anthropic(question, context, conversation_history)
#         elif self.provider == LLMProvider.HUGGINGFACE:
#             response = self._generate_huggingface(question, context, conversation_history)
#         elif self.provider == LLMProvider.BEDROCK:
#             response = self._generate_bedrock(question, context, conversation_history)
        
#         # Cache the result
#         set_cache(cache_key, response, settings.CACHE_TTL)
        
#         return response
    
#     def _generate_openai(self, question: str, context: str, 
#                         conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
#         """Generate answer using OpenAI"""
#         messages = [
#             {
#                 "role": "system",
#                 "content": self._create_system_prompt()
#             }
#         ]
        
#         # Add conversation history if provided
#         if conversation_history:
#             messages.extend(conversation_history)
        
#         # Add current context and question
#         messages.append({
#             "role": "user",
#             "content": f"Context: {context}\n\nQuestion: {question}"
#         })
        
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model_name,
#                 messages=messages,
#                 max_tokens=self.max_tokens,
#                 temperature=self.temperature
#             )
            
#             return {
#                 "answer": response.choices[0].message.content,
#                 "usage": {
#                     "prompt_tokens": response.usage.prompt_tokens,
#                     "completion_tokens": response.usage.completion_tokens,
#                     "total_tokens": response.usage.total_tokens
#                 },
#                 "model": self.model_name
#             }
#         except Exception as e:
#             logger.error(f"OpenAI API error: {str(e)}")
#             raise
    
#     def _generate_anthropic(self, question: str, context: str, 
#                            conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
#         """Generate answer using Anthropic Claude"""
#         # Prepare message history
#         message_history = ""
#         if conversation_history:
#             for msg in conversation_history:
#                 role = "Human" if msg["role"] == "user" else "Assistant"
#                 message_history += f"\n\n{role}: {msg['content']}"
        
#         prompt = f"{self._create_system_prompt()}{message_history}\n\nHuman: Context: {context}\n\nQuestion: {question}\n\nAssistant:"
        
#         try:
#             response = self.client.messages.create(
#                 model=self.model_name,
#                 max_tokens=self.max_tokens,
#                 temperature=self.temperature,
#                 messages=[{"role": "user", "content": prompt}]
#             )
            
#             return {
#                 "answer": response.content[0].text,
#                 "usage": {
#                     "input_tokens": response.usage.input_tokens,
#                     "output_tokens": response.usage.output_tokens
#                 },
#                 "model": self.model_name
#             }
#         except Exception as e:
#             logger.error(f"Anthropic API error: {str(e)}")
#             raise
    
#     def _generate_huggingface(self, question: str, context: str, 
#                              conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
#         """Generate answer using local HuggingFace model"""
#         prompt = self._create_prompt(question, context, conversation_history)
        
#         try:
#             outputs = self.pipeline(
#                 prompt,
#                 max_new_tokens=self.max_tokens,
#                 temperature=self.temperature,
#                 do_sample=True,
#                 return_full_text=False
#             )
            
#             return {
#                 "answer": outputs[0]['generated_text'],
#                 "usage": {
#                     "prompt_tokens": len(self.tokenizer.encode(prompt)),
#                     "completion_tokens": len(self.tokenizer.encode(outputs[0]['generated_text']))
#                 },
#                 "model": self.model_name
#             }
#         except Exception as e:
#             logger.error(f"HuggingFace generation error: {str(e)}")
#             raise
    
#     def _generate_bedrock(self, question: str, context: str, 
#                          conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
#         """Generate answer using AWS Bedrock"""
#         prompt = self._create_prompt(question, context, conversation_history)
        
#         try:
#             if "claude" in self.model_name.lower():
#                 # Anthropic Claude format
#                 body = json.dumps({
#                     "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
#                     "max_tokens_to_sample": self.max_tokens,
#                     "temperature": self.temperature
#                 })
#                 response = self.client.invoke_model(
#                     modelId=self.model_name,
#                     body=body
#                 )
#                 response_body = json.loads(response['body'].read())
#                 answer = response_body['completion']
#             else:
#                 # Other models
#                 body = json.dumps({
#                     "prompt": prompt,
#                     "maxTokens": self.max_tokens,
#                     "temperature": self.temperature
#                 })
#                 response = self.client.invoke_model(
#                     modelId=self.model_name,
#                     body=body
#                 )
#                 response_body = json.loads(response['body'].read())
#                 answer = response_body['generations'][0]['text']
            
#             return {
#                 "answer": answer,
#                 "model": self.model_name
#             }
#         except Exception as e:
#             logger.error(f"Bedrock API error: {str(e)}")
#             raise
    
#     def _create_system_prompt(self) -> str:
#         """Create system prompt for the LLM"""
#         return """You are an expert assistant that answers questions based strictly on the provided context.
#         Follow these rules:
#         1. Answer the question using only information from the provided context
#         2. If the answer cannot be found in the context, say "I cannot find an answer in the provided documents"
#         3. Be concise and accurate
#         4. Cite specific sections or documents when possible
#         5. If the question is ambiguous, ask for clarification
#         6. Format your response clearly with proper paragraphs and bullet points when appropriate"""
    
#     def _create_prompt(self, question: str, context: str, 
#                       conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
#         """Create a combined prompt for non-chat models"""
#         prompt = f"{self._create_system_prompt()}\n\n"
        
#         if conversation_history:
#             for msg in conversation_history:
#                 role = "User" if msg["role"] == "user" else "Assistant"
#                 prompt += f"{role}: {msg['content']}\n\n"
        
#         prompt += f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
#         return prompt
    
#     def batch_generate(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
#         """Batch generate answers for multiple questions"""
#         if len(questions) != len(contexts):
#             raise ValueError("Questions and contexts must have the same length")
        
#         results = []
#         for question, context in zip(questions, contexts):
#             try:
#                 result = self.generate_answer(question, context)
#                 results.append(result)
#             except Exception as e:
#                 logger.error(f"Error generating answer for question: {question[:50]}...: {str(e)}")
#                 results.append({
#                     "answer": "Sorry, I encountered an error while generating an answer.",
#                     "error": str(e)
#                 })
        
#         return results




import logging
import os
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


from config.settings import settings
from config.constants import LLMProvider
from utils.logger import get_logger
from utils.cache import get_cache, set_cache
import torch
import hashlib
import json
import random

logger = get_logger(__name__)

load_dotenv()

class LLMGenerator:
    """Handles answer generation using various LLM providers"""
    
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        print("self provider",self.provider)
        self.model_name = settings.LLM_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE

        self._init_gemini()
        
        
        # # FORCE MOCK MODE if no valid API key is available
        # if self.provider == LLMProvider.GEMINI and not os.getenv("GEMINI_API_KEY"):
        #     logger.warning("OPENAI_API_KEY not found. Switching to mock mode.")
        #     # self.provider = "gemini"
        #     # self.model_name = "mock"

        # # print("########################################")
        # # print("self provider",self.provider)

    
        # # elif self.provider == LLMProvider.GEMINI and not os.getenv("GEMINI_API_KEY"):
        # #     logger.warning("GEMINI_API_KEY not found. Switching to mock mode.")
        # #     self.provider = "mock"
        
        # # Initialize based on provider
        # if self.provider == LLMProvider.GEMINI:
        #     self._init_gemini()
        # elif self.provider == LLMProvider.ANTHROPIC:
        #     self._init_anthropic()
        # elif self.provider == LLMProvider.HUGGINGFACE:
        #     self._init_huggingface()
        # elif self.provider == LLMProvider.BEDROCK:
        #     self._init_bedrock()
       
        # if self.provider == LLMProvider.OPENAI:
        #     self._init_openai()

        # elif self.provider == "mock":
        #     self._init_mock()
        # else:
        #     raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        logger.info(f"LLM generator initialized with {self.provider}:{self.model_name}")
    
    def _init_mock(self):
        """Initialize mock LLM for testing without API key"""
        logger.info("Using mock LLM mode - responses will be simulated")
        # No initialization needed for mock mode
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.client = openai.OpenAI(api_key=api_key)
            
            # Test the connection
            try:
                self.client.models.list()
            except Exception as e:
                logger.error(f"OpenAI API test failed: {e}")
                raise ValueError(f"OpenAI API connection failed: {e}")
                
        except ImportError:
            logger.error("openai package is required for OpenAI provider")
            raise

    def _init_gemini(self):
        """Initialize Google Gemini client"""
        try:
            import google.generativeai as genai
            
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set")
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel(self.model_name)
            
            logger.info(f"Gemini client initialized with model: {self.model_name}")
            
        except ImportError:
            logger.error("google-generativeai package is required for Gemini provider")
            raise
        except Exception as e:
            logger.error(f"Gemini initialization failed: {str(e)}")
            raise

    def _generate_gemini(self, question: str, context: str, 
                    conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate answer using Google Gemini"""
        try:
            # Prepare the prompt
            prompt = self._create_gemini_prompt(question, context, conversation_history)
            
            # Generate response
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature
                ),
                safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }
            )
            
            # Extract the response text
            answer = response.text
            
            # Estimate token usage (Gemini doesn't provide exact counts in response)
            prompt_tokens = len(prompt.split())  # Rough estimate
            completion_tokens = len(answer.split())  # Rough estimate
            
            return {
                "answer": answer,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "model": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {str(e)}")
            raise
    

    def _generate_openai(self, question: str, context: str, 
                        conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate answer using OpenAI"""
        messages = [
            {
                "role": "system",
                "content": self._create_system_prompt()
            }
        ]
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add current context and question
        messages.append({
            "role": "user",
            "content": f"Context: {context}\n\nQuestion: {question}"
        })
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            return {
                "answer": response.choices[0].message.content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise

    def _create_gemini_prompt(self, question: str, context: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Create a prompt for Gemini"""
        system_prompt = """You are an expert assistant that answers questions based strictly on the provided context.
        Follow these rules:
        1. Answer the question using only information from the provided context
        2. If the answer cannot be found in the context, say "I cannot find an answer in the provided documents"
        3. Be concise and accurate
        4. Cite specific sections or documents when possible
        5. Format your response clearly with proper paragraphs and bullet points when appropriate"""
        
        prompt = f"{system_prompt}\n\n"
        
        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n\n"
        
        prompt += f"Context: {context}\n\n"
        prompt += f"Question: {question}\n\n"
        prompt += "Answer:"
        
        return prompt
    
    def _generate_anthropic(self, question: str, context: str, 
                           conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate answer using Anthropic Claude"""
        # Prepare message history
        message_history = ""
        if conversation_history:
            for msg in conversation_history:
                role = "Human" if msg["role"] == "user" else "Assistant"
                message_history += f"\n\n{role}: {msg['content']}"
        
        prompt = f"{self._create_system_prompt()}{message_history}\n\nHuman: Context: {context}\n\nQuestion: {question}\n\nAssistant:"
        
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return {
                "answer": response.content[0].text,
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"Anthropic API error: {str(e)}")
            raise

    def _generate_bedrock(self, question: str, context: str, 
                         conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate answer using AWS Bedrock"""
        prompt = self._create_prompt(question, context, conversation_history)
        
        try:
            if "claude" in self.model_name.lower():
                # Anthropic Claude format
                body = json.dumps({
                    "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
                    "max_tokens_to_sample": self.max_tokens,
                    "temperature": self.temperature
                })
                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=body
                )
                response_body = json.loads(response['body'].read())
                answer = response_body['completion']
            else:
                # Other models
                body = json.dumps({
                    "prompt": prompt,
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature
                })
                response = self.client.invoke_model(
                    modelId=self.model_name,
                    body=body
                )
                response_body = json.loads(response['body'].read())
                answer = response_body['generations'][0]['text']
            
            return {
                "answer": answer,
                "model": self.model_name
            }
        except Exception as e:
            logger.error(f"Bedrock API error: {str(e)}")
            raise
   
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def generate_answer(self, question: str, context: str, 
                       conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Generate answer based on question and context"""
        # Check cache first
        cache_key = f"llm_answer:{hashlib.md5(f'{question}_{context}'.encode()).hexdigest()}"
        cached = get_cache(cache_key)
        if cached:
            return cached
        
        # Use mock mode if provider is mock
        if self.provider == "mock":
            response = self._generate_mock(question, context)
            set_cache(cache_key, response, settings.CACHE_TTL)
            return response
        
        # Prepare prompt based on provider
        if self.provider == LLMProvider.OPENAI:
            response = self._generate_openai(question, context, conversation_history)
        elif self.provider == LLMProvider.ANTHROPIC:
            response = self._generate_anthropic(question, context, conversation_history)
        elif self.provider == LLMProvider.HUGGINGFACE:
            response = self._generate_huggingface(question, context, conversation_history)
        elif self.provider == LLMProvider.BEDROCK:
            response = self._generate_bedrock(question, context, conversation_history)
        elif self.provider == LLMProvider.GEMINI:
            response = self._generate_gemini(question, context, conversation_history)
        else:
            # Fallback to mock mode for any unknown provider
            response = self._generate_mock(question, context)
        
        # Cache the result
        set_cache(cache_key, response, settings.CACHE_TTL)
        
        return response
    
    def _generate_mock(self, question: str, context: str) -> Dict[str, Any]:
        """Generate mock answer for testing without API calls"""
        # Simple mock responses based on question content
        mock_responses = [
            f"Based on the provided documentation, {question.lower()} would typically involve considerations around data retention policies and security measures.",
            f"The documentation indicates that {question.lower()} requires compliance with industry-specific regulations and standards.",
            f"According to the context, {question.lower()} should follow established protocols for information management and protection.",
            f"The materials suggest that {question.lower()} needs to balance operational efficiency with regulatory requirements.",
            f"Based on the available information, {question.lower()} would be addressed through a combination of technical and administrative controls."
        ]
        
        # Select a random response or use a context-aware one
        if context and len(context) > 0:
            # If we have context, create a more informed mock response
            words = context.split()[:20]  # Take first 20 words of context
            summary = " ".join(words) + "..."
            
            answer = f"Based on the documentation which discusses {summary}, the answer to '{question}' would involve these considerations. [MOCK RESPONSE - Real API key needed for actual answers]"
        else:
            # Generic mock response
            answer = random.choice(mock_responses) + " [MOCK RESPONSE - Real API key needed for actual answers]"
        
        return {
            "answer": answer,
            "usage": {
                "prompt_tokens": len(question + context),
                "completion_tokens": random.randint(50, 150),
                "total_tokens": len(question + context) + random.randint(50, 150)
            },
            "model": "mock"
        }
    
    # ... keep the rest of your _generate_* methods the same ...
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM"""
        return """You are an expert assistant that answers questions based strictly on the provided context.
        Follow these rules:
        1. Answer the question using only information from the provided context
        2. If the answer cannot be found in the context, say "I cannot find an answer in the provided documents"
        3. Be concise and accurate
        4. Cite specific sections or documents when possible
        5. If the question is ambiguous, ask for clarification
        6. Format your response clearly with proper paragraphs and bullet points when appropriate"""
    
    # ... keep the rest of your methods the same ...
    def _create_prompt(self, question: str, context: str, 
                      conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Create a combined prompt for non-chat models"""
        prompt = f"{self._create_system_prompt()}\n\n"
        
        if conversation_history:
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n\n"
        
        prompt += f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        return prompt
    
    def batch_generate(self, questions: List[str], contexts: List[str]) -> List[Dict[str, Any]]:
        """Batch generate answers for multiple questions"""
        if len(questions) != len(contexts):
            raise ValueError("Questions and contexts must have the same length")
        
        results = []
        for question, context in zip(questions, contexts):
            try:
                result = self.generate_answer(question, context)
                results.append(result)
            except Exception as e:
                logger.error(f"Error generating answer for question: {question[:50]}...: {str(e)}")
                results.append({
                    "answer": "Sorry, I encountered an error while generating an answer.",
                    "error": str(e)
                })
        
        return results